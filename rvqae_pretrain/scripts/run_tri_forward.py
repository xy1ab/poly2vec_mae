"""Run one single-sample RVQAE forward pass from triangle shards with training-style visualization."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CURRENT_DIR.parent
    _REPO_ROOT = _PROJECT_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib

    ensure_cuda_runtime_libs = importlib.import_module(
        "rvqae_pretrain.scripts.runtime_bootstrap"
    ).ensure_cuda_runtime_libs
else:
    from .runtime_bootstrap import ensure_cuda_runtime_libs


def _inject_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def _format_float_vector(value: torch.Tensor, digits: int = 4) -> str:
    import torch

    flat = value.detach().float().flatten().cpu().tolist()
    return "[" + ", ".join(f"{item:.{digits}f}" for item in flat) + "]"


def _format_int_vector(value: torch.Tensor) -> str:
    import torch

    flat = value.detach().float().flatten().cpu().tolist()
    return "[" + ", ".join(str(int(round(item))) for item in flat) + "]"


def _resolve_sample_by_gid(helpers, tri_dir: str | Path, target_gid: int) -> tuple[np.ndarray, np.ndarray, int]:
    triplets = helpers.resolve_tri_meta_gid_triplets(tri_dir)
    matched_samples: list[tuple[np.ndarray, np.ndarray, int]] = []

    for triplet in triplets:
        tri_samples = helpers.load_torch_list(triplet.tri_path)
        meta_samples = helpers.load_torch_list(triplet.meta_path)
        gid_samples = helpers.load_torch_list(triplet.gid_path)
        if len(tri_samples) != len(meta_samples) or len(tri_samples) != len(gid_samples):
            raise ValueError(
                "Triangle/meta/gid sample count mismatch for "
                f"{triplet.tri_path.name}, {triplet.meta_path.name}, {triplet.gid_path.name}: "
                f"{len(tri_samples)}, {len(meta_samples)}, {len(gid_samples)}"
            )

        for tri_sample, meta_sample, gid_sample in zip(tri_samples, meta_samples, gid_samples):
            gid_value = int(gid_sample.item() if hasattr(gid_sample, "item") else gid_sample)
            if gid_value == int(target_gid):
                matched_samples.append(
                    (
                        np.asarray(tri_sample, dtype=np.float32),
                        np.asarray(meta_sample, dtype=np.float32),
                        gid_value,
                    )
                )

    if not matched_samples:
        raise KeyError(f"Failed to locate gid={target_gid} under tri_dir={Path(tri_dir).expanduser().resolve()}")
    if len(matched_samples) > 1:
        raise ValueError(f"Found duplicated gid={target_gid} across triangle shards.")
    return matched_samples[0]


def main() -> None:
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()
    import torch

    if __package__ in {None, ""}:
        import importlib

        helpers = importlib.import_module("rvqae_pretrain.scripts.batch_infer_common")
        pipeline_module = importlib.import_module("rvqae_pretrain.src.engine.pipeline")
        trainer_module = importlib.import_module("rvqae_pretrain.src.engine.trainer")
    else:
        from . import batch_infer_common as helpers
        from ..src.engine import pipeline as pipeline_module
        from ..src.engine import trainer as trainer_module

    parser = argparse.ArgumentParser(description="Run one single-sample RVQAE forward pass by gid.")
    parser.add_argument("--tri_dir", type=str, required=True, help="Directory containing triangle/meta/gid shard triplets.")
    parser.add_argument("--gid", type=int, required=True, help="Unique gid to locate one sample.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save one `.pt` result and one viz `.png`.")
    parser.add_argument("--nicft", type=int, default=256, help="Fallback ICFT output size. <=0 disables ICFT/exported spatial maps.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=0,
        help="Spatial resolution (meter). If >0, per-sample nicft is derived from metadata.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Runtime device type, usually `cuda`.")
    parser.add_argument("--precision", type=str, default="fp32", help="Inference precision, usually `fp32` for debugging.")
    args = parser.parse_args()

    if args.nicft < 0:
        raise ValueError(f"`nicft` must be >= 0, got {args.nicft}")
    if args.resolution < 0:
        raise ValueError(f"`resolution` must be >= 0, got {args.resolution}")

    checkpoint_path, config_path = helpers.resolve_model_paths(args.model_dir)
    triangles, meta, gid_value = _resolve_sample_by_gid(helpers, args.tri_dir, args.gid)

    pipeline = pipeline_module.PolyRvqAePipeline(
        weight_path=str(checkpoint_path),
        config_path=str(config_path),
        device=str(args.device),
        precision=str(args.precision),
    )

    imgs = pipeline.triangles_to_images([triangles]).float()
    with torch.no_grad():
        outputs = pipeline.model(imgs, use_vq=True)

    recon_imgs = outputs.recon_imgs.float()
    indices = outputs.indices.long()
    if indices.ndim == 3:
        indices = indices.unsqueeze(1)
    indices_u16 = helpers.to_uint16_indices(indices, context="tri_forward_single")

    img_orig = imgs[0].detach().cpu()
    img_masked = img_orig.clone()
    img_recon = recon_imgs[0].detach().cpu()

    meta_tensor = torch.as_tensor(meta, dtype=torch.float32)
    if int(args.resolution) > 0:
        sample_nicft = int(np.ceil(float(meta_tensor[2].item()) * 118000.0 / float(args.resolution)))
    else:
        sample_nicft = int(args.nicft)

    spatial_gt = None
    spatial_icft_orig = None
    spatial_icft_recon = None
    real_recon = imag_recon = None
    if sample_nicft > 0:
        spatial_gt = trainer_module.rasterize_tris_to_grid(
            torch.as_tensor(triangles, dtype=torch.float32).cpu(),
            height=sample_nicft,
            width=sample_nicft,
        )

        orig_mag = imgs[:, 0:1]
        orig_phase = torch.atan2(imgs[:, 2:3], imgs[:, 1:2])
        real_orig, imag_orig = trainer_module.mag_phase_to_real_imag(orig_mag, orig_phase)
        spatial_icft_orig = pipeline.codec.icft_2d(real_orig.squeeze(1), imag_orig.squeeze(1))[0].detach().cpu()

        mag_recon = recon_imgs[:, 0:1]
        phase_recon = torch.atan2(recon_imgs[:, 2:3], recon_imgs[:, 1:2])
        real_recon, imag_recon = trainer_module.mag_phase_to_real_imag(mag_recon, phase_recon)
        spatial_icft_recon = pipeline.codec.icft_2d(real_recon.squeeze(1), imag_recon.squeeze(1))[0].detach().cpu()
    else:
        spatial_gt = np.zeros((1, 1), dtype=np.float32)
        spatial_icft_orig = torch.zeros((1, 1), dtype=torch.float32)
        spatial_icft_recon = torch.zeros((1, 1), dtype=torch.float32)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = output_root / f"tri_forward_gid_{gid_value}_{timestamp}.png"
    pt_path = output_root / f"tri_forward_gid_{gid_value}_{timestamp}.pt"

    viz_dir = output_root / f".viz_tmp_{gid_value}_{timestamp}"
    trainer_module.plot_reconstruction(
        img_orig=img_orig,
        img_masked=img_masked,
        img_recon=img_recon,
        spatial_gt=spatial_gt,
        spatial_icft_orig=spatial_icft_orig.squeeze(),
        spatial_icft_recon=spatial_icft_recon.squeeze(),
        epoch=gid_value,
        save_dir=viz_dir,
    )
    generated_png = viz_dir / f"recon_epoch_{gid_value}.png"
    if not generated_png.is_file():
        raise FileNotFoundError(f"Failed to create visualization image: {generated_png}")
    generated_png.replace(png_path)
    viz_dir.rmdir()

    result_payload = {
        "gid": int(gid_value),
        "meta": meta_tensor.cpu(),
        "indices": indices_u16[0].cpu(),
        "input_img": img_orig.cpu(),
        "recon_img": img_recon.cpu(),
        "triangles": torch.as_tensor(triangles, dtype=torch.float32).cpu(),
        "perplexity": outputs.perplexity.detach().float().cpu(),
        "active_codes": outputs.active_codes.detach().float().cpu(),
        "nicft": int(sample_nicft),
        "resolution": int(args.resolution),
        "png_path": str(png_path),
    }
    if real_recon is not None and imag_recon is not None:
        result_payload["real"] = real_recon[0].detach().float().cpu()
        result_payload["imag"] = imag_recon[0].detach().float().cpu()
    if sample_nicft > 0:
        result_payload["icft_orig"] = spatial_icft_orig.squeeze().float().cpu()
        result_payload["icft_recon"] = spatial_icft_recon.squeeze().float().cpu()
        result_payload["rec_label"] = torch.as_tensor(spatial_gt, dtype=torch.float32).cpu()

    torch.save(result_payload, pt_path)

    print(f"[INFO] gid              : {gid_value}")
    print(f"[INFO] meta shape       : {tuple(meta_tensor.shape)}")
    print(f"[INFO] indices shape    : {tuple(indices_u16[0].shape)}")
    print(f"[INFO] input img shape  : {tuple(img_orig.shape)}")
    print(f"[INFO] recon img shape  : {tuple(img_recon.shape)}")
    if "real" in result_payload and "imag" in result_payload:
        print(f"[INFO] real shape       : {tuple(result_payload['real'].shape)}")
        print(f"[INFO] imag shape       : {tuple(result_payload['imag'].shape)}")
    if "icft_recon" in result_payload:
        print(f"[INFO] icft shape       : {tuple(result_payload['icft_recon'].shape)}")
    print(f"[INFO] nicft            : {sample_nicft}")
    print(f"[INFO] resolution       : {int(args.resolution)}")
    print(f"[INFO] perplexity       : {_format_float_vector(outputs.perplexity)}")
    print(f"[INFO] active_codes     : {_format_int_vector(outputs.active_codes)}")
    print(f"[INFO] Saved result     : {pt_path}")
    print(f"[INFO] Saved viz        : {png_path}")


if __name__ == "__main__":
    main()
