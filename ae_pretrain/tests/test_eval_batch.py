"""Batch-evaluate triangulated samples into downstream training records.

This script reads one directory of sharded triangle `.pt` files, loads one
encoder checkpoint plus one full AE checkpoint from `--model_dir`, and writes
one output `.pt` file where each sample stores:

1) Original triangle tensor `[T, 3, 2]`
2) Encoder embedding vector `[D]`
3) AE decoder frequency-domain real map `[H, W]`
4) AE decoder frequency-domain imaginary map `[H, W]`

The script depends only on `ae_pretrain/src/*` modules and does not import any
launcher under `ae_pretrain/scripts`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


def _inject_repo_root() -> Path:
    """Inject repository root into `sys.path` for direct script execution."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


PROJECT_ROOT = _inject_repo_root()

import torch
from tqdm import tqdm

from ae_pretrain.src.datasets.geometry_polygon import pad_triangle_batch
from ae_pretrain.src.datasets.registry import get_geometry_codec
from ae_pretrain.src.datasets.sharded_pt_dataset import _ensure_numpy_float32
from ae_pretrain.src.models.factory import load_ae_model, load_pretrained_encoder
from ae_pretrain.src.utils.config import load_config_any
from ae_pretrain.src.utils.filesystem import ensure_dir
from ae_pretrain.src.utils.precision import autocast_context, normalize_precision
from ae_pretrain.src.utils.safe_load import register_numpy_safe_globals


def _default_device() -> str:
    """Resolve default inference device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_model_artifacts(model_dir: str) -> tuple[Path, Path, Path]:
    """Resolve encoder weight, AE weight, and config under one model directory."""
    base = Path(model_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {base}")

    search_roots = [base]
    for subdir in ("best", "ckpt"):
        candidate = base / subdir
        if candidate.is_dir():
            search_roots.append(candidate)

    config_candidates: list[Path] = []
    for root in search_roots:
        for path in [root / "config.yaml", root / "config.yml", root / "poly_ae_config.json", root / "poly_mae_config.json"]:
            if path.exists() and path not in config_candidates:
                config_candidates.append(path)

    ae_candidates: list[Path] = []
    for root in search_roots:
        for path in [
            root / "ae_best.pth",
            root / "autoencoder.pth",
            root / "mae_best.pth",
            root / "encoder_decoder.pth",
        ]:
            if path.exists() and path not in ae_candidates:
                ae_candidates.append(path)
        ae_candidates.extend(
            path for path in sorted(root.glob("ae_ckpt_*.pth")) if path not in ae_candidates
        )
        ae_candidates.extend(
            path for path in sorted(root.glob("mae_ckpt_*.pth")) if path not in ae_candidates
        )

    encoder_candidates: list[Path] = []
    for root in search_roots:
        for path in [
            root / "encoder_best.pth",
            root / "encoder.pth",
        ]:
            if path.exists() and path not in encoder_candidates:
                encoder_candidates.append(path)
        encoder_candidates.extend(
            path for path in sorted(root.glob("poly_encoder_epoch_*.pth")) if path not in encoder_candidates
        )

    if not config_candidates:
        raise FileNotFoundError(f"No config file found in model_dir: {base}")
    if not ae_candidates:
        raise FileNotFoundError(f"No AE checkpoint found in model_dir: {base}")
    if not encoder_candidates:
        raise FileNotFoundError(f"No encoder checkpoint found in model_dir: {base}")

    return encoder_candidates[0], ae_candidates[0], config_candidates[0]


def _resolve_input_shards(input_dir: str) -> list[Path]:
    """Resolve sorted shard `.pt` files from input directory."""
    base = Path(input_dir).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {base}")
    shard_paths = sorted(path for path in base.glob("*.pt") if path.is_file())
    if not shard_paths:
        raise FileNotFoundError(f"No shard .pt files found under input_dir: {base}")
    return shard_paths


def _build_output_path(project_root: Path, input_dir: str, output_path: str | None) -> Path:
    """Resolve output path from explicit CLI path or one timestamped default."""
    if output_path:
        return Path(output_path).expanduser().resolve()

    input_name = Path(input_dir).expanduser().resolve().name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "outputs" / "eval_batch" / f"{input_name}_eval_batch_{timestamp}.pt"


def _pred_patches_to_complex_maps(
    pred: torch.Tensor,
    img_height: int,
    img_width: int,
    patch_size: int,
    valid_h: int,
    valid_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert AE patch predictions into valid real/imaginary frequency maps."""
    batch_size = pred.shape[0]
    h_patch = img_height // patch_size
    w_patch = img_width // patch_size

    pred_img = pred.reshape(batch_size, h_patch, w_patch, 3, patch_size, patch_size)
    pred_img = torch.einsum("nhwcpq->nchpwq", pred_img).reshape(batch_size, 3, img_height, img_width)
    pred_valid = pred_img[:, :, :valid_h, :valid_w]

    mag_pred = pred_valid[:, 0, :, :]
    cos_pred = pred_valid[:, 1, :, :]
    sin_pred = pred_valid[:, 2, :, :]

    phase_pred = torch.atan2(sin_pred, cos_pred)
    raw_mag_pred = torch.expm1(mag_pred)
    real_part = raw_mag_pred * torch.cos(phase_pred)
    imag_part = raw_mag_pred * torch.sin(phase_pred)
    return real_part, imag_part


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch evaluation export."""
    parser = argparse.ArgumentParser(
        description="Build downstream samples from triangulated shards using encoder+AE inference."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing triangulated shard .pt files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing encoder/AE checkpoints and config.")
    parser.add_argument("--output_path", type=str, default="", help="Output .pt file path.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--device", type=str, default=_default_device(), help="Runtime device, e.g. cuda or cpu.")
    parser.add_argument("--precision", type=str, default="bf16", help="Runtime precision: fp32/bf16/fp16.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional debug cap on total exported samples; 0 means all.")
    return parser


def main() -> None:
    """CLI main entrypoint."""
    args = build_arg_parser().parse_args()
    args.precision = normalize_precision(args.precision)
    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.max_samples < 0:
        raise ValueError(f"`max_samples` must be >= 0, got {args.max_samples}")

    register_numpy_safe_globals()

    shard_paths = _resolve_input_shards(args.input_dir)
    encoder_weight, ae_weight, config_path = _resolve_model_artifacts(args.model_dir)
    output_path = _build_output_path(PROJECT_ROOT, args.input_dir, args.output_path or None)
    ensure_dir(output_path.parent)

    config = load_config_any(config_path)
    device = torch.device(args.device)
    codec = get_geometry_codec(str(config.get("geom_type", "polygon")).lower(), config, device=str(device))
    encoder = load_pretrained_encoder(
        weight_path=encoder_weight,
        config_path=config_path,
        device=device,
        precision=args.precision,
    )
    ae_model, runtime_config = load_ae_model(
        weight_path=ae_weight,
        config_path=config_path,
        device=device,
        precision=args.precision,
    )
    ae_model.eval()

    patch_size = int(runtime_config.get("patch_size", config.get("patch_size", 2)))
    valid_h = codec.converter.U.shape[0] - codec.converter.pad_h
    valid_w = codec.converter.U.shape[1] - codec.converter.pad_w

    output_samples: list[dict[str, torch.Tensor]] = []
    pending_tris = []
    exported_count = 0
    processed_shard_count = 0

    def flush_pending() -> None:
        """Run one inference batch and append output sample dicts."""
        nonlocal exported_count
        if not pending_tris:
            return

        batch_tris, lengths = pad_triangle_batch(pending_tris, device=device)
        with torch.no_grad():
            mag, phase = codec.cft_batch(batch_tris, lengths)
            imgs = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)

            with autocast_context(device, args.precision):
                encoder_features = encoder(imgs)
                pred = ae_model(imgs)

            embeddings = encoder_features[:, 0, :].float().cpu()
            pred = pred.float()
            real_maps, imag_maps = _pred_patches_to_complex_maps(
                pred=pred,
                img_height=int(imgs.shape[-2]),
                img_width=int(imgs.shape[-1]),
                patch_size=patch_size,
                valid_h=valid_h,
                valid_w=valid_w,
            )
            real_maps = real_maps.cpu()
            imag_maps = imag_maps.cpu()

        for index, tri_np in enumerate(pending_tris):
            output_samples.append(
                {
                    "triangles": torch.from_numpy(tri_np.astype("float32", copy=False)),
                    "embedding": embeddings[index],
                    "freq_real": real_maps[index],
                    "freq_imag": imag_maps[index],
                }
            )
            exported_count += 1

        pending_tris.clear()

    for shard_path in tqdm(shard_paths, desc="Reading shards"):
        processed_shard_count += 1
        shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
        if not isinstance(shard_data, list):
            raise TypeError(f"Shard file must store a Python list: {shard_path}")

        for sample in shard_data:
            tri_np = _ensure_numpy_float32(sample)
            pending_tris.append(tri_np)
            if len(pending_tris) >= args.batch_size:
                flush_pending()
            if args.max_samples > 0 and exported_count + len(pending_tris) >= args.max_samples:
                remaining = args.max_samples - exported_count
                if remaining < len(pending_tris):
                    pending_tris[:] = pending_tris[:remaining]
                flush_pending()
                break

        if args.max_samples > 0 and exported_count >= args.max_samples:
            break

    flush_pending()

    torch.save(output_samples, output_path)

    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] Shards read: {processed_shard_count}")
    print(f"[DONE] Sample count: {len(output_samples)}")
    if output_samples:
        print(f"[DONE] Embedding dim: {int(output_samples[0]['embedding'].numel())}")
        print(f"[DONE] Frequency map shape: {tuple(output_samples[0]['freq_real'].shape)}")
        print("[DONE] Sample schema: {'triangles':(T,3,2), 'embedding':(D), 'freq_real':(H,W), 'freq_imag':(H,W)}")
    else:
        print("[DONE] No samples exported.")


if __name__ == "__main__":
    main()
