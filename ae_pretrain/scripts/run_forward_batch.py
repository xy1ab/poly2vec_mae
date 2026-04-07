"""Batch forward-export launcher for triangulated polygon samples.

This script reads triangulated shard `.pt` files, runs one encoder checkpoint
plus one AE checkpoint from `--model_dir`, and writes one output `.pt` file.

The output payload is a dict with two top-level keys:
1) `metadata`: self-contained runtime metadata and codec configuration.
2) `samples`: list of per-sample dicts containing:
   - `sample_index`
   - `triangles`
   - `embedding`
   - `freq_real`
   - `freq_imag`
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any

from tqdm import tqdm

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CURRENT_DIR.parent
    _REPO_ROOT = _PROJECT_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib

    ensure_cuda_runtime_libs = importlib.import_module(
        "ae_pretrain.scripts.runtime_bootstrap"
    ).ensure_cuda_runtime_libs
else:
    from .runtime_bootstrap import ensure_cuda_runtime_libs


def _inject_repo_root() -> Path:
    """Inject repository root into `sys.path` for direct script execution."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def _resolve_user_path(path_str: str, project_root: Path) -> Path:
    """Resolve one user-provided path against cwd and project root."""
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (project_root / raw_path).resolve()


def _default_device() -> str:
    """Resolve default runtime device string."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _extract_last_int(text: str) -> int | None:
    """Extract the last integer substring from text, if present."""
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def _resolve_model_artifacts(model_dir: str) -> tuple[Path, Path, Path, bool]:
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
        for path in (root / "config.yaml", root / "config.yml", root / "poly_ae_config.json", root / "poly_mae_config.json"):
            if path.exists() and path not in config_candidates:
                config_candidates.append(path)

    ae_candidates: list[Path] = []
    for root in search_roots:
        candidates = [
            path
            for path in root.glob("*.pth")
            if path.is_file()
            and (
                "encoder_decoder" in path.name.lower()
                or "mae" in path.name.lower()
                or "autoencoder" in path.name.lower()
                or "ae_best" in path.name.lower()
            )
        ]
        for path in candidates:
            if path not in ae_candidates:
                ae_candidates.append(path)

    encoder_candidates: list[Path] = []
    for root in search_roots:
        for path in (root / "encoder.pth", root / "encoder_best.pth"):
            if path.exists() and path not in encoder_candidates:
                encoder_candidates.append(path)
        for path in sorted(root.glob("*encoder*.pth")):
            name_lower = path.name.lower()
            if "encoder_decoder" in name_lower:
                continue
            if path not in encoder_candidates:
                encoder_candidates.append(path)

    if not config_candidates:
        raise FileNotFoundError(f"No config file found in model_dir: {base}")
    if not ae_candidates:
        raise FileNotFoundError(
            "No AE checkpoint found in model_dir. Expected a `.pth` file whose name "
            "contains `ae` or `autoencoder` (legacy `mae` / `encoder_decoder` names are also accepted)."
        )

    def _ae_rank(path: Path) -> tuple[int, int, float, str]:
        name_lower = path.name.lower()
        if name_lower == "autoencoder.pth":
            priority = 0
        elif name_lower == "ae_best.pth":
            priority = 1
        elif name_lower == "encoder_decoder.pth":
            priority = 2
        elif "encoder_decoder" in name_lower:
            priority = 3
        elif name_lower == "mae_best.pth":
            priority = 4
        else:
            priority = 5
        suffix = _extract_last_int(path.stem)
        suffix_rank = -(suffix if suffix is not None else -1)
        mtime_rank = -float(path.stat().st_mtime)
        return (priority, suffix_rank, mtime_rank, name_lower)

    def _encoder_rank(path: Path) -> tuple[int, int, float, str]:
        name_lower = path.name.lower()
        if name_lower == "encoder.pth":
            priority = 0
        elif name_lower == "encoder_best.pth":
            priority = 1
        else:
            priority = 2
        suffix = _extract_last_int(path.stem)
        suffix_rank = -(suffix if suffix is not None else -1)
        mtime_rank = -float(path.stat().st_mtime)
        return (priority, suffix_rank, mtime_rank, name_lower)

    ae_candidates = sorted(ae_candidates, key=_ae_rank)
    encoder_candidates = sorted(encoder_candidates, key=_encoder_rank)

    ae_path = ae_candidates[0]
    if encoder_candidates:
        return encoder_candidates[0], ae_path, config_candidates[0], False
    return ae_path, ae_path, config_candidates[0], True


def _build_output_path(project_root: Path, data_dir: str, output_path: str | None) -> Path:
    """Resolve output path from explicit CLI path or one timestamped default."""
    if output_path:
        return Path(output_path).expanduser().resolve()

    data_name = Path(data_dir).expanduser().resolve().name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "outputs" / "forward_batch" / f"{data_name}_forward_batch_{timestamp}.pt"


def _reconstruct_complex_maps(
    pred,
    img_height: int,
    img_width: int,
    patch_size: int,
    valid_h: int,
    valid_w: int,
):
    """Convert AE patch predictions into reconstructed real/imaginary maps."""
    import torch

    batch_size = int(pred.shape[0])
    h_patch = img_height // patch_size
    w_patch = img_width // patch_size

    pred_img = pred.reshape(batch_size, h_patch, w_patch, 3, patch_size, patch_size)
    pred_img = torch.einsum("nhwcpq->nchpwq", pred_img).reshape(batch_size, 3, img_height, img_width)
    recon_valid = pred_img[:, :, :valid_h, :valid_w]

    mag_pred = recon_valid[:, 0, :, :]
    cos_pred = recon_valid[:, 1, :, :]
    sin_pred = recon_valid[:, 2, :, :]

    phase_pred = torch.atan2(sin_pred, cos_pred)
    raw_mag_pred = torch.expm1(mag_pred)
    real_part = raw_mag_pred * torch.cos(phase_pred)
    imag_part = raw_mag_pred * torch.sin(phase_pred)
    return real_part, imag_part


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch forward export."""
    parser = argparse.ArgumentParser(
        description="Batch forward triangulated polygon shards into embeddings and AE frequency maps."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing encoder/AE checkpoints and config.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing triangulated shard `.pt` files.")
    parser.add_argument("--output_path", type=str, default="", help="Output `.pt` file path.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--device", type=str, default=_default_device(), help="Runtime device, e.g. cuda or cpu.")
    parser.add_argument("--precision", type=str, default="bf16", help="Runtime precision: fp32/bf16/fp16.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional export cap; 0 means all samples.")
    return parser


def main() -> None:
    """CLI main entrypoint."""
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    import torch

    if __package__ in {None, ""}:
        import importlib

        pad_triangle_batch = importlib.import_module(
            "ae_pretrain.src.datasets.geometry_polygon"
        ).pad_triangle_batch
        get_geometry_codec = importlib.import_module(
            "ae_pretrain.src.datasets.registry"
        ).get_geometry_codec
        shard_io = importlib.import_module("ae_pretrain.src.datasets.shard_io")
        resolve_triangle_shard_paths = shard_io.resolve_triangle_shard_paths
        load_triangle_shard = shard_io.load_triangle_shard
        dataset_module = importlib.import_module(
            "ae_pretrain.src.datasets.sharded_pt_dataset"
        )
        _ensure_numpy_float32 = dataset_module._ensure_numpy_float32
        factory_module = importlib.import_module("ae_pretrain.src.models.factory")
        load_ae_model = factory_module.load_ae_model
        load_pretrained_encoder = factory_module.load_pretrained_encoder
        load_config_any = importlib.import_module(
            "ae_pretrain.src.utils.config"
        ).load_config_any
        ensure_dir = importlib.import_module(
            "ae_pretrain.src.utils.filesystem"
        ).ensure_dir
        precision_module = importlib.import_module("ae_pretrain.src.utils.precision")
        autocast_context = precision_module.autocast_context
        normalize_precision = precision_module.normalize_precision
    else:
        from ..src.datasets.geometry_polygon import pad_triangle_batch
        from ..src.datasets.registry import get_geometry_codec
        from ..src.datasets.shard_io import load_triangle_shard, resolve_triangle_shard_paths
        from ..src.datasets.sharded_pt_dataset import _ensure_numpy_float32
        from ..src.models.factory import load_ae_model, load_pretrained_encoder
        from ..src.utils.config import load_config_any
        from ..src.utils.filesystem import ensure_dir
        from ..src.utils.precision import autocast_context, normalize_precision

    args = build_arg_parser().parse_args()
    args.precision = normalize_precision(args.precision)
    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.max_samples < 0:
        raise ValueError(f"`max_samples` must be >= 0, got {args.max_samples}")

    args.model_dir = str(_resolve_user_path(args.model_dir, project_root))
    args.data_dir = str(_resolve_user_path(args.data_dir, project_root))
    output_path = _build_output_path(project_root, args.data_dir, args.output_path or None)
    ensure_dir(output_path.parent)

    requested_device = args.device or _default_device()
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, fallback to CPU for forward export.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    shard_paths = resolve_triangle_shard_paths(args.data_dir, warn_fn=lambda message: print(message))
    encoder_weight, ae_weight, config_path, encoder_fallback_to_mae = _resolve_model_artifacts(args.model_dir)

    config = load_config_any(config_path)
    geom_type = str(config.get("geom_type", "polygon")).lower()
    codec = get_geometry_codec(geom_type, config, device=str(device))

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
    encoder.eval()
    ae_model.eval()

    patch_size = int(runtime_config.get("patch_size", config.get("patch_size", 2)))
    valid_h = int(codec.converter.U.shape[0] - codec.converter.pad_h)
    valid_w = int(codec.converter.U.shape[1] - codec.converter.pad_w)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "encoder_weight": str(Path(encoder_weight).expanduser().resolve()),
        "ae_weight": str(Path(ae_weight).expanduser().resolve()),
        "encoder_fallback_to_mae": bool(encoder_fallback_to_mae),
        "precision": args.precision,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "max_samples": int(args.max_samples),
        "spatial_size": 256,
        "geom_type": geom_type,
        "config": dict(config),
        "runtime_config": dict(runtime_config),
        "patch_size": patch_size,
        "valid_freq_shape": (valid_h, valid_w),
        "shard_paths": [str(path) for path in shard_paths],
        "processed_shard_count": 0,
        "sample_count": 0,
    }

    output_samples: list[dict[str, Any]] = []
    pending_tris: list[Any] = []
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
            real_maps, imag_maps = _reconstruct_complex_maps(
                pred=pred,
                img_height=int(imgs.shape[-2]),
                img_width=int(imgs.shape[-1]),
                patch_size=patch_size,
                valid_h=valid_h,
                valid_w=valid_w,
            )
            real_maps = real_maps.cpu()
            imag_maps = imag_maps.cpu()

        start_index = exported_count + 1
        for offset, tri_np in enumerate(pending_tris):
            sample_index = start_index + offset
            output_samples.append(
                {
                    "sample_index": int(sample_index),
                    "triangles": torch.from_numpy(tri_np.astype("float32", copy=False)),
                    "embedding": embeddings[offset],
                    "freq_real": real_maps[offset],
                    "freq_imag": imag_maps[offset],
                }
            )

        exported_count += len(pending_tris)
        pending_tris.clear()

    for shard_path in tqdm(shard_paths, desc="Reading shards"):
        processed_shard_count += 1
        shard_data = load_triangle_shard(shard_path)
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

    metadata["processed_shard_count"] = int(processed_shard_count)
    metadata["sample_count"] = int(len(output_samples))
    if output_samples:
        metadata["embedding_dim"] = int(output_samples[0]["embedding"].numel())
        metadata["freq_map_shape"] = tuple(output_samples[0]["freq_real"].shape)

    payload = {
        "metadata": metadata,
        "samples": output_samples,
    }
    torch.save(payload, output_path)

    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] Shards read: {processed_shard_count}")
    print(f"[DONE] Sample count: {len(output_samples)}")
    if output_samples:
        print(f"[DONE] Embedding dim: {int(output_samples[0]['embedding'].numel())}")
        print(f"[DONE] Frequency map shape: {tuple(output_samples[0]['freq_real'].shape)}")
        print(
            "[DONE] Sample schema: "
            "{'sample_index':int, 'triangles':(T,3,2), 'embedding':(D), 'freq_real':(H,W), 'freq_imag':(H,W)}"
        )
    else:
        print("[DONE] No samples exported.")


if __name__ == "__main__":
    main()
