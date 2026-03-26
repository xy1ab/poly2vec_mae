"""Model-aware MAE evaluation launcher script.

This script performs MAE reconstruction visualization by loading an exported
encoder+decoder checkpoint (typically from `outputs/exports/*`). It supports:
1) YAML defaults + CLI overrides.
2) Mask-ratio controlled MAE inference on one polygon sample.
3) Training-style multi-panel visualization export to `outputs/viz`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from runtime_bootstrap import ensure_cuda_runtime_libs


def _inject_src_path() -> Path:
    """Inject local `src` directory into `sys.path`.

    Returns:
        Project root path.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return project_root


def _build_cli_args_from_config(config_dict: dict) -> list[str]:
    """Convert config dictionary into CLI argument list.

    Args:
        config_dict: Parsed config dictionary.

    Returns:
        Flat CLI argument list.
    """
    cli_args: list[str] = []
    for key, value in config_dict.items():
        arg_name = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli_args.append(arg_name)
        else:
            cli_args.extend([arg_name, str(value)])
    return cli_args


def _resolve_model_paths(
    model_dir: str | None,
    mae_ckpt_path: str | None,
    model_config_path: str | None,
) -> tuple[Path, Path]:
    """Resolve full-MAE checkpoint and model-config paths.

    Args:
        model_dir: Export bundle directory containing checkpoint/config files.
        mae_ckpt_path: Optional explicit MAE checkpoint path.
        model_config_path: Optional explicit model config path.

    Returns:
        Tuple `(checkpoint_path, config_path)`.

    Raises:
        ValueError: If required paths are missing.
        FileNotFoundError: If resolved files do not exist.
    """
    ckpt_path = Path(mae_ckpt_path).expanduser() if mae_ckpt_path else None
    cfg_path = Path(model_config_path).expanduser() if model_config_path else None

    if model_dir:
        base = Path(model_dir).expanduser()
        if not base.exists() or not base.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {base}")

        if ckpt_path is None:
            ckpt_path = base / "encoder_decoder.pth"

        if cfg_path is None:
            candidates = [base / "config.yaml", base / "config.yml", base / "poly_mae_config.json"]
            cfg_path = next((path for path in candidates if path.exists()), None)

    if ckpt_path is None or cfg_path is None:
        raise ValueError("You must provide model path via `--model_dir` or `--mae_ckpt_path` + `--model_config_path`.")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"MAE checkpoint not found: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")

    return ckpt_path, cfg_path


def _build_eval_parser(project_root: Path) -> argparse.ArgumentParser:
    """Build CLI parser for model-aware MAE visualization evaluation.

    Args:
        project_root: Project root path.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run MAE encoder+decoder reconstruction visualization")

    parser.add_argument("--index", type=int, default=0, help="Sample index in preprocessed dataset.")
    parser.add_argument("--data_path", type=str, default="./data/processed/polygon_triangles_normalized.pt")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(project_root / "outputs" / "viz"),
        help="Visualization output directory.",
    )
    parser.add_argument("--spatial_size", type=int, default=256)

    parser.add_argument("--model_dir", type=str, default=None, help="Export bundle directory containing encoder_decoder.pth.")
    parser.add_argument("--mae_ckpt_path", type=str, default=None, help="Explicit full-MAE checkpoint path.")
    parser.add_argument("--model_config_path", type=str, default=None, help="Explicit model config path (yaml/json).")

    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Mask ratio used for MAE reconstruction.")
    parser.add_argument("--precision", type=str, default="bf16", help="Inference precision: fp32/bf16/fp16.")
    parser.add_argument("--device", type=str, default=None, help="Runtime device, e.g. cpu/cuda/cuda:0.")

    # Optional codec overrides. When omitted, values are taken from model config.
    parser.add_argument("--pos_freqs", type=int, default=None)
    parser.add_argument("--w_min", type=float, default=None)
    parser.add_argument("--w_max", type=float, default=None)
    parser.add_argument("--freq_type", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=None)

    return parser


def _extract_explicit_cli_keys(argv_tokens: list[str]) -> set[str]:
    """Extract explicitly provided long-option keys from raw CLI tokens.

    Args:
        argv_tokens: CLI tokens that remain after pre-parser handling.

    Returns:
        Set of normalized long-option names without leading dashes.
    """
    keys: set[str] = set()
    for token in argv_tokens:
        if token.startswith("--") and len(token) > 2:
            keys.add(token[2:].split("=", maxsplit=1)[0].replace("-", "_"))
    return keys


def main() -> None:
    """CLI main function for model-aware MAE evaluation."""
    ensure_cuda_runtime_libs()
    project_root = _inject_src_path()

    import torch
    from datasets.registry import get_geometry_codec
    from engine.trainer import mag_phase_to_real_imag, plot_reconstruction, rasterize_tris_to_grid
    from models.factory import load_mae_model
    from utils.config import load_yaml_config
    from utils.filesystem import ensure_dir
    from utils.precision import autocast_context, normalize_precision
    from utils.safe_load import register_numpy_safe_globals

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "eval_default.yaml"),
        type=str,
    )
    pre_args, remaining = pre_parser.parse_known_args()

    explicit_cli_keys = _extract_explicit_cli_keys(remaining)

    config = load_yaml_config(pre_args.config)
    config_cli_args = _build_cli_args_from_config(config)

    parser = _build_eval_parser(project_root)
    args = parser.parse_args(config_cli_args + remaining)

    args.precision = normalize_precision(args.precision)
    if not (0.0 <= float(args.mask_ratio) < 1.0):
        raise ValueError(f"`mask_ratio` must be in [0, 1). Got: {args.mask_ratio}")

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, fallback to CPU for evaluation.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    ckpt_path, cfg_path = _resolve_model_paths(args.model_dir, args.mae_ckpt_path, args.model_config_path)
    model, model_cfg = load_mae_model(
        weight_path=ckpt_path,
        config_path=cfg_path,
        device=device,
        precision=args.precision,
    )

    # Use model-side geometry params by default to guarantee shape compatibility.
    # Only explicit CLI overrides can replace these values.
    pos_freqs = int(args.pos_freqs) if "pos_freqs" in explicit_cli_keys and args.pos_freqs is not None else int(model_cfg.get("pos_freqs", 63))
    w_min = float(args.w_min) if "w_min" in explicit_cli_keys and args.w_min is not None else float(model_cfg.get("w_min", 0.1))
    w_max = float(args.w_max) if "w_max" in explicit_cli_keys and args.w_max is not None else float(model_cfg.get("w_max", 200.0))
    freq_type = str(args.freq_type) if "freq_type" in explicit_cli_keys and args.freq_type is not None else str(model_cfg.get("freq_type", "geometric"))
    patch_size = int(args.patch_size) if "patch_size" in explicit_cli_keys and args.patch_size is not None else int(model_cfg.get("patch_size", 4))

    codec_cfg = {
        "geom_type": "polygon",
        "pos_freqs": pos_freqs,
        "w_min": w_min,
        "w_max": w_max,
        "freq_type": freq_type,
        "patch_size": patch_size,
    }
    codec = get_geometry_codec("polygon", codec_cfg, device=str(device))

    register_numpy_safe_globals()
    all_polys = torch.load(args.data_path, weights_only=False)
    if args.index < 0 or args.index >= len(all_polys):
        raise IndexError(f"Index {args.index} out of range. Total samples: {len(all_polys)}")

    tris = torch.as_tensor(all_polys[args.index], dtype=torch.float32)
    batch_tris = tris.unsqueeze(0).to(device)
    lengths = torch.tensor([tris.shape[0]], device=device)

    with torch.no_grad():
        mag_fix, phase_fix = codec.cft_batch(batch_tris, lengths)
        imgs_fix = torch.cat([mag_fix, torch.cos(phase_fix), torch.sin(phase_fix)], dim=1)

        with autocast_context(device, args.precision):
            _, _, _, pred_fix, mask_fix = model(imgs_fix, mask_ratio=float(args.mask_ratio))

        pred_fix = pred_fix.float()
        mask_fix = mask_fix.float()

        p = patch_size
        h, w = imgs_fix.shape[2], imgs_fix.shape[3]
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Input size ({h}, {w}) is not divisible by patch_size={p}.")
        h_p, w_p = h // p, w // p

        img_orig = imgs_fix[0].detach().cpu()

        mask_map = mask_fix[0].detach().cpu().reshape(h_p, w_p, 1, 1).expand(-1, -1, p, p)
        mask_map = mask_map.permute(0, 2, 1, 3).reshape(h, w)

        img_masked = img_orig.clone()
        img_masked[:, mask_map == 1] = torch.nan

        pred_img = pred_fix[0].detach().cpu().reshape(h_p, w_p, 3, p, p)
        pred_img = torch.einsum("hwcpq->chpwq", pred_img).reshape(3, h, w)

        img_recon = img_orig.clone()
        img_recon[:, mask_map == 1] = pred_img[:, mask_map == 1]

        real_fix, imag_fix = mag_phase_to_real_imag(mag_fix, phase_fix)
        spatial_icft_orig = codec.icft_2d(
            real_fix.squeeze(1),
            imag_fix.squeeze(1),
            spatial_size=int(args.spatial_size),
        )[0].detach().cpu()

        h_sp, w_sp = int(spatial_icft_orig.shape[-2]), int(spatial_icft_orig.shape[-1])
        spatial_gt = rasterize_tris_to_grid(tris.cpu(), h_sp, w_sp)

        mag_recon = img_recon[0].unsqueeze(0).to(device)
        cos_recon = img_recon[1].unsqueeze(0).to(device)
        sin_recon = img_recon[2].unsqueeze(0).to(device)
        phase_recon = torch.atan2(sin_recon, cos_recon)

        real_recon, imag_recon = mag_phase_to_real_imag(mag_recon, phase_recon)
        spatial_icft_recon = codec.icft_2d(
            real_recon,
            imag_recon,
            spatial_size=int(args.spatial_size),
        )[0].squeeze().detach().cpu()

    save_dir = ensure_dir(args.save_dir)
    viz_id = int(args.index) + 1
    plot_reconstruction(
        img_orig=img_orig,
        img_masked=img_masked,
        img_recon=img_recon,
        spatial_gt=spatial_gt,
        spatial_icft_orig=spatial_icft_orig.squeeze(),
        spatial_icft_recon=spatial_icft_recon,
        epoch=viz_id,
        save_dir=save_dir,
    )
    save_path = Path(save_dir) / f"recon_epoch_{viz_id}.png"

    print("[INFO] Evaluation completed.")
    print(f"[INFO] Model checkpoint : {ckpt_path}")
    print(f"[INFO] Model config     : {cfg_path}")
    print(f"[INFO] Mask ratio       : {args.mask_ratio}")
    print(f"[INFO] Visualization    : {save_path}")


if __name__ == "__main__":
    main()
