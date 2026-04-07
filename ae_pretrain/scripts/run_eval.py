"""Model-aware AE evaluation launcher script.

This script performs AE reconstruction visualization by:
1) resolving one AE checkpoint + config from `--model_dir`,
2) locating one triangle sample by global row index from `--data_dir`,
3) running full-image AE inference, and
4) exporting the original multi-panel reconstruction visualization PNG.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

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
    """Inject repository root into `sys.path` for direct script execution.

    Returns:
        `ae_pretrain` project root path.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def _resolve_user_path(path_str: str, project_root: Path) -> Path:
    """Resolve a user-provided path against cwd and project root.

    Args:
        path_str: Raw CLI path string.
        project_root: Project root used as fallback base.

    Returns:
        Resolved absolute path candidate.
    """
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (project_root / raw_path).resolve()


def _extract_last_int(text: str) -> int | None:
    """Extract the last integer substring from text, if present."""
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def _resolve_model_paths(
    model_dir: str,
) -> tuple[Path, Path]:
    """Resolve one AE checkpoint and one model config from `model_dir`.

    Args:
        model_dir: Directory containing model checkpoint and config files.

    Returns:
        Tuple `(checkpoint_path, config_path)`.

    Raises:
        FileNotFoundError: If resolved files do not exist.
    """
    base = Path(model_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {base}")

    config_candidates = [base / "config.yaml", base / "config.yml", base / "poly_ae_config.json", base / "poly_mae_config.json"]
    cfg_path = next((path for path in config_candidates if path.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(
            "Model config not found in model_dir. Expected one of: "
            f"{', '.join(path.name for path in config_candidates)}"
        )

    checkpoint_candidates = [
        path
        for path in base.glob("*.pth")
        if path.is_file()
        and (
            "encoder_decoder" in path.name.lower()
            or "mae" in path.name.lower()
            or "autoencoder" in path.name.lower()
            or "ae_best" in path.name.lower()
        )
    ]
    if not checkpoint_candidates:
        raise FileNotFoundError(
            "No AE checkpoint found in model_dir. Expected a `.pth` file whose name contains "
            "`ae` or `autoencoder` (legacy `mae` / `encoder_decoder` names are also accepted)."
        )

    def _checkpoint_rank(path: Path) -> tuple[int, int, float, str]:
        name_lower = path.name.lower()
        if name_lower == "autoencoder.pth":
            priority = 0
        elif name_lower == "ae_best.pth":
            priority = 1
        elif name_lower == "encoder_decoder.pth":
            priority = 2
        elif "encoder_decoder" in name_lower:
            priority = 3
        else:
            priority = 4
        suffix = _extract_last_int(path.stem)
        suffix_rank = -(suffix if suffix is not None else -1)
        mtime_rank = -float(path.stat().st_mtime)
        return (priority, suffix_rank, mtime_rank, name_lower)

    checkpoint_candidates = sorted(checkpoint_candidates, key=_checkpoint_rank)
    return checkpoint_candidates[0], cfg_path


def _load_sample_by_row_index(manifest, shard_loader, row_index: int):
    """Load one sample from a manifest-global row index."""
    if row_index < 0:
        raise IndexError(f"`row_index` must be >= 0, got {row_index}")

    shard_id, local_index = manifest.locate_sample(row_index)
    shard_info = manifest.shards[shard_id]
    shard_data = shard_loader(shard_info.path)
    sample = shard_data[local_index]
    shard_cumulative_end = int(shard_info.start_index + shard_info.num_samples)
    return sample, shard_info.path, int(local_index), shard_cumulative_end, int(shard_info.start_index)


def _build_eval_parser(project_root: Path) -> argparse.ArgumentParser:
    """Build CLI parser for model-aware AE visualization evaluation.

    Args:
        project_root: Project root path.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run AE reconstruction visualization")

    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing AE checkpoint and model config.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing one or more triangle shard `.pt` files.")
    parser.add_argument("--row_index", type=int, default=0, help="Global row/sample index across manifest-managed or fallback-sorted shard `.pt` files.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(project_root / "outputs" / "viz"),
        help="Visualization output directory.",
    )
    parser.add_argument("--spatial_size", type=int, default=256)

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


def mag_phase_to_real_imag(mag_log, phase):
    """Convert log-magnitude and phase channels into real/imaginary parts."""
    raw_mag = np.expm1(mag_log) if isinstance(mag_log, np.ndarray) else None
    if raw_mag is not None:
        real_part = raw_mag * np.cos(phase)
        imag_part = raw_mag * np.sin(phase)
        return real_part, imag_part

    import torch

    raw_mag = torch.expm1(mag_log)
    real_part = raw_mag * torch.cos(phase)
    imag_part = raw_mag * torch.sin(phase)
    return real_part, imag_part


def rasterize_tris_to_grid(tris, height: int, width: int) -> np.ndarray:
    """Rasterize triangle set into binary occupancy grid."""
    if hasattr(tris, "detach"):
        tris_np = tris.detach().cpu().numpy()
    else:
        tris_np = np.asarray(tris)

    x = np.linspace(-1, 1, width)
    y = np.linspace(1, -1, height)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T

    mask = np.zeros(height * width, dtype=bool)
    for tri in tris_np:
        poly_path = MplPath(tri)
        mask = mask | poly_path.contains_points(points)

    return mask.reshape(height, width).astype(np.float32)


def plot_reconstruction(
    img_orig,
    img_masked,
    img_recon,
    spatial_gt: np.ndarray,
    spatial_icft_orig,
    spatial_icft_recon,
    output_stem: str,
    save_dir: str | Path,
) -> None:
    """Save the original training-style reconstruction visualization."""
    import torch

    fig = plt.figure(figsize=(30, 12))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.0, 5.1], wspace=0.10)
    gs_left = outer[0, 0].subgridspec(3, 3, wspace=0.02, hspace=0.24)
    gs_right = outer[0, 1].subgridspec(1, 3, wspace=0.20)

    vmin_mag, vmax_mag = img_orig[0].min(), img_orig[0].max()
    vmin_trig, vmax_trig = -1.0, 1.0

    vmin_sp = 0.0
    vmax_sp = max(1.0, spatial_icft_orig.max().item(), spatial_icft_recon.max().item())

    axes_left = [
        [fig.add_subplot(gs_left[0, 0]), fig.add_subplot(gs_left[0, 1]), fig.add_subplot(gs_left[0, 2])],
        [fig.add_subplot(gs_left[1, 0]), fig.add_subplot(gs_left[1, 1]), fig.add_subplot(gs_left[1, 2])],
        [fig.add_subplot(gs_left[2, 0]), fig.add_subplot(gs_left[2, 1]), fig.add_subplot(gs_left[2, 2])],
    ]
    axes_right = [fig.add_subplot(gs_right[0, 0]), fig.add_subplot(gs_right[0, 1]), fig.add_subplot(gs_right[0, 2])]

    titles_left = [
        ["Original Mag", "Input Mag", "Reconstructed Mag"],
        ["Original Cos(Phase)", "Input Cos(Phase)", "Reconstructed Cos(Phase)"],
        ["Original Sin(Phase)", "Input Sin(Phase)", "Reconstructed Sin(Phase)"],
    ]
    titles_right = ["Strict Spatial GT (0/1)", "ICFT (Orig Freqs)", "ICFT (AE Recon Freqs)"]

    data_left = [
        [(img_orig[0], vmin_mag, vmax_mag), (img_masked[0], vmin_mag, vmax_mag), (img_recon[0], vmin_mag, vmax_mag)],
        [(img_orig[1], vmin_trig, vmax_trig), (img_masked[1], vmin_trig, vmax_trig), (img_recon[1], vmin_trig, vmax_trig)],
        [(img_orig[2], vmin_trig, vmax_trig), (img_masked[2], vmin_trig, vmax_trig), (img_recon[2], vmin_trig, vmax_trig)],
    ]
    data_right = [(spatial_gt, vmin_sp, vmax_sp), (spatial_icft_orig, vmin_sp, vmax_sp), (spatial_icft_recon, vmin_sp, vmax_sp)]

    for row in range(3):
        for col in range(3):
            ax = axes_left[row][col]
            data, vmin, vmax = data_left[row][col]
            data_np = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
            data_np = np.ma.masked_invalid(data_np)
            im = ax.imshow(
                data_np,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation="nearest",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(titles_left[row][col], fontsize=14)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.040, pad=0.015)

    for col in range(3):
        ax = axes_right[col]
        data, vmin, vmax = data_right[col]
        data_np = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
        data_np = np.ma.masked_invalid(data_np)
        im = ax.imshow(
            data_np,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles_right[col], fontsize=14)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{output_stem}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """CLI main function for model-aware AE evaluation."""
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    import torch

    if __package__ in {None, ""}:
        import importlib

        PtShardManifest = importlib.import_module(
            "ae_pretrain.src.datasets.pt_manifest"
        ).PtShardManifest
        load_triangle_shard = importlib.import_module(
            "ae_pretrain.src.datasets.shard_io"
        ).load_triangle_shard
        get_geometry_codec = importlib.import_module(
            "ae_pretrain.src.datasets.registry"
        ).get_geometry_codec
        load_ae_model = importlib.import_module(
            "ae_pretrain.src.models.factory"
        ).load_ae_model
        ensure_dir = importlib.import_module(
            "ae_pretrain.src.utils.filesystem"
        ).ensure_dir
        precision_module = importlib.import_module("ae_pretrain.src.utils.precision")
        autocast_context = precision_module.autocast_context
        normalize_precision = precision_module.normalize_precision
    else:
        from ..src.datasets.pt_manifest import PtShardManifest
        from ..src.datasets.shard_io import load_triangle_shard
        from ..src.datasets.registry import get_geometry_codec
        from ..src.models.factory import load_ae_model
        from ..src.utils.filesystem import ensure_dir
        from ..src.utils.precision import autocast_context, normalize_precision

    parser = _build_eval_parser(project_root)
    args = parser.parse_args()
    explicit_cli_keys = _extract_explicit_cli_keys(sys.argv[1:])

    args.precision = normalize_precision(args.precision)
    args.model_dir = str(_resolve_user_path(args.model_dir, project_root))
    args.data_dir = str(_resolve_user_path(args.data_dir, project_root))
    args.save_dir = str(_resolve_user_path(args.save_dir, project_root))

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, fallback to CPU for evaluation.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    ckpt_path, cfg_path = _resolve_model_paths(args.model_dir)
    model, model_cfg = load_ae_model(
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

    manifest = PtShardManifest.from_data_dir(args.data_dir, warn_fn=lambda message: print(message))
    sample, shard_path, local_index, _shard_cumulative_end, shard_global_start = _load_sample_by_row_index(
        manifest,
        load_triangle_shard,
        int(args.row_index),
    )
    tris = torch.as_tensor(sample, dtype=torch.float32)
    batch_tris = tris.unsqueeze(0).to(device)
    lengths = torch.tensor([tris.shape[0]], device=device)

    with torch.no_grad():
        mag_fix, phase_fix = codec.cft_batch(batch_tris, lengths)
        imgs_fix = torch.cat([mag_fix, torch.cos(phase_fix), torch.sin(phase_fix)], dim=1)

        with autocast_context(device, args.precision):
            pred_fix = model(imgs_fix)

        pred_fix = pred_fix.float()

        p = patch_size
        h, w = imgs_fix.shape[2], imgs_fix.shape[3]
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Input size ({h}, {w}) is not divisible by patch_size={p}.")
        h_p, w_p = h // p, w // p

        img_orig = imgs_fix[0].detach().cpu()

        img_masked = img_orig.clone()

        pred_img = pred_fix[0].detach().cpu().reshape(h_p, w_p, 3, p, p)
        pred_img = torch.einsum("hwcpq->chpwq", pred_img).reshape(3, h, w)
        img_recon = pred_img

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
    output_stem = f"recon_row_{int(args.row_index)}"
    plot_reconstruction(
        img_orig=img_orig,
        img_masked=img_masked,
        img_recon=img_recon,
        spatial_gt=spatial_gt,
        spatial_icft_orig=spatial_icft_orig.squeeze(),
        spatial_icft_recon=spatial_icft_recon,
        output_stem=output_stem,
        save_dir=save_dir,
    )
    save_path = Path(save_dir) / f"{output_stem}.png"

    print("[INFO] Evaluation completed.")
    print(f"[INFO] Data directory    : {args.data_dir}")
    print(f"[INFO] Data shard        : {shard_path}")
    print(f"[INFO] Row index         : {args.row_index} (local_index={local_index}, shard_start={shard_global_start})")
    print(f"[INFO] Model checkpoint : {ckpt_path}")
    print(f"[INFO] Model config     : {cfg_path}")
    print(f"[INFO] Visualization    : {save_path}")


if __name__ == "__main__":
    main()
