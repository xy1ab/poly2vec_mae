"""Legacy evaluation engine for polygon CFT/ICFT visualization.

This module is kept only for backward compatibility with the old single-file
triangle-shard workflow. New MAE reconstruction visualization should prefer
`scripts/run_eval.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as MplPath

from ..datasets.registry import get_geometry_codec
from ..datasets.shard_io import load_triangle_shard
from ..utils.filesystem import ensure_dir


def rasterize_triangles(tris: np.ndarray, spatial_size: int = 256) -> np.ndarray:
    """Rasterize triangle set into binary occupancy map.

    Args:
        tris: Triangle array `[T,3,2]`.
        spatial_size: Raster size.

    Returns:
        Binary raster map `[S,S]`.
    """
    x = np.linspace(-1, 1, spatial_size)
    y = np.linspace(1, -1, spatial_size)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T

    mask = np.zeros(spatial_size * spatial_size, dtype=bool)
    for tri in tris:
        poly_path = MplPath(tri)
        mask |= poly_path.contains_points(points)

    return mask.reshape((spatial_size, spatial_size)).astype(np.float32)


def eval_main(args) -> None:
    """Run the legacy single-file CFT/ICFT visualization evaluation.

    Args:
        args: Parsed evaluator args.
    """
    print("[WARN] `src.engine.evaluator` is legacy. Prefer `scripts/run_eval.py` for MAE reconstruction visualization.")

    all_polys = load_triangle_shard(args.data_path)
    if args.index < 0:
        raise IndexError(f"Index {args.index} out of range. Total samples: {len(all_polys)}")
    if args.index >= len(all_polys):
        raise IndexError(f"Index {args.index} out of range. Total samples: {len(all_polys)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "geom_type": "polygon",
        "pos_freqs": args.pos_freqs,
        "w_min": args.w_min,
        "w_max": args.w_max,
        "freq_type": args.freq_type,
        "patch_size": args.patch_size,
    }
    codec = get_geometry_codec("polygon", config, device=str(device))

    tris = all_polys[args.index]
    orig_raster = rasterize_triangles(tris, spatial_size=args.spatial_size)

    batch_tris = torch.tensor(tris, dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([tris.shape[0]], device=device)

    mag_log, phase = codec.cft_batch(batch_tris, lengths)
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)

    raw_mag = torch.expm1(mag_log)
    f_uv_real = raw_mag * cos_phase
    f_uv_imag = raw_mag * sin_phase
    recon_raster = codec.icft_2d(f_uv_real.squeeze(1), f_uv_imag.squeeze(1), spatial_size=args.spatial_size)

    mag_vis = mag_log.squeeze().cpu().numpy()
    cos_vis = cos_phase.squeeze().cpu().numpy()
    sin_vis = sin_phase.squeeze().cpu().numpy()
    recon_vis = recon_raster.squeeze().cpu().numpy()
    diff_vis = np.abs(orig_raster - recon_vis)

    # Keep aspect policy explicit for frequency/spatial comparability.
    fig, axes = plt.subplots(1, 6, figsize=(36, 6))

    im0 = axes[0].imshow(orig_raster, cmap="gray", vmin=0, vmax=1, extent=[-1, 1, -1, 1], aspect="equal", interpolation="nearest")
    axes[0].set_title("Original Rasterized Polygon")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mag_vis, cmap="viridis", aspect="equal", interpolation="nearest")
    axes[1].set_title("CFT Magnitude (log1p)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(cos_vis, cmap="viridis", vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
    axes[2].set_title("CFT Cos(Phase)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(sin_vis, cmap="viridis", vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
    axes[3].set_title("CFT Sin(Phase)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    im4 = axes[4].imshow(recon_vis, cmap="gray", vmin=0, vmax=1, extent=[-1, 1, -1, 1], aspect="equal", interpolation="nearest")
    axes[4].set_title("ICFT Reconstructed Field")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    im5 = axes[5].imshow(diff_vis, cmap="Reds", vmin=0, vmax=1, extent=[-1, 1, -1, 1], aspect="equal", interpolation="nearest")
    axes[5].set_title("Absolute Difference")
    plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    save_dir = ensure_dir(args.save_dir)
    save_path = save_dir / f"cft_visualize_idx_{args.index}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[INFO] Visualization saved to: {save_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the legacy evaluator CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Legacy CFT and ICFT visualization evaluator")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./data/processed/polygon_triangles_normalized.pt")
    parser.add_argument("--save_dir", type=str, default="./outputs/ckpt/eval")
    parser.add_argument("--spatial_size", type=int, default=256)

    parser.add_argument("--pos_freqs", type=int, default=63)
    parser.add_argument("--w_min", type=float, default=0.1)
    parser.add_argument("--w_max", type=float, default=200.0)
    parser.add_argument("--freq_type", type=str, default="geometric")
    parser.add_argument("--patch_size", type=int, default=4)
    return parser


def run_cli(argv=None) -> None:
    """CLI wrapper for evaluator.

    Args:
        argv: Optional argv list.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    eval_main(args)
