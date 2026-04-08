"""Run one VQAE reconstruction visualization for a selected shard sample."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as MplPath

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CURRENT_DIR.parent
    _REPO_ROOT = _PROJECT_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib

    ensure_cuda_runtime_libs = importlib.import_module(
        "vqae_pretrain.scripts.runtime_bootstrap"
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


def _resolve_model_paths(model_dir: str) -> tuple[Path, Path]:
    base = Path(model_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {base}")

    search_roots = [base]
    for subdir in ("best", "ckpt"):
        candidate = base / subdir
        if candidate.is_dir():
            search_roots.append(candidate)

    checkpoint_candidates: list[Path] = []
    config_candidates: list[Path] = []
    for root in search_roots:
        for path in (root / "vqae_best.pth",):
            if path.exists() and path not in checkpoint_candidates:
                checkpoint_candidates.append(path)
        for path in (root / "config.yaml", root / "config.yml", root / "poly_vqae_config.json"):
            if path.exists() and path not in config_candidates:
                config_candidates.append(path)

    if not checkpoint_candidates:
        raise FileNotFoundError(f"No `vqae_best.pth` found under: {base}")
    if not config_candidates:
        raise FileNotFoundError(f"No config file found under: {base}")
    return checkpoint_candidates[0], config_candidates[0]


def _load_sample_by_row_index(manifest, shard_loader, row_index: int):
    if row_index < 0:
        raise IndexError(f"`row_index` must be >= 0, got {row_index}")
    shard_id, local_index = manifest.locate_sample(row_index)
    shard_info = manifest.shards[shard_id]
    shard_data = shard_loader(shard_info.path)
    return shard_data[local_index]


def _rasterize_tris_to_grid(tris: np.ndarray, height: int, width: int) -> np.ndarray:
    x = np.linspace(-1, 1, width)
    y = np.linspace(1, -1, height)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    mask = np.zeros(height * width, dtype=bool)
    for tri in tris:
        poly_path = MplPath(tri)
        mask |= poly_path.contains_points(points)
    return mask.reshape(height, width).astype(np.float32)


def main() -> None:
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    if __package__ in {None, ""}:
        import importlib

        PtShardManifest = importlib.import_module("vqae_pretrain.src.datasets.pt_manifest").PtShardManifest
        shard_io = importlib.import_module("vqae_pretrain.src.datasets.shard_io")
        resolve_triangle_shard_paths = shard_io.resolve_triangle_shard_paths
        load_triangle_shard = shard_io.load_triangle_shard
        pipeline_module = importlib.import_module("vqae_pretrain.src.engine.pipeline")
        PolyVqAePipeline = pipeline_module.PolyVqAePipeline
        filesystem_module = importlib.import_module("vqae_pretrain.src.utils.filesystem")
        ensure_dir = filesystem_module.ensure_dir
    else:
        from ..src.datasets.pt_manifest import PtShardManifest
        from ..src.datasets.shard_io import load_triangle_shard, resolve_triangle_shard_paths
        from ..src.engine.pipeline import PolyVqAePipeline
        from ..src.utils.filesystem import ensure_dir

    parser = argparse.ArgumentParser(description="Run one VQAE reconstruction visualization")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing VQAE checkpoint and config.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing triangle shard `.pt` files.")
    parser.add_argument("--row_index", type=int, default=0, help="Global row/sample index across all shard files.")
    parser.add_argument("--save_dir", type=str, default=str(project_root / "outputs" / "eval"))
    parser.add_argument("--spatial_size", type=int, default=256, help="Spatial raster size for GT visualization.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="bf16")
    args = parser.parse_args()

    checkpoint_path, config_path = _resolve_model_paths(args.model_dir)
    pt_files = resolve_triangle_shard_paths(Path(args.data_dir).expanduser().resolve())
    manifest = PtShardManifest.from_pt_files(pt_files)
    sample = _load_sample_by_row_index(manifest, load_triangle_shard, args.row_index)
    triangles = np.asarray(sample, dtype=np.float32)

    pipeline = PolyVqAePipeline(
        weight_path=str(checkpoint_path),
        config_path=str(config_path),
        device=args.device,
        precision=args.precision,
    )

    imgs = pipeline.triangles_to_images([triangles]).float().cpu()
    with torch.no_grad():
        with torch.autocast(device_type=pipeline.device.type, enabled=False):
            outputs = pipeline.model(
                pipeline.triangles_to_images([triangles]),
                use_vq=True,
            )
    recon_imgs = outputs.recon_imgs.float().cpu()
    real_part, imag_part, perplexity, active_codes = pipeline.reconstruct_triangles([triangles])

    orig_mag = imgs[0, 0].numpy()
    orig_cos = imgs[0, 1].numpy()
    orig_sin = imgs[0, 2].numpy()
    recon_mag = recon_imgs[0, 0].numpy()
    recon_cos = recon_imgs[0, 1].numpy()
    recon_sin = recon_imgs[0, 2].numpy()

    spatial_gt = _rasterize_tris_to_grid(triangles, height=args.spatial_size, width=args.spatial_size)
    spatial_recon = pipeline.codec.icft_2d(real_part.to(pipeline.device), imag_part.to(pipeline.device))[0].squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    panels = [
        (orig_mag, "Orig Mag", "viridis"),
        (orig_cos, "Orig Cos", "viridis"),
        (orig_sin, "Orig Sin", "viridis"),
        (spatial_gt, "Spatial GT", "gray"),
        (recon_mag, "Recon Mag", "viridis"),
        (recon_cos, "Recon Cos", "viridis"),
        (recon_sin, "Recon Sin", "viridis"),
        (spatial_recon, f"Spatial Recon\nPerp={perplexity.item():.2f}, Active={int(active_codes.item())}", "gray"),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(data, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_dir = ensure_dir(Path(args.save_dir))
    output_path = output_dir / f"vqae_eval_row_{args.row_index:06d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved visualization to: {output_path}")
    print(f"[INFO] Perplexity={perplexity.item():.4f} | ActiveCodes={int(active_codes.item())}")


if __name__ == "__main__":
    main()
