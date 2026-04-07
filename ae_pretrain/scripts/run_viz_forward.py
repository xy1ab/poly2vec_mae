"""Visualize one sample exported by `run_forward_batch.py`.

This script reads one forward-export `.pt` file and saves a four-panel PNG:
1) Triangle mesh
2) Frequency-domain real map
3) Frequency-domain imaginary map
4) ICFT raster reconstructed from real/imag maps
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _build_output_path(project_root: Path, input_path: str, output_path: str | None, sample_index: int) -> Path:
    """Resolve output path from explicit CLI path or one default PNG path."""
    if output_path:
        return Path(output_path).expanduser().resolve()

    input_stem = Path(input_path).expanduser().resolve().stem
    return project_root / "outputs" / "viz_forward" / f"{input_stem}_sample_{sample_index:06d}.png"


def _load_forward_payload(input_path: Path):
    """Load and validate one forward-export payload."""
    import torch

    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Forward export must be a dict payload: {input_path}")

    metadata = payload.get("metadata")
    samples = payload.get("samples")
    if not isinstance(metadata, dict):
        raise TypeError(f"Forward export payload is missing dict `metadata`: {input_path}")
    if not isinstance(samples, list):
        raise TypeError(f"Forward export payload is missing list `samples`: {input_path}")
    return metadata, samples


def _symmetric_abs_limit(array: np.ndarray) -> float:
    """Compute one symmetric color limit for real/imag panels."""
    finite = np.asarray(array, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    bound = float(np.max(np.abs(finite)))
    return bound if bound > 0.0 else 1.0


def _triangle_bounds(triangles: np.ndarray) -> tuple[float, float, float, float]:
    """Compute one padded plotting box for a triangle mesh."""
    min_xy = triangles.reshape(-1, 2).min(axis=0)
    max_xy = triangles.reshape(-1, 2).max(axis=0)

    span_x = float(max_xy[0] - min_xy[0])
    span_y = float(max_xy[1] - min_xy[1])
    span = max(span_x, span_y, 1e-6)
    pad = span * 0.08

    cx = float((min_xy[0] + max_xy[0]) * 0.5)
    cy = float((min_xy[1] + max_xy[1]) * 0.5)
    half = span * 0.5 + pad
    return cx - half, cx + half, cy - half, cy + half


def _plot_triangle_mesh(ax, triangles: np.ndarray) -> None:
    """Plot one triangle mesh as black wireframe lines."""
    for tri in triangles:
        closed = np.vstack([tri, tri[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=0.8)

    x_min, x_max, y_min, y_max = _triangle_bounds(triangles)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Triangles", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _restore_padded_frequency_map(freq_map, codec, map_name: str):
    """Restore one exported valid frequency map to codec-padded grid shape.

    Exported forward samples store only the valid, non-padded frequency area.
    `codec.icft_2d()` expects the full padded grid shape defined by the codec.
    """
    import torch

    freq_tensor = torch.as_tensor(freq_map, dtype=torch.float32).cpu()
    if freq_tensor.ndim != 2:
        raise ValueError(f"`{map_name}` must be 2D, got {tuple(freq_tensor.shape)}")

    full_h = int(codec.converter.U.shape[0])
    full_w = int(codec.converter.U.shape[1])
    valid_h = int(full_h - codec.converter.pad_h)
    valid_w = int(full_w - codec.converter.pad_w)
    current_h, current_w = int(freq_tensor.shape[0]), int(freq_tensor.shape[1])

    if (current_h, current_w) == (full_h, full_w):
        return freq_tensor
    if (current_h, current_w) != (valid_h, valid_w):
        raise ValueError(
            f"`{map_name}` has incompatible shape {tuple(freq_tensor.shape)}. "
            f"Expected either valid shape {(valid_h, valid_w)} or padded shape {(full_h, full_w)}."
        )

    padded = torch.zeros((full_h, full_w), dtype=freq_tensor.dtype)
    padded[:valid_h, :valid_w] = freq_tensor
    return padded


def _plot_forward_sample(
    triangles: np.ndarray,
    freq_real: np.ndarray,
    freq_imag: np.ndarray,
    spatial_raster: np.ndarray,
    sample_index: int,
    output_path: Path,
) -> None:
    """Save one 1x4 visualization figure for one exported sample."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    _plot_triangle_mesh(axes[0], triangles)

    real_lim = _symmetric_abs_limit(freq_real)
    imag_lim = _symmetric_abs_limit(freq_imag)

    im_real = axes[1].imshow(
        freq_real,
        cmap="RdBu_r",
        vmin=-real_lim,
        vmax=real_lim,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].set_title("Freq Real", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im_real, ax=axes[1], fraction=0.046, pad=0.04)

    im_imag = axes[2].imshow(
        freq_imag,
        cmap="RdBu_r",
        vmin=-imag_lim,
        vmax=imag_lim,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    axes[2].set_title("Freq Imag", fontsize=14)
    axes[2].axis("off")
    plt.colorbar(im_imag, ax=axes[2], fraction=0.046, pad=0.04)

    im_raster = axes[3].imshow(
        spatial_raster,
        cmap="gray",
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    axes[3].set_title("ICFT Raster", fontsize=14)
    axes[3].axis("off")
    plt.colorbar(im_raster, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"Forward Sample {sample_index}", fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.08, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for forward-export visualization."""
    parser = argparse.ArgumentParser(description="Visualize one sample exported by run_forward_batch.py")
    parser.add_argument("--input_path", type=str, required=True, help="Forward export `.pt` file path.")
    parser.add_argument("--index", type=int, required=True, help="1-based sample index inside the export payload.")
    parser.add_argument("--output_path", type=str, default="", help="Output PNG file path.")
    return parser


def main() -> None:
    """CLI main entrypoint."""
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    import torch

    if __package__ in {None, ""}:
        import importlib

        get_geometry_codec = importlib.import_module(
            "ae_pretrain.src.datasets.registry"
        ).get_geometry_codec
        ensure_dir = importlib.import_module(
            "ae_pretrain.src.utils.filesystem"
        ).ensure_dir
    else:
        from ..src.datasets.registry import get_geometry_codec
        from ..src.utils.filesystem import ensure_dir

    args = build_arg_parser().parse_args()
    if args.index < 1:
        raise ValueError(f"`index` must be >= 1, got {args.index}")

    input_path = _resolve_user_path(args.input_path, project_root)
    if not input_path.is_file():
        raise FileNotFoundError(f"Forward export file does not exist: {input_path}")

    metadata, samples = _load_forward_payload(input_path)
    if not samples:
        raise ValueError(f"Forward export payload contains no samples: {input_path}")
    if args.index > len(samples):
        raise IndexError(f"`index` out of range: {args.index} > {len(samples)}")

    config = metadata.get("config", {})
    runtime_config = metadata.get("runtime_config", {})
    if not isinstance(config, dict):
        raise TypeError("Forward export `metadata.config` must be a dict.")
    if not isinstance(runtime_config, dict):
        raise TypeError("Forward export `metadata.runtime_config` must be a dict.")

    codec_config = dict(config)
    codec_config.update(runtime_config)
    geom_type = str(codec_config.get("geom_type", metadata.get("geom_type", "polygon"))).lower()
    codec = get_geometry_codec(geom_type, codec_config, device="cpu")

    sample = samples[args.index - 1]
    if not isinstance(sample, dict):
        raise TypeError(f"Sample payload must be a dict, got {type(sample).__name__}")

    required_keys = ("triangles", "freq_real", "freq_imag")
    missing_keys = [key for key in required_keys if key not in sample]
    if missing_keys:
        raise KeyError(f"Sample is missing required keys: {missing_keys}")

    triangles = torch.as_tensor(sample["triangles"], dtype=torch.float32).cpu().numpy()
    freq_real = _restore_padded_frequency_map(sample["freq_real"], codec=codec, map_name="freq_real")
    freq_imag = _restore_padded_frequency_map(sample["freq_imag"], codec=codec, map_name="freq_imag")

    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        raise ValueError(f"`triangles` must have shape [T,3,2], got {tuple(triangles.shape)}")

    spatial_size = int(metadata.get("spatial_size", 256))
    with torch.no_grad():
        spatial_raster = codec.icft_2d(
            freq_real.unsqueeze(0),
            freq_imag.unsqueeze(0),
            spatial_size=spatial_size,
        )[0].squeeze().cpu().numpy()

    sample_index = int(sample.get("sample_index", args.index))
    output_path = _build_output_path(project_root, str(input_path), args.output_path or None, sample_index)
    ensure_dir(output_path.parent)

    _plot_forward_sample(
        triangles=triangles,
        freq_real=freq_real[: int(freq_real.shape[0] - codec.converter.pad_h), : int(freq_real.shape[1] - codec.converter.pad_w)].numpy(),
        freq_imag=freq_imag[: int(freq_imag.shape[0] - codec.converter.pad_h), : int(freq_imag.shape[1] - codec.converter.pad_w)].numpy(),
        spatial_raster=spatial_raster,
        sample_index=sample_index,
        output_path=output_path,
    )

    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] Input: {input_path}")
    print(f"[DONE] Sample index: {sample_index}")


if __name__ == "__main__":
    main()
