"""Batch encoding launcher script.

This script reads vector polygons, applies optional augmentation, and encodes
samples into embeddings via `engine.pipeline.PolyEncoderPipeline`.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
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


def _default_device_for_runtime() -> str:
    """Resolve default runtime device string.

    Returns:
        `"cuda"` when CUDA is available, otherwise `"cpu"`.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _iter_vector_files(input_dirs: list[str], recursive: bool = True) -> list[Path]:
    """Collect supported vector files from source directories.

    Args:
        input_dirs: Source directory list.
        recursive: Whether to scan recursively.

    Returns:
        Sorted unique vector file paths.
    """
    suffixes = {".shp", ".geojson"}
    all_files: list[Path] = []

    for directory in input_dirs:
        root = Path(directory)
        if not root.exists():
            continue

        if recursive:
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in suffixes:
                    all_files.append(path)
        else:
            for path in root.iterdir():
                if path.is_file() and path.suffix.lower() in suffixes:
                    all_files.append(path)

    return sorted(set(all_files))


def _expand_polygons(geom):
    """Normalize geometry objects into a list of Polygon objects.

    Args:
        geom: Shapely geometry object.

    Returns:
        List of polygon geometries.
    """
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return []


def _normalize_with_bbox_limit(coords: np.ndarray, eps: float = 1e-6):
    """Normalize polygon coordinates into bbox-based limit range.

    This function performs strict bbox normalization:
    1) Compute physical bbox center `(cx, cy)`.
    2) Compute physical bbox square side length `side_len = max(dx, dy)`.
    3) Normalize by `(coords - center) / (side_len / 2)`.

    After normalization, all coordinates are within `[-1, 1]`, and the longer
    bbox axis reaches the limit value `±1` (up to floating-point precision).

    Args:
        coords: Coordinate array with shape `[N,2]` in physical coordinates.
        eps: Degenerate threshold for side length.

    Returns:
        Tuple `(coords_norm, cx, cy, side_len)` or None for degenerate input.
    """
    coords_np = np.asarray(coords, dtype=np.float64)
    if coords_np.ndim != 2 or coords_np.shape[1] != 2:
        return None

    min_xy = coords_np.min(axis=0)
    max_xy = coords_np.max(axis=0)
    cx = float((min_xy[0] + max_xy[0]) * 0.5)
    cy = float((min_xy[1] + max_xy[1]) * 0.5)
    side_len = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))

    if side_len <= eps:
        return None

    coords_norm = (coords_np - np.array([cx, cy], dtype=np.float64)) / (side_len * 0.5)
    return coords_norm.astype(np.float32), cx, cy, side_len


def _augment_triangles(tris: np.ndarray, rng: np.random.Generator, scale_min: float = 0.5, scale_max: float = 1.0) -> np.ndarray:
    """Apply geometric augmentation to triangle arrays.

    Args:
        tris: Triangle array `[T,3,2]`.
        rng: NumPy random generator.
        scale_min: Minimum random scale.
        scale_max: Maximum random scale.

    Returns:
        Augmented triangle array.
    """
    pts = tris.reshape(-1, 2).astype(np.float32, copy=True)

    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    pts = pts.dot(rot)

    min_c = pts.min(axis=0)
    max_c = pts.max(axis=0)
    span = float(np.max(max_c - min_c))
    if span <= 1e-6:
        return tris.astype(np.float32, copy=True)

    fit_scale = min(1.0, (2.0 - 1e-6) / span)
    random_scale = float(rng.uniform(scale_min, scale_max))
    scale = min(random_scale, fit_scale)
    pts *= scale

    min_c = pts.min(axis=0)
    max_c = pts.max(axis=0)
    tx_low, tx_high = -1.0 - min_c[0], 1.0 - max_c[0]
    ty_low, ty_high = -1.0 - min_c[1], 1.0 - max_c[1]

    tx = float(rng.uniform(tx_low, tx_high)) if tx_low <= tx_high else float((tx_low + tx_high) * 0.5)
    ty = float(rng.uniform(ty_low, ty_high)) if ty_low <= ty_high else float((ty_low + ty_high) * 0.5)
    pts += np.array([tx, ty], dtype=np.float32)

    return pts.reshape(tris.shape).astype(np.float32)


def _build_meta(cx: float, cy: float, side_len: float, node_count: int) -> np.ndarray:
    """Build physical metadata vector for one polygon sample.

    Args:
        cx: Physical bbox center x.
        cy: Physical bbox center y.
        side_len: Physical bbox square side length.
        node_count: Original polygon node count.

    Returns:
        Metadata array `[cx, cy, side_len, node_count]`.
    """
    return np.array([cx, cy, side_len, float(node_count)], dtype=np.float32)


def _normalize_output_mode(mode_value: str | int) -> int:
    """Normalize user-provided output mode into canonical integer id.

    Supported modes:
    1) `meta(4) + embedding(N) + triangles(T,3,2)`.
    2) `meta(4) + embedding(N)`.

    Args:
        mode_value: Raw CLI mode value.

    Returns:
        Canonical mode id (1 or 2).

    Raises:
        ValueError: If mode is unsupported.
    """
    key = str(mode_value).strip().lower()
    if key in {"1", "mode1", "meta_embedding_triangles", "with_triangles"}:
        return 1
    if key in {"2", "mode2", "meta_embedding", "no_triangles", "without_triangles"}:
        return 2
    raise ValueError(
        "Unsupported output mode. Use mode 1 (`meta+embedding+triangles`) "
        "or mode 2 (`meta+embedding`)."
    )


def _resolve_model_artifacts(model_dir: str) -> tuple[Path, Path]:
    """Resolve encoder weight and config file paths from model directory.

    Args:
        model_dir: Export/checkpoint directory.

    Returns:
        Tuple `(encoder_weight_path, config_path)`.
    """
    base = Path(model_dir).expanduser()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {base}")

    config_candidates = []
    for path in [base / "config.yaml", base / "config.yml", base / "poly_ae_config.json", base / "poly_mae_config.json"]:
        if path.exists():
            config_candidates.append(path)
    config_candidates.extend(sorted(path for path in base.glob("*.yaml") if path not in config_candidates))
    config_candidates.extend(sorted(path for path in base.glob("*.yml") if path not in config_candidates))
    config_candidates.extend(sorted(path for path in base.glob("*.json") if path not in config_candidates))
    if not config_candidates:
        raise FileNotFoundError(f"No config file found in model_dir: {base}")

    weight_candidates = []
    for path in [base / "encoder.pth"]:
        if path.exists():
            weight_candidates.append(path)
    weight_candidates.extend(sorted(path for path in base.glob("poly_encoder_epoch_*.pth") if path not in weight_candidates))
    weight_candidates.extend(sorted(path for path in base.glob("*encoder*.pth") if path not in weight_candidates))
    if not weight_candidates:
        raise FileNotFoundError(f"No encoder checkpoint found in model_dir: {base}")

    return weight_candidates[0], config_candidates[0]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Batch encode vector polygons")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--recursive", dest="recursive", action="store_true")
    parser.add_argument("--no_recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=True)

    parser.add_argument("--augment_times", type=int, default=10)
    parser.add_argument("--scale_min", type=float, default=0.5)
    parser.add_argument("--scale_max", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default=_default_device_for_runtime())
    parser.add_argument("--precision", type=str, default="bf16")

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument(
        "--output_mode",
        type=str,
        default="1",
        help=(
            "Output schema mode: "
            "1 => meta+embedding+triangles; "
            "2 => meta+embedding."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    """CLI main function for batch encoding."""
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    import torch

    if __package__ in {None, ""}:
        import importlib

        build_poly_fourier_converter_from_config = importlib.import_module(
            "ae_pretrain.src.datasets.geometry_polygon"
        ).build_poly_fourier_converter_from_config
        PolyEncoderPipeline = importlib.import_module(
            "ae_pretrain.src.engine.pipeline"
        ).PolyEncoderPipeline
        load_config_any = importlib.import_module(
            "ae_pretrain.src.utils.config"
        ).load_config_any
        ensure_dir = importlib.import_module(
            "ae_pretrain.src.utils.filesystem"
        ).ensure_dir
        normalize_precision = importlib.import_module(
            "ae_pretrain.src.utils.precision"
        ).normalize_precision
    else:
        from ..src.datasets.geometry_polygon import build_poly_fourier_converter_from_config
        from ..src.engine.pipeline import PolyEncoderPipeline
        from ..src.utils.config import load_config_any
        from ..src.utils.filesystem import ensure_dir
        from ..src.utils.precision import normalize_precision

    args = build_arg_parser().parse_args()
    args.precision = normalize_precision(args.precision)
    args.output_mode = _normalize_output_mode(args.output_mode)

    vector_files = _iter_vector_files([args.data_dir], recursive=args.recursive)
    if not vector_files:
        raise FileNotFoundError(f"No .shp/.geojson files found in {args.data_dir}")

    encoder_weight, encoder_config = _resolve_model_artifacts(args.model_dir)
    config = load_config_any(str(encoder_config))
    triangulator = build_poly_fourier_converter_from_config(config, device="cpu")

    rng = np.random.default_rng(args.seed)

    base_records = []
    skipped_invalid = 0

    for file_path in tqdm(vector_files, desc="Reading vectors"):
        try:
            gdf = gpd.read_file(file_path)
        except Exception as exc:
            print(f"[WARN] Failed reading {file_path}: {exc}")
            continue

        for row_idx, geom in enumerate(gdf.geometry):
            polygons = _expand_polygons(geom)
            for part_idx, poly in enumerate(polygons):
                coords = np.asarray(poly.exterior.coords, dtype=np.float32)
                if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
                    coords = coords[:-1]

                node_count = int(coords.shape[0])
                if node_count < 3:
                    skipped_invalid += 1
                    continue

                normalized = _normalize_with_bbox_limit(coords)
                if normalized is None:
                    skipped_invalid += 1
                    continue
                norm_coords, cx_phy, cy_phy, side_len_phy = normalized

                tris = triangulator.triangulate_polygon(norm_coords)
                if tris.shape[0] == 0:
                    skipped_invalid += 1
                    continue

                base_records.append(
                    {
                        "triangles": tris.astype(np.float32),
                        "meta": _build_meta(
                            cx=cx_phy,
                            cy=cy_phy,
                            side_len=side_len_phy,
                            node_count=node_count,
                        ),
                        "node_count": node_count,
                        "src_file": str(file_path),
                        "row_idx": row_idx,
                        "part_idx": part_idx,
                    }
                )

    if not base_records:
        raise RuntimeError("No valid polygon samples found.")

    planned_total = len(base_records) * (args.augment_times + 1)
    print(f"[INFO] Base valid samples: {len(base_records)}")
    print(f"[INFO] Planned total samples (with augmentation): {planned_total}")
    print(f"[INFO] Invalid skipped: {skipped_invalid}")

    if args.dry_run:
        print("[DRY-RUN] Completed checks without encoding.")
        return

    pipeline = PolyEncoderPipeline(
        weight_path=str(encoder_weight),
        config_path=str(encoder_config),
        device=args.device,
        precision=args.precision,
    )
    pipeline.encoder.eval()

    pending_tris = []
    pending_meta = []
    output_samples = []

    def flush_pending() -> None:
        """Encode and flush pending samples into output list."""
        if not pending_tris:
            return

        emb = pipeline.triangles_to_embedding(pending_tris).cpu()
        for i, tri_np in enumerate(pending_tris):
            sample = {
                "meta": torch.from_numpy(pending_meta[i]),
                "embedding": emb[i],
            }
            if args.output_mode == 1:
                sample["triangles"] = torch.from_numpy(tri_np.astype(np.float32))
            output_samples.append(sample)

        pending_tris.clear()
        pending_meta.clear()

    for rec in tqdm(base_records, desc="Augment + Encode"):
        base_tri = rec["triangles"]
        base_meta = rec["meta"]

        for aug_id in range(args.augment_times + 1):
            if aug_id == 0:
                tri_aug = base_tri.astype(np.float32, copy=True)
            else:
                tri_aug = _augment_triangles(
                    base_tri,
                    rng=rng,
                    scale_min=args.scale_min,
                    scale_max=args.scale_max,
                )

            # Metadata describes physical bbox of the original polygon and
            # must stay unchanged across augmented variants.
            meta = base_meta
            pending_tris.append(tri_aug)
            pending_meta.append(meta)

            if len(pending_tris) >= args.batch_size:
                flush_pending()

    flush_pending()

    if args.output_path:
        output_path = Path(args.output_path).expanduser()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = project_root / "data" / "emb" / f"encoded_samples_{timestamp}.pt"

    ensure_dir(output_path.parent)
    torch.save(output_samples, str(output_path))

    if output_samples:
        emb_dim = int(output_samples[0]["embedding"].numel())
        print(f"[DONE] Saved samples to: {output_path}")
        print(f"[DONE] Sample count: {len(output_samples)}")
        print(f"[DONE] Embedding dimension: {emb_dim}")
        if args.output_mode == 1:
            print(f"[DONE] Output mode: 1 (meta+embedding+triangles)")
            print(f"[DONE] Sample schema: {{'meta':(4), 'embedding':({emb_dim}), 'triangles':(T,3,2)}}")
        else:
            print(f"[DONE] Output mode: 2 (meta+embedding)")
            print(f"[DONE] Sample schema: {{'meta':(4), 'embedding':({emb_dim})}}")
    else:
        print("[DONE] No samples were produced.")


if __name__ == "__main__":
    main()
