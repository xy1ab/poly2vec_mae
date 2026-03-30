"""Dataset building utilities for polygon triangulation with robustness controls.

This module reads vector files, normalizes polygons, triangulates each polygon
(including MultiPolygon parts and donut holes), filters degenerate triangles,
and saves triangle tensors into one or multiple `.pt` shards.

Key features:
1) File-level parallel triangulation with process pool.
2) Stable file-order merge even when futures complete out-of-order.
3) Size-based output sharding for large datasets.
4) Degenerate-triangle filtering (small-area / near-collinear).
5) Optional triangulation log output for quality auditing.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import Point, Polygon
import triangle as tr
from tqdm import tqdm


_PER_SAMPLE_OVERHEAD_BYTES = 1024
_NORMALIZATION_EPS = 1e-6


def _normalize_file_type(file_type: str) -> str:
    """Normalize user-provided file type token.

    Args:
        file_type: Raw file type string.

    Returns:
        Canonical file type in {"shp", "geojs", "gdb"}.

    Raises:
        ValueError: If file_type is unsupported.
    """
    key = str(file_type).strip().lower()
    if key in {"shp", "shape", "shapefile"}:
        return "shp"
    if key in {"geojs", "geojson"}:
        return "geojs"
    if key in {"gdb", "filegdb", "geodatabase"}:
        return "gdb"
    raise ValueError(f"Unsupported file_type: {file_type}. Use one of: shp, gdb, geojs")


def _list_gdb_layers(gdb_path: Path) -> list[str]:
    """List all layers inside one `.gdb` dataset path.

    Args:
        gdb_path: File geodatabase directory path.

    Returns:
        Layer name list. Empty list when listing fails.
    """
    try:
        import pyogrio

        layers_arr = pyogrio.list_layers(str(gdb_path))
        if layers_arr is None:
            return []
        names: list[str] = []
        for row in layers_arr:
            if len(row) > 0:
                names.append(str(row[0]))
        return names
    except Exception:
        pass

    try:
        import fiona

        return [str(name) for name in fiona.listlayers(str(gdb_path))]
    except Exception:
        return []


def _collect_vector_tasks(
    input_dirs: Iterable[str | Path],
    file_type: str,
    layer: str,
) -> list[dict[str, str | None]]:
    """Collect vector read tasks according to requested file type.

    Args:
        input_dirs: Directory list to scan.
        file_type: Canonical file type token.
        layer: Layer selector when `file_type == "gdb"`.

    Returns:
        Stable task list. Each task contains:
        `{"path": <str>, "layer": <str|None>, "source_type": <str>}`.
    """
    tasks: list[dict[str, str | None]] = []

    canonical = _normalize_file_type(file_type)
    layer_key = str(layer).strip() if layer is not None else "all"

    for directory in input_dirs:
        directory = Path(directory)
        if canonical == "shp":
            shp_files = sorted(Path(p) for p in glob.glob(str(directory / "**" / "*.shp"), recursive=True))
            tasks.extend({"path": str(path), "layer": None, "source_type": "shp"} for path in shp_files)
            continue

        if canonical == "geojs":
            geojson_files = sorted(Path(p) for p in glob.glob(str(directory / "**" / "*.geojson"), recursive=True))
            geojs_files = sorted(Path(p) for p in glob.glob(str(directory / "**" / "*.geojs"), recursive=True))
            all_files = sorted(set(geojson_files + geojs_files))
            tasks.extend({"path": str(path), "layer": None, "source_type": "geojs"} for path in all_files)
            continue

        # canonical == "gdb"
        gdb_dirs = sorted(
            Path(p) for p in glob.glob(str(directory / "**" / "*.gdb"), recursive=True) if Path(p).is_dir()
        )
        for gdb_dir in gdb_dirs:
            if layer_key.lower() != "all":
                tasks.append({"path": str(gdb_dir), "layer": layer_key, "source_type": "gdb"})
                continue

            layer_list = _list_gdb_layers(gdb_dir)
            for layer_name in layer_list:
                tasks.append({"path": str(gdb_dir), "layer": layer_name, "source_type": "gdb"})

    # Stable unique tasks.
    dedup: dict[tuple[str, str | None, str], dict[str, str | None]] = {}
    for task in tasks:
        key = (str(task["path"]), task.get("layer"), str(task.get("source_type")))
        dedup[key] = task
    return [dedup[key] for key in sorted(dedup.keys())]


def _clean_ring_coords(coords: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    """Clean ring coordinates by removing closure duplicate and repeated vertices.

    Args:
        coords: Ring coordinates, typically shaped `[N,2]` or `[N,3]`.
        eps: Duplicate-point tolerance.

    Returns:
        Cleaned ring coordinates shaped `[M,2]`, or None if invalid.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None

    arr = arr[:, :2]
    if arr.shape[0] < 3:
        return None

    if np.allclose(arr[0], arr[-1], atol=eps, rtol=0.0):
        arr = arr[:-1]

    if arr.shape[0] < 3:
        return None

    keep = np.ones(arr.shape[0], dtype=bool)
    diffs = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
    keep[1:] = diffs > eps
    arr = arr[keep]

    if arr.shape[0] < 3:
        return None

    return arr.astype(np.float32)


def _normalize_polygon_to_unit_box(poly: Polygon, eps: float = _NORMALIZATION_EPS) -> Polygon | None:
    """Normalize one polygon to `[-1,1]` by its bbox max side length.

    Args:
        poly: Input polygon geometry.
        eps: Degenerate bbox threshold.

    Returns:
        Normalized polygon, or None when geometry is invalid/degenerate.
    """
    if poly is None or poly.is_empty:
        return None

    minx, miny, maxx, maxy = poly.bounds
    cx = (float(minx) + float(maxx)) * 0.5
    cy = (float(miny) + float(maxy)) * 0.5
    half_side = max(float(maxx) - float(minx), float(maxy) - float(miny)) * 0.5

    if half_side < eps:
        return None

    ext = _clean_ring_coords(np.asarray(poly.exterior.coords))
    if ext is None:
        return None

    holes_norm: list[np.ndarray] = []
    for interior in poly.interiors:
        hole = _clean_ring_coords(np.asarray(interior.coords))
        if hole is None:
            continue
        hole_norm = (hole - np.array([cx, cy], dtype=np.float32)) / float(half_side)
        if hole_norm.shape[0] >= 3:
            holes_norm.append(hole_norm.astype(np.float32))

    ext_norm = (ext - np.array([cx, cy], dtype=np.float32)) / float(half_side)

    try:
        poly_norm = Polygon(ext_norm, holes_norm)
    except Exception:
        return None

    poly_norm = poly_norm.buffer(0)
    if poly_norm.is_empty or poly_norm.geom_type != "Polygon":
        return None

    return poly_norm


def _build_triangle_input(poly: Polygon) -> dict[str, Any] | None:
    """Build Triangle-library input dictionary for one polygon with holes.

    Args:
        poly: Polygon in normalized coordinates.

    Returns:
        Dictionary with `vertices/segments/holes`, or None when invalid.
    """
    vertices: list[tuple[float, float]] = []
    segments: list[tuple[int, int]] = []
    holes: list[tuple[float, float]] = []

    def add_ring(ring_coords: np.ndarray) -> tuple[int, int] | None:
        cleaned = _clean_ring_coords(ring_coords)
        if cleaned is None:
            return None
        start = len(vertices)
        ring_n = int(cleaned.shape[0])
        for xy in cleaned:
            vertices.append((float(xy[0]), float(xy[1])))
        for i in range(ring_n):
            segments.append((start + i, start + ((i + 1) % ring_n)))
        return start, ring_n

    ext = np.asarray(poly.exterior.coords)
    ext_info = add_ring(ext)
    if ext_info is None:
        return None

    for interior in poly.interiors:
        hole_coords = np.asarray(interior.coords)
        hole_info = add_ring(hole_coords)
        if hole_info is None:
            continue
        ring_poly = Polygon(_clean_ring_coords(hole_coords))
        if ring_poly.is_valid and not ring_poly.is_empty:
            rep_pt = ring_poly.representative_point()
            holes.append((float(rep_pt.x), float(rep_pt.y)))

    if len(vertices) < 3:
        return None

    tri_input: dict[str, Any] = {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "segments": np.asarray(segments, dtype=np.int32),
    }
    if holes:
        tri_input["holes"] = np.asarray(holes, dtype=np.float64)

    return tri_input


def _triangulate_polygon_with_holes(poly_norm: Polygon) -> np.ndarray:
    """Triangulate one normalized polygon while honoring interior holes.

    Args:
        poly_norm: Normalized polygon geometry.

    Returns:
        Triangle array `[T,3,2]` in float32. Empty when triangulation fails.
    """
    tri_input = _build_triangle_input(poly_norm)
    if tri_input is None:
        return np.zeros((0, 3, 2), dtype=np.float32)

    try:
        tri_data = tr.triangulate(tri_input, "pq")
    except Exception:
        return np.zeros((0, 3, 2), dtype=np.float32)

    vertices = tri_data.get("vertices")
    tri_index = tri_data.get("triangles")
    if vertices is None or tri_index is None:
        return np.zeros((0, 3, 2), dtype=np.float32)

    tris = np.asarray(vertices, dtype=np.float64)[np.asarray(tri_index, dtype=np.int64)]
    if tris.ndim != 3 or tris.shape[1:] != (3, 2):
        return np.zeros((0, 3, 2), dtype=np.float32)

    # Final safety check: keep only triangles whose centroids are covered by the polygon.
    # This avoids accidental hole leakage from triangulation edge cases.
    centroids = tris.mean(axis=1)
    keep = np.array([poly_norm.covers(Point(float(c[0]), float(c[1]))) for c in centroids], dtype=bool)
    tris = tris[keep]

    return tris.astype(np.float32)


def _filter_degenerate_triangles(
    tris: np.ndarray,
    min_triangle_area: float,
    min_triangle_height: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Filter degenerate triangles by area and near-collinear shape tests.

    Args:
        tris: Triangle array `[T,3,2]`.
        min_triangle_area: Minimum allowed triangle area.
        min_triangle_height: Minimum allowed triangle altitude proxy.

    Returns:
        Tuple `(kept_tris, stats)` where `stats` contains filter counts.
    """
    tri_np = np.asarray(tris, dtype=np.float64)
    if tri_np.ndim != 3 or tri_np.shape[1:] != (3, 2) or tri_np.shape[0] == 0:
        return np.zeros((0, 3, 2), dtype=np.float32), {
            "filtered_total": 0,
            "filtered_by_area_small": 0,
            "filtered_by_near_collinear": 0,
            "kept_count": 0,
        }

    a = tri_np[:, 0, :]
    b = tri_np[:, 1, :]
    c = tri_np[:, 2, :]

    ab = b - a
    ac = c - a
    bc = c - b

    area = 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])

    len_ab = np.linalg.norm(ab, axis=1)
    len_ac = np.linalg.norm(ac, axis=1)
    len_bc = np.linalg.norm(bc, axis=1)
    max_edge = np.maximum(np.maximum(len_ab, len_ac), len_bc)

    altitude = (2.0 * area) / np.maximum(max_edge, 1e-12)

    area_small = area < float(min_triangle_area)
    near_collinear_raw = altitude < float(min_triangle_height)
    # Make type counts mutually exclusive: area_small takes precedence.
    near_collinear = near_collinear_raw & (~area_small)
    remove_mask = area_small | near_collinear

    kept = tri_np[~remove_mask].astype(np.float32)

    return kept, {
        "filtered_total": int(remove_mask.sum()),
        "filtered_by_area_small": int(area_small.sum()),
        "filtered_by_near_collinear": int(near_collinear.sum()),
        "kept_count": int(kept.shape[0]),
    }


def _expand_geometry_to_polygons(geom) -> list[tuple[Polygon, bool]]:
    """Expand a geometry into polygon parts.

    Args:
        geom: Shapely geometry object.

    Returns:
        List of tuples `(polygon, from_multipolygon)`.
    """
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [(geom, False)]

    if geom.geom_type == "MultiPolygon":
        return [(poly, True) for poly in geom.geoms]

    return []


def _triangulate_file_worker(
    file_index: int,
    file_path: str,
    layer_name: str | None,
    source_type: str,
    min_triangle_area: float,
    min_triangle_height: float,
    enable_log: bool,
) -> dict[str, Any]:
    """Triangulate polygons from one vector file in a worker process.

    Args:
        file_index: Stable index of source file in sorted file list.
        file_path: Source vector path.
        layer_name: Layer name when source is `.gdb`, else None.
        source_type: Input source type token (`shp` / `geojs` / `gdb`).
        min_triangle_area: Minimum area threshold in normalized space.
        min_triangle_height: Minimum height threshold in normalized space.
        enable_log: Whether to collect detailed degenerate sample records.

    Returns:
        Dictionary containing worker status, triangle outputs, and statistics.
    """
    out_triangles: list[np.ndarray] = []
    skipped_count = 0

    total_samples = 0
    multi_sample_count = 0
    donut_sample_count = 0
    triangulated_raw_sample_count = 0
    normal_sample_count = 0
    degenerate_sample_count = 0
    dropped_sample_count = 0

    degenerate_records: list[dict[str, Any]] = []

    try:
        if layer_name is None:
            gdf = gpd.read_file(file_path)
        else:
            gdf = gpd.read_file(file_path, layer=layer_name)
    except Exception as exc:
        return {
            "index": file_index,
            "file_path": file_path,
            "layer_name": layer_name,
            "source_type": source_type,
            "ok": False,
            "error": str(exc),
            "source_geometry_count": 0,
            "skipped_count": 0,
            "triangles": [],
            "total_samples": 0,
            "multi_sample_count": 0,
            "donut_sample_count": 0,
            "triangulated_raw_sample_count": 0,
            "normal_sample_count": 0,
            "degenerate_sample_count": 0,
            "dropped_sample_count": 0,
            "degenerate_records": [],
        }

    source_geometry_count = int(len(gdf.geometry))

    for row_idx, geom in enumerate(gdf.geometry):
        polygon_parts = _expand_geometry_to_polygons(geom)
        if not polygon_parts:
            skipped_count += 1
            continue

        for part_idx, (poly_raw, from_multi) in enumerate(polygon_parts):
            total_samples += 1
            local_sample_index = total_samples - 1
            if from_multi:
                multi_sample_count += 1

            poly_fixed = poly_raw.buffer(0)
            if poly_fixed.is_empty:
                skipped_count += 1
                dropped_sample_count += 1
                continue

            if poly_fixed.geom_type != "Polygon":
                # Non-polygon after repair is treated as dropped sample.
                skipped_count += 1
                dropped_sample_count += 1
                continue

            is_donut = len(poly_fixed.interiors) > 0
            if is_donut:
                donut_sample_count += 1

            poly_norm = _normalize_polygon_to_unit_box(poly_fixed)
            if poly_norm is None:
                skipped_count += 1
                dropped_sample_count += 1
                continue

            tris_raw = _triangulate_polygon_with_holes(poly_norm)
            if tris_raw.shape[0] == 0:
                skipped_count += 1
                dropped_sample_count += 1
                continue

            triangulated_raw_sample_count += 1

            tris_kept, filter_stats = _filter_degenerate_triangles(
                tris_raw,
                min_triangle_area=min_triangle_area,
                min_triangle_height=min_triangle_height,
            )

            filtered_total = int(filter_stats["filtered_total"])
            if filtered_total > 0:
                degenerate_sample_count += 1
                if enable_log:
                    degenerate_records.append(
                        {
                            "local_sample_index": int(local_sample_index),
                            "file_path": str(file_path),
                            "layer_name": layer_name,
                            "source_type": source_type,
                            "row_idx": int(row_idx),
                            "part_idx": int(part_idx),
                            "from_multipolygon": bool(from_multi),
                            "is_donut": bool(is_donut),
                            "filtered_triangle_count": filtered_total,
                            "filtered_by_area_small": int(filter_stats["filtered_by_area_small"]),
                            "filtered_by_near_collinear": int(filter_stats["filtered_by_near_collinear"]),
                            "kept_triangle_count": int(filter_stats["kept_count"]),
                        }
                    )

            if tris_kept.shape[0] > 0:
                out_triangles.append(tris_kept.astype(np.float32))
                if filtered_total == 0:
                    normal_sample_count += 1
            else:
                skipped_count += 1
                dropped_sample_count += 1

    return {
        "index": file_index,
        "file_path": file_path,
        "layer_name": layer_name,
        "source_type": source_type,
        "ok": True,
        "error": "",
        "source_geometry_count": source_geometry_count,
        "skipped_count": skipped_count,
        "triangles": out_triangles,
        "total_samples": int(total_samples),
        "multi_sample_count": int(multi_sample_count),
        "donut_sample_count": int(donut_sample_count),
        "triangulated_raw_sample_count": int(triangulated_raw_sample_count),
        "normal_sample_count": int(normal_sample_count),
        "degenerate_sample_count": int(degenerate_sample_count),
        "dropped_sample_count": int(dropped_sample_count),
        "degenerate_records": degenerate_records,
    }


def _estimate_triangle_sample_bytes(tris: np.ndarray) -> int:
    """Estimate serialized byte contribution for one triangle sample.

    Args:
        tris: Triangle array `[T,3,2]`.

    Returns:
        Estimated serialized bytes.
    """
    return int(tris.nbytes) + _PER_SAMPLE_OVERHEAD_BYTES


def _build_shard_path(base_output_path: Path, part_index: int) -> Path:
    """Build shard output path from base path and part index.

    Args:
        base_output_path: User-provided output base path.
        part_index: 1-based shard index.

    Returns:
        Shard file path.
    """
    suffix = base_output_path.suffix if base_output_path.suffix else ".pt"
    return base_output_path.with_name(f"{base_output_path.stem}.part{part_index:04d}{suffix}")


def _default_log_path(output_path: Path) -> Path:
    """Build default triangulation-log path beside output `.pt` base file.

    Args:
        output_path: Output `.pt` base path.

    Returns:
        JSON log path under the same directory.
    """
    return output_path.with_name(f"{output_path.stem}.triangulation_log.json")


class _ShardWriter:
    """Incremental writer that flushes samples to `.pt` shards by size estimate."""

    def __init__(self, output_path: Path, shard_size_mb: float):
        """Initialize shard writer.

        Args:
            output_path: Base output path.
            shard_size_mb: Target shard size in MB. `<=0` means single-file output.
        """
        self.output_path = output_path
        self.shard_size_bytes = int(max(0.0, float(shard_size_mb)) * 1024.0 * 1024.0)
        self._buffer: list[np.ndarray] = []
        self._buffer_estimated_bytes = 0

        self.shard_paths: list[Path] = []
        self.shard_sample_counts: list[int] = []

    def add_many(self, samples: list[np.ndarray]) -> None:
        """Add multiple samples into current shard buffer.

        Args:
            samples: Triangle sample list.
        """
        for sample in samples:
            est_bytes = _estimate_triangle_sample_bytes(sample)
            if self.shard_size_bytes > 0 and self._buffer and (self._buffer_estimated_bytes + est_bytes > self.shard_size_bytes):
                self.flush()
            self._buffer.append(sample)
            self._buffer_estimated_bytes += est_bytes

    def flush(self) -> Path | None:
        """Flush current buffer to disk as one `.pt` file.

        Returns:
            Written file path, or None when buffer is empty.
        """
        if not self._buffer:
            return None

        if self.shard_size_bytes > 0:
            output_file = _build_shard_path(self.output_path, len(self.shard_paths) + 1)
        else:
            output_file = self.output_path

        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._buffer, output_file)

        self.shard_paths.append(output_file)
        self.shard_sample_counts.append(len(self._buffer))

        self._buffer = []
        self._buffer_estimated_bytes = 0
        return output_file

    def finalize(self) -> tuple[list[Path], Path | None]:
        """Flush remaining data and optionally write shard manifest.

        Returns:
            Tuple `(shard_paths, manifest_path)`.
        """
        self.flush()
        if not self.shard_paths:
            return [], None

        if self.shard_size_bytes <= 0:
            return self.shard_paths, None

        manifest_path = self.output_path.with_name(f"{self.output_path.stem}.manifest.json")
        manifest = {
            "base_output_path": str(self.output_path),
            "shard_size_mb": self.shard_size_bytes / (1024.0 * 1024.0),
            "num_shards": len(self.shard_paths),
            "total_samples": int(sum(self.shard_sample_counts)),
            "shards": [
                {
                    "path": str(path),
                    "sample_count": int(sample_count),
                    "size_bytes": int(path.stat().st_size) if path.exists() else -1,
                }
                for path, sample_count in zip(self.shard_paths, self.shard_sample_counts)
            ],
        }
        with manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(manifest, fp, ensure_ascii=False, indent=2)

        return self.shard_paths, manifest_path


def _resolve_num_workers(num_workers: int, file_count: int) -> int:
    """Resolve effective process count for file-level parallelism.

    Args:
        num_workers: Requested worker count. `<=0` means auto.
        file_count: Number of source files.

    Returns:
        Effective worker count in `[1, file_count]`.
    """
    if file_count <= 1:
        return 1

    if int(num_workers) <= 0:
        cpu_count = os.cpu_count() or 1
        auto_workers = max(1, cpu_count - 1)
        return max(1, min(auto_workers, file_count))

    return max(1, min(int(num_workers), file_count))


def _consume_worker_result(result: dict[str, Any], writer: _ShardWriter) -> dict[str, Any]:
    """Consume one worker result and append triangles to shard writer.

    Args:
        result: Worker result dictionary.
        writer: Shard writer instance.

    Returns:
        Aggregated scalar statistics dictionary for this worker result.
    """
    ok = bool(result.get("ok", False))

    stats = {
        "ok": ok,
        "source_geometry_count": int(result.get("source_geometry_count", 0)),
        "skipped_count": int(result.get("skipped_count", 0)),
        "triangulated_count": 0,
        "total_samples": int(result.get("total_samples", 0)),
        "multi_sample_count": int(result.get("multi_sample_count", 0)),
        "donut_sample_count": int(result.get("donut_sample_count", 0)),
        "triangulated_raw_sample_count": int(result.get("triangulated_raw_sample_count", 0)),
        "normal_sample_count": int(result.get("normal_sample_count", 0)),
        "degenerate_sample_count": int(result.get("degenerate_sample_count", 0)),
        "dropped_sample_count": int(result.get("dropped_sample_count", 0)),
        "degenerate_records": list(result.get("degenerate_records", [])),
    }

    if not ok:
        file_path = result.get("file_path", "<unknown>")
        layer_name = result.get("layer_name")
        error = result.get("error", "unknown error")
        if layer_name is None:
            tqdm.write(f"[WARN] Failed to read/process {file_path}: {error}")
        else:
            tqdm.write(f"[WARN] Failed to read/process {file_path} (layer={layer_name}): {error}")
        return stats

    triangles = result.get("triangles", [])
    stats["triangulated_count"] = int(len(triangles))
    writer.add_many(triangles)
    return stats


def process_and_save(
    input_dirs: Iterable[str | Path],
    output_path: str | Path,
    file_type: str = "shp",
    layer: str = "all",
    num_workers: int = 0,
    shard_size_mb: float = 0.0,
    min_triangle_area: float = 1e-8,
    min_triangle_height: float = 1e-5,
    log: bool = False,
) -> None:
    """Build triangulated polygon dataset and save to disk.

    Args:
        input_dirs: Iterable of source directories containing vector files.
        output_path: Output `.pt` base path.
        file_type: Input vector source type (`shp`, `gdb`, `geojs`).
        layer: Layer selector when `file_type='gdb'`. Use `'all'` for all layers.
        num_workers: File-level process count. `<=0` means auto.
        shard_size_mb: Target shard size in MB. `<=0` means single `.pt` output.
        min_triangle_area: Minimum triangle area threshold in normalized space.
        min_triangle_height: Minimum altitude threshold in normalized space.
        log: Whether to save triangulation audit log JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_file_type = _normalize_file_type(file_type)
    task_list = _collect_vector_tasks(input_dirs, file_type=canonical_file_type, layer=layer)
    if not task_list:
        if canonical_file_type == "gdb":
            print(f"[WARN] No .gdb layer tasks found under given input_dirs (layer={layer}).")
        elif canonical_file_type == "geojs":
            print("[WARN] No .geojson/.geojs files found under given input_dirs.")
        else:
            print("[WARN] No .shp files found under given input_dirs.")
        return

    worker_count = _resolve_num_workers(num_workers=num_workers, file_count=len(task_list))
    shard_size_mb = float(shard_size_mb)

    print(f"[INFO] Input file_type         : {canonical_file_type}")
    if canonical_file_type == "gdb":
        print(f"[INFO] Input layer selector    : {layer}")
    print(f"[INFO] Discovered input tasks : {len(task_list)}")
    print(f"[INFO] File-level workers: {worker_count}")
    print(f"[INFO] Degenerate filter: min_triangle_area={min_triangle_area:.3e}, min_triangle_height={min_triangle_height:.3e}")
    if shard_size_mb > 0:
        print(f"[INFO] Sharding enabled: target {shard_size_mb:.2f} MB per output .pt")
    else:
        print("[INFO] Sharding disabled: single output .pt")

    writer = _ShardWriter(output_path=output_path, shard_size_mb=shard_size_mb)

    files_ok = 0
    files_failed = 0
    source_geometries_total = 0
    skipped_total = 0
    triangulated_total = 0

    total_samples = 0
    multi_samples = 0
    donut_samples = 0
    triangulated_raw_samples = 0
    normal_samples = 0
    degenerate_samples = 0
    dropped_samples = 0

    global_sample_cursor = 0
    degenerate_log_records: list[dict[str, Any]] = []

    def consume_and_merge(result: dict[str, Any]) -> None:
        """Merge one ordered worker result into global counters and writer.

        Args:
            result: Worker output dictionary.
        """
        nonlocal files_ok, files_failed
        nonlocal source_geometries_total, skipped_total, triangulated_total
        nonlocal total_samples, multi_samples, donut_samples
        nonlocal triangulated_raw_samples, normal_samples, degenerate_samples, dropped_samples
        nonlocal global_sample_cursor

        merged = _consume_worker_result(result, writer)

        files_ok += int(merged["ok"])
        files_failed += int(not merged["ok"])

        source_geometries_total += int(merged["source_geometry_count"])
        skipped_total += int(merged["skipped_count"])
        triangulated_total += int(merged["triangulated_count"])

        local_sample_count = int(merged["total_samples"])
        total_samples += local_sample_count
        multi_samples += int(merged["multi_sample_count"])
        donut_samples += int(merged["donut_sample_count"])
        triangulated_raw_samples += int(merged["triangulated_raw_sample_count"])
        normal_samples += int(merged["normal_sample_count"])
        degenerate_samples += int(merged["degenerate_sample_count"])
        dropped_samples += int(merged["dropped_sample_count"])

        if log:
            for rec in merged["degenerate_records"]:
                local_idx = int(rec.get("local_sample_index", -1))
                out_rec = dict(rec)
                out_rec["sample_index"] = int(global_sample_cursor + local_idx)
                degenerate_log_records.append(out_rec)

        global_sample_cursor += local_sample_count

    if worker_count <= 1:
        with tqdm(total=len(task_list), desc="Triangulating tasks", unit="task") as pbar:
            for file_index, task in enumerate(task_list):
                result = _triangulate_file_worker(
                    file_index=file_index,
                    file_path=str(task["path"]),
                    layer_name=task.get("layer"),
                    source_type=str(task.get("source_type", canonical_file_type)),
                    min_triangle_area=min_triangle_area,
                    min_triangle_height=min_triangle_height,
                    enable_log=bool(log),
                )
                consume_and_merge(result)
                pbar.update(1)
    else:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
            future_to_meta = {
                executor.submit(
                    _triangulate_file_worker,
                    file_index,
                    str(task["path"]),
                    task.get("layer"),
                    str(task.get("source_type", canonical_file_type)),
                    float(min_triangle_area),
                    float(min_triangle_height),
                    bool(log),
                ): (file_index, task)
                for file_index, task in enumerate(task_list)
            }

            pending_results: dict[int, dict[str, Any]] = {}
            next_index = 0

            with tqdm(total=len(task_list), desc="Triangulating tasks", unit="task") as pbar:
                for future in as_completed(future_to_meta):
                    file_index, task = future_to_meta[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "index": file_index,
                            "file_path": str(task.get("path")),
                            "layer_name": task.get("layer"),
                            "source_type": str(task.get("source_type", canonical_file_type)),
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "source_geometry_count": 0,
                            "skipped_count": 0,
                            "triangles": [],
                            "total_samples": 0,
                            "multi_sample_count": 0,
                            "donut_sample_count": 0,
                            "triangulated_raw_sample_count": 0,
                            "normal_sample_count": 0,
                            "degenerate_sample_count": 0,
                            "dropped_sample_count": 0,
                            "degenerate_records": [],
                        }

                    pending_results[int(result["index"])] = result
                    pbar.update(1)

                    while next_index in pending_results:
                        ordered_result = pending_results.pop(next_index)
                        consume_and_merge(ordered_result)
                        next_index += 1

    if triangulated_total == 0:
        print("[WARN] No valid polygons were triangulated. Nothing to save.")
        shard_paths: list[Path] = []
        manifest_path: Path | None = None
    else:
        shard_paths, manifest_path = writer.finalize()
        if not shard_paths:
            print("[WARN] No output shard was written.")

    if log:
        log_path = _default_log_path(output_path)
        log_payload = {
            "output_base_path": str(output_path),
            "input_file_type": canonical_file_type,
            "input_layer_selector": str(layer),
            "min_triangle_area": float(min_triangle_area),
            "min_triangle_height": float(min_triangle_height),
            "total_samples": int(total_samples),
            "multi_polygon_samples": int(multi_samples),
            "donut_polygon_samples": int(donut_samples),
            "triangulated_raw_samples": int(triangulated_raw_samples),
            "normal_samples": int(normal_samples),
            "degenerate_samples": int(degenerate_samples),
            "dropped_samples": int(dropped_samples),
            "degenerate_sample_records": degenerate_log_records,
        }
        with log_path.open("w", encoding="utf-8") as fp:
            json.dump(log_payload, fp, ensure_ascii=False, indent=2)
        print(f"[INFO] Triangulation log      : {log_path}")

    print("[INFO] Triangulation completed.")
    print(f"[INFO] Tasks succeeded/failed : {files_ok}/{files_failed}")
    print(f"[INFO] Source geometries       : {source_geometries_total}")
    print(f"[INFO] Total samples           : {total_samples}")
    print(f"[INFO] MultiPolygon samples    : {multi_samples}")
    print(f"[INFO] Donut samples           : {donut_samples}")
    print(f"[INFO] Triangulated raw        : {triangulated_raw_samples}")
    print(f"[INFO] Normal samples          : {normal_samples}")
    print(f"[INFO] Degenerate samples      : {degenerate_samples}")
    print(f"[INFO] Dropped samples         : {dropped_samples}")
    print(f"[INFO] Triangulated outputs    : {triangulated_total}")
    print(f"[INFO] Skipped geometries      : {skipped_total}")

    if triangulated_total == 0:
        return

    if shard_size_mb > 0:
        print(f"[INFO] Output shards           : {len(shard_paths)}")
        for shard_path in shard_paths:
            size_mb = shard_path.stat().st_size / (1024.0 * 1024.0)
            print(f"[INFO]   - {shard_path} ({size_mb:.2f} MB)")
        if manifest_path is not None:
            print(f"[INFO] Shard manifest         : {manifest_path}")
    else:
        print(f"[INFO] Saved output           : {shard_paths[0]}")
