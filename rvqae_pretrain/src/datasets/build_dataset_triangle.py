"""Dataset building utilities for polygon triangulation with robustness controls.

This module reads vector files, normalizes polygons, triangulates each polygon
(including MultiPolygon parts and donut holes), filters degenerate triangles,
and saves triangle tensors into one or multiple `.pt` shards.

Serialization note:
Shard files are saved with `torch.save` for full compatibility with existing
training/evaluation readers (`torch.load`).

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
from shapely import wkb as shapely_wkb
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, triangulate as shapely_triangulate, unary_union
import triangle as tr
from tqdm import tqdm

from .shard_io import TORCH_SHARD_SERIALIZATION, save_triangle_shard

try:
    from shapely.validation import make_valid as _shapely_make_valid
except Exception:  # pragma: no cover - compatibility fallback
    _shapely_make_valid = None


# Keep this threshold tiny because source coordinates may be geodetic and very
# small polygons can differ only in deep decimal places. We therefore keep
# float64 through de-centering/normalization and reject only near-zero extents.
_NORMALIZATION_EPS = 1e-12
_TRIANGLE_SUBPROC_TIMEOUT_SEC = 20.0
_SAFE_MODE_ALL = "all"
_SAFE_MODE_RISKY = "risky"
_SAFE_MODE_OFF = "off"
_SAFE_MODE_CHOICES = (_SAFE_MODE_ALL, _SAFE_MODE_RISKY, _SAFE_MODE_OFF)


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
        Cleaned ring coordinates shaped `[M,2]` in float64, or None if invalid.
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

    return arr


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

    center = np.array([cx, cy], dtype=np.float64)
    ext = _clean_ring_coords(np.asarray(poly.exterior.coords))
    if ext is None:
        return None

    holes_norm: list[np.ndarray] = []
    for interior in poly.interiors:
        hole = _clean_ring_coords(np.asarray(interior.coords))
        if hole is None:
            continue
        hole_norm = (hole - center) / float(half_side)
        if hole_norm.shape[0] >= 3:
            holes_norm.append(hole_norm.astype(np.float64, copy=False))

    ext_norm = (ext - center) / float(half_side)

    try:
        poly_norm = Polygon(ext_norm, holes_norm)
    except Exception:
        return None

    poly_norm = poly_norm.buffer(0)
    if poly_norm.is_empty or poly_norm.geom_type != "Polygon":
        return None

    return poly_norm


def _normalize_polygon_with_row_frame(
    poly: Polygon,
    center: np.ndarray,
    half_side: float,
    norm_max: float,
) -> Polygon | None:
    """Normalize one polygon using a row-level center/scale frame.

    Args:
        poly: Input polygon geometry.
        center: Row-level bbox center shaped `[2]`.
        half_side: Row-level longest-bbox half-side length.
        norm_max: Target max absolute coordinate after normalization.

    Returns:
        Normalized polygon without any repair step, or None when construction
        fails or the result is not a polygon.
    """
    if poly is None or poly.is_empty:
        return None
    if float(half_side) < _NORMALIZATION_EPS:
        return None

    scale = float(norm_max) / float(half_side)

    try:
        ext = (np.asarray(poly.exterior.coords, dtype=np.float64)[:, :2] - center) * scale
    except Exception:
        return None

    holes_norm: list[np.ndarray] = []
    for interior in poly.interiors:
        try:
            hole = (np.asarray(interior.coords, dtype=np.float64)[:, :2] - center) * scale
        except Exception:
            return None
        holes_norm.append(hole)

    try:
        poly_norm = Polygon(ext, holes_norm)
    except Exception:
        return None

    if poly_norm.is_empty or poly_norm.geom_type != "Polygon":
        return None
    return poly_norm


def _normalize_row_parts(
    geom,
    norm_max: float,
) -> list[Polygon]:
    """Normalize every polygon part in one row with the same row-level frame.

    Args:
        geom: Shapely geometry from one source row.
        norm_max: Target max absolute coordinate after normalization.

    Returns:
        Normalized polygon-part list in original part order. Parts that fail
        normalization are omitted.
    """
    polygon_parts = _expand_geometry_to_polygons(geom)
    if not polygon_parts:
        return []

    minx, miny, maxx, maxy = geom.bounds
    cx = (float(minx) + float(maxx)) * 0.5
    cy = (float(miny) + float(maxy)) * 0.5
    half_side = max(float(maxx) - float(minx), float(maxy) - float(miny)) * 0.5
    if half_side < _NORMALIZATION_EPS:
        return []

    center = np.array([cx, cy], dtype=np.float64)
    normalized_parts: list[Polygon] = []
    for poly_raw, _from_multi in polygon_parts:
        poly_norm = _normalize_polygon_with_row_frame(
            poly=poly_raw,
            center=center,
            half_side=float(half_side),
            norm_max=float(norm_max),
        )
        if poly_norm is not None:
            normalized_parts.append(poly_norm)
    return normalized_parts


def _polygon_node_count(poly: Polygon) -> int:
    """Count cleaned vertices of one polygon across shell and holes."""
    if poly is None or poly.is_empty:
        return 0
    total = 0
    ext = _clean_ring_coords(np.asarray(poly.exterior.coords))
    if ext is not None:
        total += int(ext.shape[0])
    for interior in poly.interiors:
        hole = _clean_ring_coords(np.asarray(interior.coords))
        if hole is not None:
            total += int(hole.shape[0])
    return total


def _ring_min_edge(coords: np.ndarray) -> float:
    """Compute the minimum edge length of one ring after minimal ring cleanup."""
    cleaned = _clean_ring_coords(coords)
    if cleaned is None or cleaned.shape[0] < 3:
        return float("inf")
    closed = np.concatenate([cleaned, cleaned[:1]], axis=0)
    edge_lengths = np.linalg.norm(closed[1:] - closed[:-1], axis=1)
    if edge_lengths.size == 0:
        return float("inf")
    return float(edge_lengths.min())


def _polygon_min_edge(poly: Polygon) -> float:
    """Compute minimum edge length across shell and hole rings."""
    if poly is None or poly.is_empty:
        return float("inf")
    min_edge = _ring_min_edge(np.asarray(poly.exterior.coords))
    for interior in poly.interiors:
        min_edge = min(min_edge, _ring_min_edge(np.asarray(interior.coords)))
    return float(min_edge)


def _summarize_row_geometry(geom) -> dict[str, Any]:
    """Build row-level geometry summary for logging and statistics.

    Args:
        geom: Raw shapely geometry from one source row.

    Returns:
        Summary dictionary describing the row before normalization/filtering.
    """
    if geom is None or geom.is_empty or geom.geom_type not in {"Polygon", "MultiPolygon"}:
        return {
            "geom_type": getattr(geom, "geom_type", None),
            "is_multipolygon": False,
            "raw_part_count": 0,
            "has_holes": False,
            "parts_with_holes": 0,
            "total_hole_count": 0,
            "max_part_hole_count": 0,
        }

    polygon_parts = _expand_geometry_to_polygons(geom)
    hole_counts = [int(len(poly.interiors)) for poly, _ in polygon_parts]
    total_hole_count = int(sum(hole_counts))
    return {
        "geom_type": str(geom.geom_type),
        "is_multipolygon": bool(geom.geom_type == "MultiPolygon"),
        "raw_part_count": int(len(polygon_parts)),
        "has_holes": bool(total_hole_count > 0),
        "parts_with_holes": int(sum(1 for count in hole_counts if count > 0)),
        "total_hole_count": total_hole_count,
        "max_part_hole_count": int(max(hole_counts) if hole_counts else 0),
    }


def _build_special_row_log_record(
    *,
    file_path: str,
    layer_name: str | None,
    source_type: str,
    row_idx: int,
    profile: dict[str, Any],
    safe_mode: str,
    isolated: bool,
    status: str,
    drop_reason: str,
    filtered_part_count: int,
    filtered_triangle_count: int,
    kept_triangle_count: int,
    degenerated: bool,
) -> dict[str, Any]:
    """Build one detailed row log record for MultiPolygon / hole rows."""
    return {
        "file_path": str(file_path),
        "layer_name": layer_name,
        "source_type": str(source_type),
        "row_idx": int(row_idx),
        "geom_type": profile.get("geom_type"),
        "is_multipolygon": bool(profile.get("is_multipolygon", False)),
        "raw_part_count": int(profile.get("raw_part_count", 0)),
        "has_holes": bool(profile.get("has_holes", False)),
        "parts_with_holes": int(profile.get("parts_with_holes", 0)),
        "total_hole_count": int(profile.get("total_hole_count", 0)),
        "max_part_hole_count": int(profile.get("max_part_hole_count", 0)),
        "safe_mode": str(safe_mode),
        "isolated": bool(isolated),
        "status": str(status),
        "drop_reason": str(drop_reason),
        "degenerated": bool(degenerated),
        "filtered_part_count": int(filtered_part_count),
        "filtered_triangle_count": int(filtered_triangle_count),
        "kept_triangle_count": int(kept_triangle_count),
    }


def _safe_row_profile_for_failure(geom) -> dict[str, Any]:
    """Best-effort geometry summary used when row processing raises unexpectedly."""
    try:
        return _summarize_row_geometry(geom)
    except Exception:
        return {
            "geom_type": str(getattr(geom, "geom_type", "<unknown>")),
            "is_multipolygon": bool(getattr(geom, "geom_type", None) == "MultiPolygon"),
            "raw_part_count": 0,
            "has_holes": False,
            "parts_with_holes": 0,
            "total_hole_count": 0,
            "max_part_hole_count": 0,
        }


def _build_failed_row_record(
    *,
    file_path: str,
    layer_name: str | None,
    source_type: str,
    row_idx: int,
    profile: dict[str, Any],
    error_type: str,
    error_message: str,
    sample_count: int,
) -> dict[str, Any]:
    """Build one row-level failure record for unexpected exceptions."""
    return {
        "file_path": str(file_path),
        "layer_name": layer_name,
        "source_type": str(source_type),
        "row_idx": int(row_idx),
        "geom_type": profile.get("geom_type"),
        "is_multipolygon": bool(profile.get("is_multipolygon", False)),
        "raw_part_count": int(profile.get("raw_part_count", 0)),
        "has_holes": bool(profile.get("has_holes", False)),
        "parts_with_holes": int(profile.get("parts_with_holes", 0)),
        "total_hole_count": int(profile.get("total_hole_count", 0)),
        "max_part_hole_count": int(profile.get("max_part_hole_count", 0)),
        "sample_count": int(sample_count),
        "error_type": str(error_type),
        "error_message": str(error_message),
    }


def _build_chunk_failure_record(
    *,
    chunk_index: int,
    row_count: int,
    row_sample_count: int,
    file_path: str,
    layer_name: str | None,
    source_type: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Build one chunk-level failure record for worker crashes."""
    return {
        "chunk_index": int(chunk_index),
        "file_path": str(file_path),
        "layer_name": layer_name,
        "source_type": str(source_type),
        "row_count": int(row_count),
        "row_sample_count": int(row_sample_count),
        "error_type": str(error_type),
        "error_message": str(error_message),
    }


def _filter_row_parts(normalized_parts: list[Polygon]) -> tuple[list[Polygon], int]:
    """Filter row parts using strict validity and shell-hole-touching rules."""
    kept_parts: list[Polygon] = []
    filtered_count = 0
    for poly in normalized_parts:
        if poly is None or poly.is_empty or poly.geom_type != "Polygon":
            filtered_count += 1
            continue
        if not bool(poly.is_valid):
            filtered_count += 1
            continue
        if _polygon_has_shell_hole_intersection(poly):
            filtered_count += 1
            continue
        kept_parts.append(poly)
    return kept_parts, int(filtered_count)


def _should_isolate_row(
    safe_mode: str,
    filtered_parts: list[Polygon],
    part_safe: int,
    node_safe: int,
    hole_safe: int,
    edge_safe: float,
) -> bool:
    """Decide whether one row should be processed in a safe subprocess."""
    mode = str(safe_mode).strip().lower()
    if mode == _SAFE_MODE_ALL:
        return True
    if mode == _SAFE_MODE_OFF:
        return False

    if len(filtered_parts) > int(part_safe):
        return True
    for poly in filtered_parts:
        if _polygon_node_count(poly) > int(node_safe):
            return True
        if len(poly.interiors) > int(hole_safe):
            return True
        if _polygon_min_edge(poly) < float(edge_safe):
            return True
    return False


def _triangulate_polygon_triangle_only(poly_norm: Polygon) -> np.ndarray:
    """Triangulate one normalized polygon without repair or fallback."""
    tri_input = _build_triangle_input(poly_norm)
    if tri_input is None:
        return np.zeros((0, 3, 2), dtype=np.float32)

    try:
        tri_data = tr.triangulate(tri_input, "pq")
    except Exception:
        return np.zeros((0, 3, 2), dtype=np.float32)

    if not isinstance(tri_data, dict):
        return np.zeros((0, 3, 2), dtype=np.float32)

    vertices = tri_data.get("vertices")
    tri_index = tri_data.get("triangles")
    if vertices is None or tri_index is None:
        return np.zeros((0, 3, 2), dtype=np.float32)

    tris = np.asarray(vertices, dtype=np.float64)[np.asarray(tri_index, dtype=np.int64)]
    if tris.ndim != 3 or tris.shape[1:] != (3, 2):
        return np.zeros((0, 3, 2), dtype=np.float32)

    centroids = tris.mean(axis=1)
    keep = np.array([poly_norm.covers(Point(float(c[0]), float(c[1]))) for c in centroids], dtype=bool)
    tris = tris[keep]
    if tris.shape[0] == 0:
        return np.zeros((0, 3, 2), dtype=np.float32)
    return tris.astype(np.float32)


def _process_normalized_row_parts(
    normalized_parts: list[Polygon],
    min_triangle_area: float,
    min_triangle_height: float,
) -> dict[str, Any]:
    """Process one row after row-level normalization.

    Flow:
    1) Strictly filter invalid or shell-hole-touching parts.
    2) Triangulate each kept part independently.
    3) Merge all part triangles as one row sample.
    4) Filter degenerate triangles on the merged sample.
    """
    filtered_parts, filtered_part_count = _filter_row_parts(normalized_parts)
    had_part_filter = bool(filtered_part_count > 0)
    if not filtered_parts:
        return {
            "ok": False,
            "triangles": None,
            "filtered_part_count": int(filtered_part_count),
            "had_part_filter": had_part_filter,
            "filtered_triangle_count": 0,
            "failure_reason": "all_parts_filtered",
        }

    tri_blocks: list[np.ndarray] = []
    for poly in filtered_parts:
        tris_part = _triangulate_polygon_triangle_only(poly)
        if tris_part.shape[0] == 0:
            return {
                "ok": False,
                "triangles": None,
                "filtered_part_count": int(filtered_part_count),
                "had_part_filter": had_part_filter,
                "filtered_triangle_count": 0,
                "failure_reason": "triangulate_empty",
            }
        tri_blocks.append(tris_part)

    tris_raw = np.concatenate(tri_blocks, axis=0).astype(np.float32)
    tris_kept, filter_stats = _filter_degenerate_triangles(
        tris_raw,
        min_triangle_area=min_triangle_area,
        min_triangle_height=min_triangle_height,
    )
    filtered_total = int(filter_stats["filtered_total"])
    if tris_kept.shape[0] == 0:
        return {
            "ok": False,
            "triangles": None,
            "filtered_part_count": int(filtered_part_count),
            "had_part_filter": had_part_filter,
            "filtered_triangle_count": filtered_total,
            "failure_reason": "all_triangles_filtered",
        }

    return {
        "ok": True,
        "triangles": tris_kept.astype(np.float32),
        "filtered_part_count": int(filtered_part_count),
        "had_part_filter": had_part_filter,
        "filtered_triangle_count": filtered_total,
        "failure_reason": "",
    }


def _row_safe_subprocess_entry(
    conn,
    normalized_parts_wkb: list[bytes],
    min_triangle_area: float,
    min_triangle_height: float,
) -> None:
    """Child-process entry for safe row-level processing."""
    try:
        normalized_parts = [shapely_wkb.loads(blob) for blob in normalized_parts_wkb]
        result = _process_normalized_row_parts(
            normalized_parts=normalized_parts,
            min_triangle_area=float(min_triangle_area),
            min_triangle_height=float(min_triangle_height),
        )
        conn.send({"ok": True, "result": result})
    except Exception as exc:
        try:
            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _process_row_isolated(
    normalized_parts: list[Polygon],
    min_triangle_area: float,
    min_triangle_height: float,
    timeout_safe: float,
) -> dict[str, Any]:
    """Run row processing in an isolated subprocess with timeout control."""
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    payload = [bytes(poly.wkb) for poly in normalized_parts]
    proc = ctx.Process(
        target=_row_safe_subprocess_entry,
        args=(send_conn, payload, float(min_triangle_area), float(min_triangle_height)),
        daemon=True,
    )
    proc.start()
    send_conn.close()

    message: dict[str, Any] | None = None
    try:
        if recv_conn.poll(timeout=max(0.1, float(timeout_safe))):
            message = recv_conn.recv()
    except Exception:
        message = None
    finally:
        try:
            recv_conn.close()
        except Exception:
            pass
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=1.0)

    if message is None:
        return {
            "ok": False,
            "triangles": None,
            "filtered_part_count": 0,
            "had_part_filter": False,
            "filtered_triangle_count": 0,
            "failure_reason": "timeout_safe",
        }
    if not bool(message.get("ok", False)):
        return {
            "ok": False,
            "triangles": None,
            "filtered_part_count": 0,
            "had_part_filter": False,
            "filtered_triangle_count": 0,
            "failure_reason": str(message.get("error", "safe_subprocess_error")),
        }
    result = message.get("result")
    if not isinstance(result, dict):
        return {
            "ok": False,
            "triangles": None,
            "filtered_part_count": 0,
            "had_part_filter": False,
            "filtered_triangle_count": 0,
            "failure_reason": "safe_subprocess_invalid_result",
        }
    return result


def _extract_polygons_from_geometry(geom) -> list[Polygon]:
    """Extract polygon components from a generic geometry object.

    Args:
        geom: Any shapely geometry.

    Returns:
        Flat polygon list. Empty when no polygon components exist.
    """
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if poly is not None and not poly.is_empty]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for sub in geom.geoms:
            out.extend(_extract_polygons_from_geometry(sub))
        return out
    return []


def _polygon_has_shell_hole_intersection(poly: Polygon) -> bool:
    """Check whether polygon shell intersects any hole boundary.

    This catches topology patterns where an inner ring touches the shell at a
    point/segment, which is a known instability trigger for some triangulators.

    Args:
        poly: Input polygon.

    Returns:
        True when shell and at least one hole ring intersect.
    """
    if poly is None or poly.is_empty or len(poly.interiors) == 0:
        return False

    shell_line = LineString(np.asarray(poly.exterior.coords)[:, :2])
    for interior in poly.interiors:
        hole_line = LineString(np.asarray(interior.coords)[:, :2])
        inter = shell_line.intersection(hole_line)
        if not inter.is_empty:
            return True
    return False


def _split_polygon_touching_holes(poly: Polygon) -> list[Polygon]:
    """Split polygons whose holes touch shell using boundary polygonization.

    Args:
        poly: Input polygon geometry.

    Returns:
        Repaired polygon parts. Returns `[poly]` when no split is needed or
        when split fails.
    """
    if poly is None or poly.is_empty:
        return []
    if len(poly.interiors) == 0:
        return [poly]
    if not _polygon_has_shell_hole_intersection(poly):
        return [poly]

    lines = [LineString(np.asarray(poly.exterior.coords)[:, :2])]
    lines.extend(LineString(np.asarray(interior.coords)[:, :2]) for interior in poly.interiors)

    try:
        merged = unary_union(lines)
        pieces = list(polygonize(merged))
    except Exception:
        return [poly]

    out: list[Polygon] = []
    for piece in pieces:
        if piece is None or piece.is_empty:
            continue
        rep_pt = piece.representative_point()
        if not poly.covers(rep_pt):
            continue
        fixed = piece.buffer(0)
        out.extend(_extract_polygons_from_geometry(fixed))

    return out if out else [poly]


def _shrink_touching_holes(poly: Polygon) -> Polygon | None:
    """Try to separate touching shell-hole boundaries by shrinking holes.

    Args:
        poly: Input polygon geometry that may contain touching interior rings.

    Returns:
        Repaired polygon when shrinking succeeds, otherwise None.
    """
    if poly is None or poly.is_empty or len(poly.interiors) == 0:
        return None

    shell = _clean_ring_coords(np.asarray(poly.exterior.coords))
    if shell is None:
        return None

    shell_poly = Polygon(shell)
    if shell_poly.is_empty:
        return None

    # Escalate epsilon gradually to avoid over-shrinking small holes.
    for eps in (1e-9, 1e-8, 1e-7, 1e-6):
        hole_parts: list[Polygon] = []
        for interior in poly.interiors:
            hole = _clean_ring_coords(np.asarray(interior.coords))
            if hole is None:
                continue
            ring_poly = Polygon(hole)
            if ring_poly.is_empty:
                continue
            shrunk = ring_poly.buffer(-float(eps), join_style=2)
            hole_parts.extend(_extract_polygons_from_geometry(shrunk))

        if hole_parts:
            try:
                holes_union = unary_union(hole_parts)
                repaired = shell_poly.difference(holes_union).buffer(0)
            except Exception:
                continue
        else:
            repaired = shell_poly.buffer(0)

        repaired_polys = _extract_polygons_from_geometry(repaired)
        if len(repaired_polys) != 1:
            continue

        candidate = repaired_polys[0]
        if candidate.is_empty:
            continue
        if _polygon_has_shell_hole_intersection(candidate):
            continue
        return candidate

    return None


def _prepare_polygon_candidates(poly: Polygon) -> list[Polygon]:
    """Prepare robust polygon candidates before triangulation.

    Repair flow:
    1) `buffer(0)` cleanup (already done by caller in most paths but harmless).
    2) `make_valid` expansion to polygon components.
    3) Split shell-hole touching cases with polygonize.

    Args:
        poly: Raw polygon geometry.

    Returns:
        Candidate polygon list for triangulation.
    """
    if poly is None or poly.is_empty:
        return []

    # Fast path: already valid and no shell-hole touching topology.
    if poly.geom_type == "Polygon" and poly.is_valid:
        if len(poly.interiors) == 0:
            return [poly]
        if not _polygon_has_shell_hole_intersection(poly):
            return [poly]

    fixed = poly.buffer(0)
    if fixed.is_empty:
        return []

    if _shapely_make_valid is not None:
        try:
            valid_geom = _shapely_make_valid(fixed)
        except Exception:
            valid_geom = fixed
    else:
        valid_geom = fixed

    polygons = _extract_polygons_from_geometry(valid_geom)
    out: list[Polygon] = []
    for pg in polygons:
        if pg is None or pg.is_empty:
            continue
        if len(pg.interiors) > 0 and _polygon_has_shell_hole_intersection(pg):
            shrunk = _shrink_touching_holes(pg)
            if shrunk is not None:
                out.append(shrunk)
                continue
        out.extend(_split_polygon_touching_holes(pg))

    # Deduplicate candidates to avoid redundant triangulation work.
    deduped: list[Polygon] = []
    seen_wkb: set[bytes] = set()
    for pg in out:
        if pg is None or pg.is_empty:
            continue
        key = bytes(pg.wkb)
        if key in seen_wkb:
            continue
        seen_wkb.add(key)
        deduped.append(pg)
    return deduped


def _triangle_triangulate_subprocess_entry(conn, tri_input: dict[str, Any], options: str) -> None:
    """Child-process entry for isolated triangle triangulation.

    Args:
        conn: Multiprocessing pipe endpoint.
        tri_input: Triangle-library input dictionary.
        options: Triangle options string.
    """
    try:
        result = tr.triangulate(tri_input, options)
        conn.send({"ok": True, "result": result})
    except Exception as exc:
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _triangle_triangulate_isolated(
    tri_input: dict[str, Any],
    options: str = "pq",
    timeout_sec: float = _TRIANGLE_SUBPROC_TIMEOUT_SEC,
) -> dict[str, Any] | None:
    """Run triangle triangulation in an isolated child process.

    This prevents native segfault from crashing the main worker process.

    Args:
        tri_input: Triangle-library input dictionary.
        options: Triangle options string.
        timeout_sec: Child-process timeout in seconds.

    Returns:
        Triangle output dict when successful, else None.
    """
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_triangle_triangulate_subprocess_entry,
        args=(send_conn, tri_input, options),
        daemon=True,
    )
    proc.start()
    send_conn.close()

    payload: dict[str, Any] | None = None
    try:
        if recv_conn.poll(timeout=max(0.1, float(timeout_sec))):
            payload = recv_conn.recv()
    except Exception:
        payload = None
    finally:
        try:
            recv_conn.close()
        except Exception:
            pass
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=1.0)

    if not payload or not bool(payload.get("ok", False)):
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    return result


def _triangulate_polygon_fallback(poly_norm: Polygon) -> np.ndarray:
    """Fallback triangulation using shapely triangulate + centroid clipping.

    Args:
        poly_norm: Normalized polygon geometry.

    Returns:
        Triangle array `[T,3,2]`.
    """
    try:
        tri_geoms = shapely_triangulate(poly_norm)
    except Exception:
        return np.zeros((0, 3, 2), dtype=np.float32)

    tris: list[np.ndarray] = []
    for tri_poly in tri_geoms:
        if tri_poly is None or tri_poly.is_empty or tri_poly.geom_type != "Polygon":
            continue
        coords = np.asarray(tri_poly.exterior.coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 4 or coords.shape[1] < 2:
            continue
        tri = coords[:3, :2]
        centroid = tri.mean(axis=0)
        if not poly_norm.covers(Point(float(centroid[0]), float(centroid[1]))):
            continue
        tris.append(tri.astype(np.float32))

    if not tris:
        return np.zeros((0, 3, 2), dtype=np.float32)
    return np.stack(tris, axis=0).astype(np.float32)


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

    # Keep fast path in-process for regular polygons.
    # Isolated subprocess is enabled only for risky shell-hole touching cases.
    has_holes = len(poly_norm.interiors) > 0
    need_isolated = bool(has_holes and _polygon_has_shell_hole_intersection(poly_norm))

    tri_data: dict[str, Any] | None
    if need_isolated:
        tri_data = _triangle_triangulate_isolated(tri_input, options="pq")
    else:
        try:
            tri_data = tr.triangulate(tri_input, "pq")
        except Exception:
            tri_data = None
            if has_holes:
                tri_data = _triangle_triangulate_isolated(tri_input, options="pq")

    if tri_data is None:
        return _triangulate_polygon_fallback(poly_norm)

    vertices = tri_data.get("vertices")
    tri_index = tri_data.get("triangles")
    if vertices is None or tri_index is None:
        return _triangulate_polygon_fallback(poly_norm)

    tris = np.asarray(vertices, dtype=np.float64)[np.asarray(tri_index, dtype=np.int64)]
    if tris.ndim != 3 or tris.shape[1:] != (3, 2):
        return _triangulate_polygon_fallback(poly_norm)

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


def _init_result_counters() -> dict[str, Any]:
    """Create a reusable row-level counter payload for triangulation statistics."""
    return {
        "triangles": [],
        "triangulated_output_count": 0,
        "total_rows": 0,
        "degenerate_row_count": 0,
        "dropped_row_count": 0,
        "failed_row_count": 0,
        "failed_sample_count": 0,
        "isolated_row_count": 0,
        "multipolygon_row_count": 0,
        "hole_row_count": 0,
        "triangulated_multipolygon_row_count": 0,
        "triangulated_hole_row_count": 0,
        "dropped_multipolygon_row_count": 0,
        "dropped_hole_row_count": 0,
        "chunk_failure_count": 0,
        "degenerate_records": [],
        "multipolygon_records": [],
        "hole_records": [],
        "failed_rows": [],
        "chunk_failures": [],
    }


def _merge_result_counters(target: dict[str, Any], partial: dict[str, Any]) -> None:
    """Merge one partial statistics payload into target payload in-place.

    Args:
        target: Mutable aggregate payload.
        partial: Partial payload from row/chunk computation.
    """
    target["triangles"].extend(list(partial.get("triangles", [])))
    target["triangulated_output_count"] += int(partial.get("triangulated_output_count", 0))
    target["total_rows"] += int(partial.get("total_rows", 0))
    target["degenerate_row_count"] += int(partial.get("degenerate_row_count", 0))
    target["dropped_row_count"] += int(partial.get("dropped_row_count", 0))
    target["failed_row_count"] += int(partial.get("failed_row_count", 0))
    target["failed_sample_count"] += int(partial.get("failed_sample_count", 0))
    target["isolated_row_count"] += int(partial.get("isolated_row_count", 0))
    target["multipolygon_row_count"] += int(partial.get("multipolygon_row_count", 0))
    target["hole_row_count"] += int(partial.get("hole_row_count", 0))
    target["triangulated_multipolygon_row_count"] += int(partial.get("triangulated_multipolygon_row_count", 0))
    target["triangulated_hole_row_count"] += int(partial.get("triangulated_hole_row_count", 0))
    target["dropped_multipolygon_row_count"] += int(partial.get("dropped_multipolygon_row_count", 0))
    target["dropped_hole_row_count"] += int(partial.get("dropped_hole_row_count", 0))
    target["chunk_failure_count"] += int(partial.get("chunk_failure_count", 0))
    target["degenerate_records"].extend(list(partial.get("degenerate_records", [])))
    target["multipolygon_records"].extend(list(partial.get("multipolygon_records", [])))
    target["hole_records"].extend(list(partial.get("hole_records", [])))
    target["failed_rows"].extend(list(partial.get("failed_rows", [])))
    target["chunk_failures"].extend(list(partial.get("chunk_failures", [])))


def _count_geometry_samples(geom) -> int:
    """Count row-level output samples contributed by one geometry row.

    Args:
        geom: Shapely geometry object.

    Returns:
        Number of row-level samples (`Polygon/MultiPolygon=1`, others=0).
    """
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type in {"Polygon", "MultiPolygon"}:
        return 1
    return 0


def _triangulate_row_geometry(
    row_idx: int,
    geom,
    file_path: str,
    layer_name: str | None,
    source_type: str,
    min_triangle_area: float,
    min_triangle_height: float,
    safe_mode: str,
    part_safe: int,
    node_safe: int,
    hole_safe: int,
    edge_safe: float,
    timeout_safe: float,
    norm_max: float,
    enable_log: bool,
) -> dict[str, Any]:
    """Triangulate one geometry row and collect row-level counters.

    Args:
        row_idx: Geometry row index in current task.
        geom: Geometry instance from GeoDataFrame.
        file_path: Source vector path.
        layer_name: Optional layer name for gdb task.
        source_type: Input source type token (`shp` / `geojs` / `gdb`).
        min_triangle_area: Minimum area threshold in normalized space.
        min_triangle_height: Minimum altitude threshold in normalized space.
        safe_mode: Safe-subprocess mode in {"all", "risky", "off"}.
        part_safe: Risk threshold for filtered part count.
        node_safe: Risk threshold for filtered part node count.
        hole_safe: Risk threshold for filtered part hole count.
        edge_safe: Risk threshold for filtered part minimum edge length.
        timeout_safe: Safe-subprocess timeout in seconds.
        norm_max: Target max absolute normalized coordinate.
        enable_log: Whether to collect detailed degenerate-sample records.

    Returns:
        Per-row result payload with counters and triangle sample list.
    """
    row_out = _init_result_counters()
    row_out["total_rows"] = 1

    if geom is None or geom.is_empty or geom.geom_type not in {"Polygon", "MultiPolygon"}:
        row_out["dropped_row_count"] = 1
        return row_out

    row_profile = _summarize_row_geometry(geom)
    is_multipolygon = bool(row_profile["is_multipolygon"])
    has_holes = bool(row_profile["has_holes"])
    row_out["multipolygon_row_count"] = int(is_multipolygon)
    row_out["hole_row_count"] = int(has_holes)

    def append_special_log(
        *,
        status: str,
        drop_reason: str,
        isolated: bool,
        filtered_part_count: int,
        filtered_triangle_count: int,
        kept_triangle_count: int,
        degenerated: bool,
    ) -> None:
        if not enable_log:
            return
        record = _build_special_row_log_record(
            file_path=file_path,
            layer_name=layer_name,
            source_type=source_type,
            row_idx=int(row_idx),
            profile=row_profile,
            safe_mode=str(safe_mode),
            isolated=bool(isolated),
            status=str(status),
            drop_reason=str(drop_reason),
            filtered_part_count=int(filtered_part_count),
            filtered_triangle_count=int(filtered_triangle_count),
            kept_triangle_count=int(kept_triangle_count),
            degenerated=bool(degenerated),
        )
        if is_multipolygon:
            row_out["multipolygon_records"].append(record)
        if has_holes:
            row_out["hole_records"].append(record)

    normalized_parts = _normalize_row_parts(geom=geom, norm_max=float(norm_max))
    if not normalized_parts:
        row_out["dropped_row_count"] = 1
        row_out["dropped_multipolygon_row_count"] = int(is_multipolygon)
        row_out["dropped_hole_row_count"] = int(has_holes)
        append_special_log(
            status="dropped",
            drop_reason="row_normalization_failed",
            isolated=False,
            filtered_part_count=0,
            filtered_triangle_count=0,
            kept_triangle_count=0,
            degenerated=False,
        )
        return row_out

    if str(safe_mode).strip().lower() == _SAFE_MODE_ALL:
        isolate_row = True
    else:
        filtered_parts, _ = _filter_row_parts(normalized_parts)
        if not filtered_parts:
            row_out["dropped_row_count"] = 1
            row_out["dropped_multipolygon_row_count"] = int(is_multipolygon)
            row_out["dropped_hole_row_count"] = int(has_holes)
            append_special_log(
                status="dropped",
                drop_reason="all_parts_filtered",
                isolated=False,
                filtered_part_count=int(len(normalized_parts)),
                filtered_triangle_count=0,
                kept_triangle_count=0,
                degenerated=False,
            )
            return row_out
        isolate_row = _should_isolate_row(
            safe_mode=safe_mode,
            filtered_parts=filtered_parts,
            part_safe=int(part_safe),
            node_safe=int(node_safe),
            hole_safe=int(hole_safe),
            edge_safe=float(edge_safe),
        )

    if isolate_row:
        row_out["isolated_row_count"] = 1
        row_result = _process_row_isolated(
            normalized_parts=normalized_parts,
            min_triangle_area=min_triangle_area,
            min_triangle_height=min_triangle_height,
            timeout_safe=float(timeout_safe),
        )
    else:
        row_result = _process_normalized_row_parts(
            normalized_parts=normalized_parts,
            min_triangle_area=min_triangle_area,
            min_triangle_height=min_triangle_height,
        )

    if not bool(row_result.get("ok", False)):
        row_out["dropped_row_count"] = 1
        row_out["dropped_multipolygon_row_count"] = int(is_multipolygon)
        row_out["dropped_hole_row_count"] = int(has_holes)
        append_special_log(
            status="dropped",
            drop_reason=str(row_result.get("failure_reason", "unknown_failure")),
            isolated=bool(isolate_row),
            filtered_part_count=int(row_result.get("filtered_part_count", 0)),
            filtered_triangle_count=int(row_result.get("filtered_triangle_count", 0)),
            kept_triangle_count=0,
            degenerated=False,
        )
        return row_out

    triangles = row_result.get("triangles")
    if not isinstance(triangles, np.ndarray) or triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        row_out["dropped_row_count"] = 1
        row_out["dropped_multipolygon_row_count"] = int(is_multipolygon)
        row_out["dropped_hole_row_count"] = int(has_holes)
        append_special_log(
            status="dropped",
            drop_reason="invalid_triangle_array",
            isolated=bool(isolate_row),
            filtered_part_count=int(row_result.get("filtered_part_count", 0)),
            filtered_triangle_count=int(row_result.get("filtered_triangle_count", 0)),
            kept_triangle_count=0,
            degenerated=False,
        )
        return row_out

    filtered_triangle_count = int(row_result.get("filtered_triangle_count", 0))
    had_part_filter = bool(row_result.get("had_part_filter", False))
    is_degenerated_row = bool(had_part_filter or filtered_triangle_count > 0)

    row_out["triangles"].append(triangles.astype(np.float32))
    row_out["triangulated_output_count"] = 1
    row_out["triangulated_multipolygon_row_count"] = int(is_multipolygon)
    row_out["triangulated_hole_row_count"] = int(has_holes)
    if is_degenerated_row:
        row_out["degenerate_row_count"] = 1
        if enable_log:
            row_out["degenerate_records"].append(
                {
                    "file_path": str(file_path),
                    "layer_name": layer_name,
                    "source_type": source_type,
                    "row_idx": int(row_idx),
                    "geom_type": row_profile.get("geom_type"),
                    "is_multipolygon": bool(is_multipolygon),
                    "raw_part_count": int(row_profile.get("raw_part_count", 0)),
                    "has_holes": bool(has_holes),
                    "parts_with_holes": int(row_profile.get("parts_with_holes", 0)),
                    "total_hole_count": int(row_profile.get("total_hole_count", 0)),
                    "max_part_hole_count": int(row_profile.get("max_part_hole_count", 0)),
                    "safe_mode": str(safe_mode),
                    "isolated": bool(isolate_row),
                    "filtered_part_count": int(row_result.get("filtered_part_count", 0)),
                    "filtered_triangle_count": int(filtered_triangle_count),
                    "kept_triangle_count": int(triangles.shape[0]),
                }
            )
    append_special_log(
        status="triangulated",
        drop_reason="",
        isolated=bool(isolate_row),
        filtered_part_count=int(row_result.get("filtered_part_count", 0)),
        filtered_triangle_count=int(filtered_triangle_count),
        kept_triangle_count=int(triangles.shape[0]),
        degenerated=bool(is_degenerated_row),
    )

    return row_out


def _triangulate_chunk_worker(
    chunk_index: int,
    row_payloads: list[tuple[int, Any]],
    file_path: str,
    layer_name: str | None,
    source_type: str,
    min_triangle_area: float,
    min_triangle_height: float,
    safe_mode: str,
    part_safe: int,
    node_safe: int,
    hole_safe: int,
    edge_safe: float,
    timeout_safe: float,
    norm_max: float,
    enable_log: bool,
) -> dict[str, Any]:
    """Worker function for one row-chunk in task-internal parallelization.

    Args:
        chunk_index: Stable chunk index in current task.
        row_payloads: List of `(row_idx, geometry)` tuples.
        file_path: Source vector path.
        layer_name: Optional layer name for gdb task.
        source_type: Input source type token (`shp` / `geojs` / `gdb`).
        min_triangle_area: Minimum area threshold in normalized space.
        min_triangle_height: Minimum altitude threshold in normalized space.
        safe_mode: Safe-subprocess mode in {"all", "risky", "off"}.
        part_safe: Risk threshold for filtered part count.
        node_safe: Risk threshold for filtered part node count.
        hole_safe: Risk threshold for filtered part hole count.
        edge_safe: Risk threshold for filtered part minimum edge length.
        timeout_safe: Safe-subprocess timeout in seconds.
        norm_max: Target max absolute normalized coordinate.
        enable_log: Whether to collect detailed degenerate-sample records.

    Returns:
        Chunk-level result payload.
    """
    chunk_out = _init_result_counters()
    row_sample_count = int(sum(_count_geometry_samples(payload[1]) for payload in row_payloads))
    for row_idx, geom in row_payloads:
        try:
            row_out = _triangulate_row_geometry(
                row_idx=int(row_idx),
                geom=geom,
                file_path=file_path,
                layer_name=layer_name,
                source_type=source_type,
                min_triangle_area=min_triangle_area,
                min_triangle_height=min_triangle_height,
                safe_mode=safe_mode,
                part_safe=int(part_safe),
                node_safe=int(node_safe),
                hole_safe=int(hole_safe),
                edge_safe=float(edge_safe),
                timeout_safe=float(timeout_safe),
                norm_max=float(norm_max),
                enable_log=enable_log,
            )
        except Exception as exc:
            row_profile = _safe_row_profile_for_failure(geom)
            sample_count = int(_count_geometry_samples(geom))
            is_multipolygon = bool(row_profile.get("is_multipolygon", False))
            has_holes = bool(row_profile.get("has_holes", False))
            row_out = _init_result_counters()
            row_out["total_rows"] = 1
            row_out["dropped_row_count"] = 1
            row_out["failed_row_count"] = 1
            row_out["failed_sample_count"] = int(sample_count)
            row_out["multipolygon_row_count"] = int(is_multipolygon)
            row_out["hole_row_count"] = int(has_holes)
            row_out["dropped_multipolygon_row_count"] = int(is_multipolygon)
            row_out["dropped_hole_row_count"] = int(has_holes)
            row_out["failed_rows"].append(
                _build_failed_row_record(
                    file_path=file_path,
                    layer_name=layer_name,
                    source_type=source_type,
                    row_idx=int(row_idx),
                    profile=row_profile,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    sample_count=int(sample_count),
                )
            )
        _merge_result_counters(chunk_out, row_out)

    return {
        "index": int(chunk_index),
        "ok": True,
        "error": "",
        "row_count": int(len(row_payloads)),
        "row_sample_count": int(row_sample_count),
        **chunk_out,
    }


def _build_chunk_failure_result(
    *,
    chunk_index: int,
    row_count: int,
    row_sample_count: int,
    file_path: str,
    layer_name: str | None,
    source_type: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    """Build one synthetic chunk result when chunk execution itself fails."""
    failure_out = _init_result_counters()
    failure_out["total_rows"] = int(row_count)
    failure_out["dropped_row_count"] = int(row_count)
    failure_out["chunk_failure_count"] = 1
    failure_out["chunk_failures"].append(
        _build_chunk_failure_record(
            chunk_index=int(chunk_index),
            row_count=int(row_count),
            row_sample_count=int(row_sample_count),
            file_path=file_path,
            layer_name=layer_name,
            source_type=source_type,
            error_type=error_type,
            error_message=error_message,
        )
    )
    return {
        "index": int(chunk_index),
        "ok": False,
        "error": f"{error_type}: {error_message}",
        "row_count": int(row_count),
        "row_sample_count": int(row_sample_count),
        **failure_out,
    }


def _triangulate_task_worker(
    task_index: int,
    task_count: int,
    file_path: str,
    layer_name: str | None,
    source_type: str,
    num_workers: int,
    rows_per_chunk: int,
    progress_every_chunks: int,
    min_triangle_area: float,
    min_triangle_height: float,
    safe_mode: str,
    part_safe: int,
    node_safe: int,
    hole_safe: int,
    edge_safe: float,
    timeout_safe: float,
    norm_max: float,
    enable_log: bool,
    writer: "_ShardWriter | None" = None,
) -> dict[str, Any]:
    """Triangulate one input task with task-serial and chunk-internal parallelism.

    Args:
        task_index: 0-based index of current task.
        task_count: Total number of tasks.
        file_path: Source vector path.
        layer_name: Optional layer name for gdb task.
        source_type: Input source type token (`shp` / `geojs` / `gdb`).
        num_workers: Requested worker count for intra-task chunk parallelism.
        rows_per_chunk: Row count per chunk.
        progress_every_chunks: Print summary every N merged chunks (`<=0` disables).
        min_triangle_area: Minimum area threshold in normalized space.
        min_triangle_height: Minimum altitude threshold in normalized space.
        safe_mode: Safe-subprocess mode in {"all", "risky", "off"}.
        part_safe: Risk threshold for filtered part count.
        node_safe: Risk threshold for filtered part node count.
        hole_safe: Risk threshold for filtered part hole count.
        edge_safe: Risk threshold for filtered part minimum edge length.
        timeout_safe: Safe-subprocess timeout in seconds.
        norm_max: Target max absolute normalized coordinate.
        enable_log: Whether to collect detailed degenerate-sample records.
        writer: Optional shard writer. When provided, chunk triangles are
            streamed into the writer during ordered merge instead of being kept
            in task-level memory until the whole source file finishes.

    Returns:
        Task-level triangulation result payload (same schema as legacy file worker).
    """
    try:
        if layer_name is None:
            gdf = gpd.read_file(file_path)
        else:
            gdf = gpd.read_file(file_path, layer=layer_name)
    except Exception as exc:
        return {
            "index": int(task_index),
            "file_path": file_path,
            "layer_name": layer_name,
            "source_type": source_type,
            "ok": False,
            "error": str(exc),
            "source_geometry_count": 0,
            **_init_result_counters(),
        }

    source_geometry_count = int(len(gdf.geometry))
    row_payloads: list[tuple[int, Any]] = []
    for row_idx, geom in enumerate(gdf.geometry):
        row_payloads.append((int(row_idx), geom))

    chunk_size = max(1, int(rows_per_chunk))
    chunks = [row_payloads[i : i + chunk_size] for i in range(0, len(row_payloads), chunk_size)]
    chunk_worker_count = _resolve_intra_workers(num_workers=num_workers, unit_count=len(chunks))

    layer_suffix = f" (layer={layer_name})" if layer_name is not None else ""
    print(f"[INFO] Task {task_index + 1}/{task_count}: {file_path}{layer_suffix}")
    print(
        f"[INFO]   Task rows/chunks/workers : {source_geometry_count}/{len(chunks)}/{chunk_worker_count}"
    )

    task_out = _init_result_counters()
    task_error_count = 0
    task_output_sample_count = 0

    if not chunks:
        return {
            "index": int(task_index),
            "file_path": file_path,
            "layer_name": layer_name,
            "source_type": source_type,
            "ok": True,
            "error": "",
            "source_geometry_count": source_geometry_count,
            **task_out,
        }

    merged_chunks = 0
    with tqdm(total=source_geometry_count, desc=f"Task {task_index + 1}/{task_count} rows", unit="row", leave=False) as row_pbar:
        if chunk_worker_count <= 1:
            for chunk_index, chunk_payload in enumerate(chunks):
                chunk_row_count = int(len(chunk_payload))
                chunk_row_sample_count = int(sum(_count_geometry_samples(p[1]) for p in chunk_payload))
                try:
                    chunk_result = _triangulate_chunk_worker(
                        chunk_index=chunk_index,
                        row_payloads=chunk_payload,
                        file_path=file_path,
                        layer_name=layer_name,
                        source_type=source_type,
                        min_triangle_area=min_triangle_area,
                        min_triangle_height=min_triangle_height,
                        safe_mode=safe_mode,
                        part_safe=int(part_safe),
                        node_safe=int(node_safe),
                        hole_safe=int(hole_safe),
                        edge_safe=float(edge_safe),
                        timeout_safe=float(timeout_safe),
                        norm_max=float(norm_max),
                        enable_log=enable_log,
                    )
                except Exception as exc:
                    chunk_result = _build_chunk_failure_result(
                        chunk_index=chunk_index,
                        row_count=chunk_row_count,
                        row_sample_count=chunk_row_sample_count,
                        file_path=file_path,
                        layer_name=layer_name,
                        source_type=source_type,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                row_pbar.update(int(chunk_result.get("row_count", len(chunk_payload))))

                if not bool(chunk_result.get("ok", False)):
                    task_error_count += 1
                    tqdm.write(
                        f"[WARN] Chunk failed in {file_path}{layer_suffix}: {chunk_result.get('error', 'unknown error')}"
                    )
                    _merge_result_counters(task_out, chunk_result)
                else:
                    chunk_triangles = list(chunk_result.get("triangles", []))
                    if writer is not None and chunk_triangles:
                        writer.add_many(chunk_triangles)
                        chunk_result = dict(chunk_result)
                        chunk_result["triangles"] = []
                    task_output_sample_count += int(len(chunk_triangles))
                    _merge_result_counters(task_out, chunk_result)

                merged_chunks += 1
                if progress_every_chunks > 0 and merged_chunks % int(progress_every_chunks) == 0:
                    tqdm.write(
                        "[INFO]   Chunk merge progress: "
                        f"{merged_chunks}/{len(chunks)}, "
                        f"triangulated={task_output_sample_count}, "
                        f"dropped={task_out['dropped_row_count']}, "
                        f"degenerated={task_out['degenerate_row_count']}"
                    )
        else:
            mp_context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=chunk_worker_count, mp_context=mp_context) as executor:
                future_to_meta = {
                    executor.submit(
                        _triangulate_chunk_worker,
                        chunk_index,
                        chunk_payload,
                        file_path,
                        layer_name,
                        source_type,
                        float(min_triangle_area),
                        float(min_triangle_height),
                        str(safe_mode),
                        int(part_safe),
                        int(node_safe),
                        int(hole_safe),
                        float(edge_safe),
                        float(timeout_safe),
                        float(norm_max),
                        bool(enable_log),
                    ): (chunk_index, int(len(chunk_payload)), int(sum(_count_geometry_samples(p[1]) for p in chunk_payload)))
                    for chunk_index, chunk_payload in enumerate(chunks)
                }

                pending_results: dict[int, dict[str, Any]] = {}
                next_chunk_index = 0

                for future in as_completed(future_to_meta):
                    chunk_index, row_count, row_sample_count = future_to_meta[future]
                    try:
                        chunk_result = future.result()
                    except Exception as exc:
                        chunk_result = _build_chunk_failure_result(
                            chunk_index=chunk_index,
                            row_count=row_count,
                            row_sample_count=row_sample_count,
                            file_path=file_path,
                            layer_name=layer_name,
                            source_type=source_type,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )

                    row_pbar.update(int(chunk_result.get("row_count", row_count)))
                    pending_results[int(chunk_result["index"])] = chunk_result

                    while next_chunk_index in pending_results:
                        ordered_chunk = pending_results.pop(next_chunk_index)
                        if not bool(ordered_chunk.get("ok", False)):
                            task_error_count += 1
                            tqdm.write(
                                f"[WARN] Chunk failed in {file_path}{layer_suffix}: "
                                f"{ordered_chunk.get('error', 'unknown error')}"
                            )
                            _merge_result_counters(task_out, ordered_chunk)
                        else:
                            chunk_triangles = list(ordered_chunk.get("triangles", []))
                            if writer is not None and chunk_triangles:
                                writer.add_many(chunk_triangles)
                                ordered_chunk = dict(ordered_chunk)
                                ordered_chunk["triangles"] = []
                            task_output_sample_count += int(len(chunk_triangles))
                            _merge_result_counters(task_out, ordered_chunk)

                        merged_chunks += 1
                        if progress_every_chunks > 0 and merged_chunks % int(progress_every_chunks) == 0:
                            tqdm.write(
                                "[INFO]   Chunk merge progress: "
                                f"{merged_chunks}/{len(chunks)}, "
                                f"triangulated={task_output_sample_count}, "
                                f"dropped={task_out['dropped_row_count']}, "
                                f"degenerated={task_out['degenerate_row_count']}"
                            )
                        next_chunk_index += 1

    if task_error_count > 0:
        print(f"[WARN] Task {task_index + 1}/{task_count} chunk failures: {task_error_count}")

    return {
        "index": int(task_index),
        "file_path": file_path,
        "layer_name": layer_name,
        "source_type": source_type,
        "ok": True,
        "error": "",
        "source_geometry_count": source_geometry_count,
        **task_out,
    }


def _estimate_triangle_sample_bytes(tris: np.ndarray) -> int:
    """Estimate serialized byte contribution for one triangle sample.

    Args:
        tris: Triangle array `[T,3,2]`.

    Returns:
        Estimated serialized bytes for torch-save oriented sharding.
    """
    tri_np = np.asarray(tris, dtype=np.float32)
    # Heuristic calibrated for list[np.ndarray] serialized by torch.save.
    # It keeps shard sizes close to target without per-sample torch serialization overhead.
    return int(tri_np.nbytes) + 640


def _build_shard_path(base_output_path: Path, part_index: int) -> Path:
    """Build shard output path from base path and part index.

    Args:
        base_output_path: User-provided output base path.
        part_index: 1-based shard index.

    Returns:
        Shard file path.
    """
    suffix = base_output_path.suffix if base_output_path.suffix else ".pt"
    return base_output_path.with_name(f"{base_output_path.stem}_part_{part_index:04d}{suffix}")


def _default_log_path(output_path: Path) -> Path:
    """Build default triangulation-log path beside output `.pt` base file.

    Args:
        output_path: Output `.pt` base path.

    Returns:
        JSON log path under the same directory.
    """
    return output_path.with_name(f"{output_path.stem}.triangulation_log.json")


def _default_row_failures_path(output_path: Path) -> Path:
    """Build default row-failure summary path beside output `.pt` base file."""
    return output_path.with_name(f"{output_path.stem}.row_failures.json")


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

        written_path = save_triangle_shard(output_file, self._buffer)

        self.shard_paths.append(written_path)
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
            "serialization": TORCH_SHARD_SERIALIZATION,
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


def _resolve_intra_workers(num_workers: int, unit_count: int) -> int:
    """Resolve effective process count for task-internal parallelism.

    Args:
        num_workers: Requested worker count. `<=0` means auto.
        unit_count: Number of executable units (for example, chunk count).

    Returns:
        Effective worker count in `[1, unit_count]` when `unit_count>0`.
    """
    if int(num_workers) <= 0:
        cpu_count = os.cpu_count() or 1
        resolved = max(1, cpu_count - 1)
    else:
        resolved = max(1, int(num_workers))

    if unit_count <= 0:
        return resolved
    return max(1, min(resolved, int(unit_count)))


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
        "triangulated_count": int(result.get("triangulated_output_count", len(result.get("triangles", [])))),
        "total_rows": int(result.get("total_rows", 0)),
        "degenerate_row_count": int(result.get("degenerate_row_count", 0)),
        "dropped_row_count": int(result.get("dropped_row_count", 0)),
        "failed_row_count": int(result.get("failed_row_count", 0)),
        "failed_sample_count": int(result.get("failed_sample_count", 0)),
        "isolated_row_count": int(result.get("isolated_row_count", 0)),
        "multipolygon_row_count": int(result.get("multipolygon_row_count", 0)),
        "hole_row_count": int(result.get("hole_row_count", 0)),
        "triangulated_multipolygon_row_count": int(result.get("triangulated_multipolygon_row_count", 0)),
        "triangulated_hole_row_count": int(result.get("triangulated_hole_row_count", 0)),
        "dropped_multipolygon_row_count": int(result.get("dropped_multipolygon_row_count", 0)),
        "dropped_hole_row_count": int(result.get("dropped_hole_row_count", 0)),
        "chunk_failure_count": int(result.get("chunk_failure_count", 0)),
        "degenerate_records": list(result.get("degenerate_records", [])),
        "multipolygon_records": list(result.get("multipolygon_records", [])),
        "hole_records": list(result.get("hole_records", [])),
        "failed_rows": list(result.get("failed_rows", [])),
        "chunk_failures": list(result.get("chunk_failures", [])),
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
    writer.add_many(triangles)
    return stats


def process_and_save(
    input_dirs: Iterable[str | Path],
    output_path: str | Path,
    file_type: str = "shp",
    layer: str = "all",
    num_workers: int = 0,
    rows_per_chunk: int = 2000,
    progress_every_chunks: int = 10,
    shard_size_mb: float = 0.0,
    min_triangle_area: float = 1e-8,
    min_triangle_height: float = 1e-5,
    safe_mode: str = _SAFE_MODE_RISKY,
    part_safe: int = 1,
    node_safe: int = 2048,
    hole_safe: int = 1,
    edge_safe: float = 1e-5,
    timeout_safe: float = _TRIANGLE_SUBPROC_TIMEOUT_SEC,
    norm_max: float = 1.0,
    log: bool = False,
) -> None:
    """Build triangulated polygon dataset and save to disk.

    Args:
        input_dirs: Iterable of source directories containing vector files.
        output_path: Output `.pt` base path.
        file_type: Input vector source type (`shp`, `gdb`, `geojs`).
        layer: Layer selector when `file_type='gdb'`. Use `'all'` for all layers.
        num_workers: Task-internal process count. `<=0` means auto.
        rows_per_chunk: Number of source rows in one intra-task chunk.
        progress_every_chunks: Print summary every N merged chunks (`<=0` disables).
        shard_size_mb: Target shard size in MB. `<=0` means single `.pt` output.
        min_triangle_area: Minimum triangle area threshold in normalized space.
        min_triangle_height: Minimum altitude threshold in normalized space.
        safe_mode: Row isolation trigger mode in {"all", "risky", "off"}.
        part_safe: Isolation threshold for filtered part count.
        node_safe: Isolation threshold for filtered part node count.
        hole_safe: Isolation threshold for filtered part hole count.
        edge_safe: Isolation threshold for filtered part minimum edge length.
        timeout_safe: Row-isolation subprocess timeout in seconds.
        norm_max: Maximum absolute normalized coordinate.
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

    suggested_intra_workers = _resolve_intra_workers(num_workers=num_workers, unit_count=0)
    rows_per_chunk = max(1, int(rows_per_chunk))
    shard_size_mb = float(shard_size_mb)
    progress_every_chunks = int(progress_every_chunks)
    safe_mode = str(safe_mode).strip().lower()
    if safe_mode not in _SAFE_MODE_CHOICES:
        raise ValueError(f"Unsupported safe_mode: {safe_mode}. Use one of: {', '.join(_SAFE_MODE_CHOICES)}")
    part_safe = int(part_safe)
    node_safe = int(node_safe)
    hole_safe = int(hole_safe)
    edge_safe = float(edge_safe)
    timeout_safe = float(timeout_safe)
    norm_max = float(norm_max)
    if norm_max <= 0.0:
        raise ValueError("--norm_max must be > 0.")
    if timeout_safe <= 0.0:
        raise ValueError("--timeout_safe must be > 0.")

    print(f"[INFO] Input file_type         : {canonical_file_type}")
    if canonical_file_type == "gdb":
        print(f"[INFO] Input layer selector    : {layer}")
    print(f"[INFO] Discovered input tasks : {len(task_list)}")
    print("[INFO] Parallel strategy      : task-serial + intra-task parallel")
    print(f"[INFO] Intra-task workers    : {suggested_intra_workers}")
    print(f"[INFO] Rows per chunk         : {rows_per_chunk}")
    if progress_every_chunks > 0:
        print(f"[INFO] Chunk summary interval : every {progress_every_chunks} chunks")
    else:
        print("[INFO] Chunk summary interval : disabled")
    print(f"[INFO] Degenerate filter: min_triangle_area={min_triangle_area:.3e}, min_triangle_height={min_triangle_height:.3e}")
    print(
        "[INFO] Safe mode / thresholds : "
        f"{safe_mode} | part>{part_safe}, node>{node_safe}, hole>{hole_safe}, edge<{edge_safe:.3e}, timeout={timeout_safe:.1f}s"
    )
    print(f"[INFO] Row normalization max  : {norm_max:.3f}")
    if shard_size_mb > 0:
        print(f"[INFO] Sharding enabled: target {shard_size_mb:.2f} MB per output .pt")
    else:
        print("[WARN] Sharding disabled: single output .pt keeps all triangulated samples buffered until finalize().")

    writer = _ShardWriter(output_path=output_path, shard_size_mb=shard_size_mb)

    files_ok = 0
    files_failed = 0
    source_geometries_total = 0
    triangulated_total = 0
    total_rows = 0
    degenerated_rows = 0
    dropped_rows = 0
    failed_rows = 0
    failed_samples = 0
    isolated_rows = 0
    degenerate_log_records: list[dict[str, Any]] = []
    multipolygon_rows = 0
    hole_rows = 0
    triangulated_multipolygon_rows = 0
    triangulated_hole_rows = 0
    dropped_multipolygon_rows = 0
    dropped_hole_rows = 0
    chunk_failures = 0
    multipolygon_log_records: list[dict[str, Any]] = []
    hole_log_records: list[dict[str, Any]] = []
    failed_row_records: list[dict[str, Any]] = []
    chunk_failure_records: list[dict[str, Any]] = []

    def consume_and_merge(result: dict[str, Any]) -> None:
        """Merge one ordered worker result into global counters and writer.

        Args:
            result: Worker output dictionary.
        """
        nonlocal files_ok, files_failed
        nonlocal source_geometries_total, triangulated_total
        nonlocal total_rows, degenerated_rows, dropped_rows, failed_rows, failed_samples, isolated_rows
        nonlocal multipolygon_rows, hole_rows
        nonlocal triangulated_multipolygon_rows, triangulated_hole_rows
        nonlocal dropped_multipolygon_rows, dropped_hole_rows
        nonlocal chunk_failures

        merged = _consume_worker_result(result, writer)

        files_ok += int(merged["ok"])
        files_failed += int(not merged["ok"])

        source_geometries_total += int(merged["source_geometry_count"])
        triangulated_total += int(merged["triangulated_count"])
        total_rows += int(merged["total_rows"])
        degenerated_rows += int(merged["degenerate_row_count"])
        dropped_rows += int(merged["dropped_row_count"])
        failed_rows += int(merged["failed_row_count"])
        failed_samples += int(merged["failed_sample_count"])
        isolated_rows += int(merged["isolated_row_count"])
        multipolygon_rows += int(merged["multipolygon_row_count"])
        hole_rows += int(merged["hole_row_count"])
        triangulated_multipolygon_rows += int(merged["triangulated_multipolygon_row_count"])
        triangulated_hole_rows += int(merged["triangulated_hole_row_count"])
        dropped_multipolygon_rows += int(merged["dropped_multipolygon_row_count"])
        dropped_hole_rows += int(merged["dropped_hole_row_count"])
        chunk_failures += int(merged["chunk_failure_count"])

        failed_row_records.extend(merged["failed_rows"])
        chunk_failure_records.extend(merged["chunk_failures"])

        if log:
            degenerate_log_records.extend(merged["degenerate_records"])
            multipolygon_log_records.extend(merged["multipolygon_records"])
            hole_log_records.extend(merged["hole_records"])

    with tqdm(total=len(task_list), desc="Triangulating tasks", unit="task") as pbar:
        for task_index, task in enumerate(task_list):
            result = _triangulate_task_worker(
                task_index=task_index,
                task_count=len(task_list),
                file_path=str(task["path"]),
                layer_name=task.get("layer"),
                source_type=str(task.get("source_type", canonical_file_type)),
                num_workers=int(num_workers),
                rows_per_chunk=int(rows_per_chunk),
                progress_every_chunks=int(progress_every_chunks),
                min_triangle_area=float(min_triangle_area),
                min_triangle_height=float(min_triangle_height),
                safe_mode=str(safe_mode),
                part_safe=int(part_safe),
                node_safe=int(node_safe),
                hole_safe=int(hole_safe),
                edge_safe=float(edge_safe),
                timeout_safe=float(timeout_safe),
                norm_max=float(norm_max),
                enable_log=bool(log),
                writer=writer,
            )
            consume_and_merge(result)
            pbar.update(1)

    if triangulated_total == 0:
        print("[WARN] No valid polygons were triangulated. Nothing to save.")
        shard_paths: list[Path] = []
        manifest_path: Path | None = None
    else:
        shard_paths, manifest_path = writer.finalize()
        if not shard_paths:
            print("[WARN] No output shard was written.")

    if failed_rows > 0 or chunk_failures > 0:
        row_failures_path = _default_row_failures_path(output_path)
        row_failures_payload = {
            "output_base_path": str(output_path),
            "input_file_type": canonical_file_type,
            "input_layer_selector": str(layer),
            "safe_mode": str(safe_mode),
            "failed_row_count": int(failed_rows),
            "failed_sample_count": int(failed_samples),
            "chunk_failure_count": int(chunk_failures),
            "failed_rows": failed_row_records,
            "chunk_failures": chunk_failure_records,
        }
        with row_failures_path.open("w", encoding="utf-8") as fp:
            json.dump(row_failures_payload, fp, ensure_ascii=False, indent=2)
        print(f"[INFO] Row failure summary    : {row_failures_path}")

    if log:
        log_path = _default_log_path(output_path)
        log_payload = {
            "output_base_path": str(output_path),
            "input_file_type": canonical_file_type,
            "input_layer_selector": str(layer),
            "safe_mode": str(safe_mode),
            "part_safe": int(part_safe),
            "node_safe": int(node_safe),
            "hole_safe": int(hole_safe),
            "edge_safe": float(edge_safe),
            "timeout_safe": float(timeout_safe),
            "norm_max": float(norm_max),
            "min_triangle_area": float(min_triangle_area),
            "min_triangle_height": float(min_triangle_height),
            "total_rows": int(total_rows),
            "triangulated_rows": int(triangulated_total),
            "dropped_rows": int(dropped_rows),
            "failed_rows": int(failed_rows),
            "failed_sample_count": int(failed_samples),
            "chunk_failures": int(chunk_failures),
            "degenerated_rows": int(degenerated_rows),
            "isolated_rows": int(isolated_rows),
            "multipolygon_rows": int(multipolygon_rows),
            "triangulated_multipolygon_rows": int(triangulated_multipolygon_rows),
            "dropped_multipolygon_rows": int(dropped_multipolygon_rows),
            "hole_rows": int(hole_rows),
            "triangulated_hole_rows": int(triangulated_hole_rows),
            "dropped_hole_rows": int(dropped_hole_rows),
            "degenerated_row_records": degenerate_log_records,
            "multipolygon_row_records": multipolygon_log_records,
            "hole_row_records": hole_log_records,
        }
        with log_path.open("w", encoding="utf-8") as fp:
            json.dump(log_payload, fp, ensure_ascii=False, indent=2)
        print(f"[INFO] Triangulation log      : {log_path}")

    print("[INFO] Triangulation completed.")
    print(f"[INFO] Tasks succeeded/failed : {files_ok}/{files_failed}")
    print(f"[INFO] Source geometries       : {source_geometries_total}")
    print(f"[INFO] Total rows              : {total_rows}")
    print(f"[INFO] Triangulated rows       : {triangulated_total}")
    print(f"[INFO] Dropped rows           : {dropped_rows}")
    print(f"[INFO] Failed rows            : {failed_rows}")
    print(f"[INFO] Chunk failures         : {chunk_failures}")
    print(f"[INFO] Degenerated rows       : {degenerated_rows}")
    print(f"[INFO] Isolated rows          : {isolated_rows}")
    print(
        "[INFO] MultiPolygon rows      : "
        f"total={multipolygon_rows}, triangulated={triangulated_multipolygon_rows}, dropped={dropped_multipolygon_rows}"
    )
    print(
        "[INFO] Hole rows              : "
        f"total={hole_rows}, triangulated={triangulated_hole_rows}, dropped={dropped_hole_rows}"
    )

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
