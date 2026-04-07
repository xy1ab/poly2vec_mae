"""Visualize one source row under the current row-level triangulation flow.

The script is row-centric and mirrors the current build semantics:
- one source row -> one merged training sample
- row-level normalization
- strict part filtering (`not is_valid` or `shell-hole-touching`)
- per-part triangulation merged back into one row sample

It depends only on `src/` helpers and does not import any other script.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygonPatch
from shapely import wkb as shapely_wkb
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


_BUILD_MODULE = None
_DEFAULT_MIN_TRIANGLE_AREA = 1e-8
_DEFAULT_MIN_TRIANGLE_HEIGHT = 1e-5
_DEFAULT_TIMEOUT_SAFE = 20.0
_PLOT_COLORS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
]
_GEOM_PANEL_PAD_RATIO = 0.03
_NORM_PANEL_PAD_RATIO = 1.03


def _compactify_axis(ax) -> None:
    """Reduce non-data whitespace inside one subplot."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#9ca3af")


def _inject_repo_root() -> Path:
    """Inject repository root into `sys.path` for direct script execution."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def _load_build_module():
    """Load build_dataset_triangle lazily from `src/` only."""
    global _BUILD_MODULE
    if _BUILD_MODULE is None:
        _BUILD_MODULE = importlib.import_module("ae_pretrain.src.datasets.build_dataset_triangle")
    return _BUILD_MODULE


def _resolve_single_shp(input_dir: Path) -> Path:
    """Resolve exactly one `.shp` file from the given directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"--input_dir must be a directory that contains one .shp file: {input_dir}")

    shp_files = sorted(path for path in input_dir.glob("*.shp") if path.is_file())
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found under input directory: {input_dir}")
    if len(shp_files) > 1:
        shp_list = ", ".join(path.name for path in shp_files[:10])
        raise ValueError(
            f"Expected exactly one .shp file under {input_dir}, but found {len(shp_files)}: {shp_list}"
        )
    return shp_files[0]


def _iter_polygons(geom) -> list[Polygon]:
    """Flatten a generic geometry into polygon parts."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if poly is not None and not poly.is_empty]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for sub in geom.geoms:
            out.extend(_iter_polygons(sub))
        return out
    return []


def _classify_polygon_type(geom) -> str:
    """Classify one source geometry into row-level connectivity categories."""
    if geom is None or getattr(geom, "is_empty", True):
        return "complex"
    geom_type = str(getattr(geom, "geom_type", ""))
    if geom_type == "Polygon":
        hole_count = int(len(getattr(geom, "interiors", [])))
        if hole_count == 0:
            return "simple"
        if hole_count == 1:
            return "donut"
        return "porous"
    if geom_type == "MultiPolygon":
        parts = list(getattr(geom, "geoms", []))
        if not parts:
            return "complex"
        any_holes = any(
            int(len(getattr(poly, "interiors", []))) > 0
            for poly in parts
            if poly is not None and not poly.is_empty
        )
        return "complex" if any_holes else "multi"
    return "complex"


def _resolve_row_part_target(build_module, gdf: gpd.GeoDataFrame, row_index: int, part_index: int) -> dict[str, Any]:
    """Resolve one source row plus one selected raw polygon part."""
    if row_index < 0 or row_index >= len(gdf):
        raise IndexError(f"row_index out of range: {row_index} (total_rows={len(gdf)})")

    row_geom = gdf.geometry.iloc[row_index]
    expanded_parts = build_module._expand_geometry_to_polygons(row_geom)
    if not expanded_parts:
        raise ValueError(f"row_index={row_index} does not contain a Polygon or MultiPolygon geometry")

    requested_part_index = max(1, int(part_index))
    resolved_part_index = min(requested_part_index, len(expanded_parts))
    selected_poly, from_multi = expanded_parts[resolved_part_index - 1]

    return {
        "row_index": int(row_index),
        "part_index": int(resolved_part_index),
        "requested_part_index": int(requested_part_index),
        "part_count": int(len(expanded_parts)),
        "row_geom": row_geom,
        "expanded_parts": [poly for poly, _ in expanded_parts],
        "selected_poly": selected_poly,
        "from_multipolygon": bool(from_multi),
    }


def _compute_row_frame(geom) -> tuple[np.ndarray | None, float]:
    """Compute the row-level normalization frame used by the build pipeline."""
    if geom is None or geom.is_empty:
        return None, 0.0
    minx, miny, maxx, maxy = geom.bounds
    cx = (float(minx) + float(maxx)) * 0.5
    cy = (float(miny) + float(maxy)) * 0.5
    half_side = max(float(maxx) - float(minx), float(maxy) - float(miny)) * 0.5
    center = np.array([cx, cy], dtype=np.float64)
    return center, float(half_side)


def _empty_triangles() -> np.ndarray:
    """Create one empty triangle array in the expected schema."""
    return np.zeros((0, 3, 2), dtype=np.float32)


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/scalar objects into JSON-friendly values."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _triangulate_part_subprocess_entry(
    conn,
    polygon_wkb: bytes,
    min_triangle_area: float,
    min_triangle_height: float,
) -> None:
    """Run one normalized-part triangulation in a child process."""
    try:
        build_module = _load_build_module()
        poly_norm = shapely_wkb.loads(polygon_wkb)
        tris_raw = build_module._triangulate_polygon_triangle_only(poly_norm)
        tris_filtered, filter_stats = build_module._filter_degenerate_triangles(
            tris_raw,
            min_triangle_area=min_triangle_area,
            min_triangle_height=min_triangle_height,
        )
        payload = {
            "ok": True,
            "tris_raw": np.asarray(tris_raw, dtype=np.float32),
            "tris_filtered": np.asarray(tris_filtered, dtype=np.float32),
            "filter_stats": dict(filter_stats),
        }
    except Exception as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        conn.send(payload)
    except BrokenPipeError:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _triangulate_part_with_timeout(
    candidate_poly: Polygon,
    timeout_safe: float,
    min_triangle_area: float,
    min_triangle_height: float,
) -> dict[str, Any]:
    """Triangulate one kept normalized part in a guarded subprocess."""
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_triangulate_part_subprocess_entry,
        args=(send_conn, bytes(candidate_poly.wkb), float(min_triangle_area), float(min_triangle_height)),
        daemon=True,
    )

    start_time = time.perf_counter()
    proc.start()
    send_conn.close()

    payload: dict[str, Any] | None = None
    timed_out = False
    try:
        if recv_conn.poll(timeout=max(0.01, float(timeout_safe))):
            payload = recv_conn.recv()
        else:
            timed_out = True
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

    elapsed_sec = float(time.perf_counter() - start_time)
    exitcode = proc.exitcode

    if timed_out:
        return {
            "status": "failed",
            "failure_stage": "timeout_safe",
            "error": f"Triangulation exceeded timeout_safe={timeout_safe}",
            "raw_triangles": _empty_triangles(),
            "filtered_triangles": _empty_triangles(),
            "filter_stats": {"filtered_total": 0, "filtered_by_area_small": 0, "filtered_by_near_collinear": 0, "kept_count": 0},
            "elapsed_sec": elapsed_sec,
            "child_exitcode": exitcode,
        }

    if not payload:
        return {
            "status": "failed",
            "failure_stage": "triangulate_crash",
            "error": f"Child process exited without payload (exitcode={exitcode})",
            "raw_triangles": _empty_triangles(),
            "filtered_triangles": _empty_triangles(),
            "filter_stats": {"filtered_total": 0, "filtered_by_area_small": 0, "filtered_by_near_collinear": 0, "kept_count": 0},
            "elapsed_sec": elapsed_sec,
            "child_exitcode": exitcode,
        }

    if not bool(payload.get("ok", False)):
        return {
            "status": "failed",
            "failure_stage": "triangulate_error",
            "error": str(payload.get("error", "unknown error")),
            "raw_triangles": _empty_triangles(),
            "filtered_triangles": _empty_triangles(),
            "filter_stats": {"filtered_total": 0, "filtered_by_area_small": 0, "filtered_by_near_collinear": 0, "kept_count": 0},
            "elapsed_sec": elapsed_sec,
            "child_exitcode": exitcode,
        }

    tris_raw = np.asarray(payload.get("tris_raw", _empty_triangles()), dtype=np.float32)
    tris_filtered = np.asarray(payload.get("tris_filtered", _empty_triangles()), dtype=np.float32)
    filter_stats = dict(payload.get("filter_stats", {}))

    if tris_raw.shape[0] == 0:
        return {
            "status": "failed",
            "failure_stage": "triangulate_empty",
            "error": "Triangulation returned zero raw triangles.",
            "raw_triangles": tris_raw,
            "filtered_triangles": tris_filtered,
            "filter_stats": filter_stats,
            "elapsed_sec": elapsed_sec,
            "child_exitcode": exitcode,
        }

    if tris_filtered.shape[0] == 0:
        return {
            "status": "failed",
            "failure_stage": "all_triangles_filtered",
            "error": "All triangulated primitives were removed by degenerate filtering.",
            "raw_triangles": tris_raw,
            "filtered_triangles": tris_filtered,
            "filter_stats": filter_stats,
            "elapsed_sec": elapsed_sec,
            "child_exitcode": exitcode,
        }

    return {
        "status": "success",
        "failure_stage": None,
        "error": "",
        "raw_triangles": tris_raw,
        "filtered_triangles": tris_filtered,
        "filter_stats": filter_stats,
        "elapsed_sec": elapsed_sec,
        "child_exitcode": exitcode,
    }


def _select_overall_failure_stage(part_results: list[dict[str, Any]]) -> str | None:
    """Select one representative failure stage for the full row."""
    priority = [
        "timeout_safe",
        "triangulate_crash",
        "triangulate_error",
        "all_triangles_filtered",
        "triangulate_empty",
        "all_parts_filtered",
        "normalize_row_failed",
    ]
    observed = [result.get("failure_stage") for result in part_results if result.get("failure_stage")]
    for stage in priority:
        if stage in observed:
            return stage
    return observed[0] if observed else None


def _build_part_infos(build_module, row_geom, expanded_parts: list[Polygon], norm_max: float) -> list[dict[str, Any]]:
    """Build per-part row-normalized diagnostics while preserving raw-part order."""
    center, half_side = _compute_row_frame(row_geom)
    if center is None or half_side < build_module._NORMALIZATION_EPS:
        return []

    part_infos: list[dict[str, Any]] = []
    for part_index, raw_poly in enumerate(expanded_parts, start=1):
        color = _PLOT_COLORS[(part_index - 1) % len(_PLOT_COLORS)]
        poly_norm = build_module._normalize_polygon_with_row_frame(
            poly=raw_poly,
            center=center,
            half_side=float(half_side),
            norm_max=float(norm_max),
        )
        if poly_norm is None:
            part_infos.append(
                {
                    "part_index": int(part_index),
                    "color": color,
                    "raw_poly": raw_poly,
                    "normalized_poly": None,
                    "is_valid": False,
                    "shell_hole_touching": False,
                    "filtered": True,
                    "filter_reason": "normalize_failed",
                    "hole_count": int(len(raw_poly.interiors)),
                    "node_count": 0,
                    "min_edge": None,
                }
            )
            continue

        is_valid = bool(poly_norm.is_valid)
        shell_hole_touching = bool(build_module._polygon_has_shell_hole_intersection(poly_norm))
        filtered = (not is_valid) or shell_hole_touching
        if not is_valid:
            filter_reason = "invalid_part"
        elif shell_hole_touching:
            filter_reason = "shell_hole_touching"
        else:
            filter_reason = ""

        part_infos.append(
            {
                "part_index": int(part_index),
                "color": color,
                "raw_poly": raw_poly,
                "normalized_poly": poly_norm,
                "is_valid": bool(is_valid),
                "shell_hole_touching": bool(shell_hole_touching),
                "filtered": bool(filtered),
                "filter_reason": filter_reason,
                "hole_count": int(len(poly_norm.interiors)),
                "node_count": int(build_module._polygon_node_count(poly_norm)),
                "min_edge": float(build_module._polygon_min_edge(poly_norm)),
            }
        )
    return part_infos


def _evaluate_row(
    build_module,
    row_geom,
    *,
    safe_mode: str,
    part_safe: int,
    node_safe: int,
    hole_safe: int,
    edge_safe: float,
    timeout_safe: float,
    norm_max: float,
    min_triangle_area: float,
    min_triangle_height: float,
) -> dict[str, Any]:
    """Evaluate one row under the current row-level triangulation semantics."""
    expanded_parts = [poly for poly, _ in build_module._expand_geometry_to_polygons(row_geom)]
    polygon_type = _classify_polygon_type(row_geom)

    if not expanded_parts:
        return {
            "expanded_parts": [],
            "part_infos": [],
            "triangulation_status": "failed",
            "failure_stage": "non_polygon_row",
            "status_note": "FAILED: row is not a Polygon/MultiPolygon",
            "polygon_type": polygon_type,
            "build_would_isolate_row": False,
            "row_degenerated": False,
            "part_results": [],
        }

    part_infos = _build_part_infos(build_module, row_geom, expanded_parts, norm_max=float(norm_max))
    normalized_part_count = sum(1 for info in part_infos if info["normalized_poly"] is not None)
    kept_infos = [info for info in part_infos if info["normalized_poly"] is not None and not info["filtered"]]

    if normalized_part_count == 0:
        return {
            "expanded_parts": expanded_parts,
            "part_infos": part_infos,
            "triangulation_status": "failed",
            "failure_stage": "normalize_row_failed",
            "status_note": "FAILED: row-level normalization produced no valid parts",
            "polygon_type": polygon_type,
            "build_would_isolate_row": False,
            "row_degenerated": False,
            "part_results": [],
        }

    if not kept_infos:
        return {
            "expanded_parts": expanded_parts,
            "part_infos": part_infos,
            "triangulation_status": "failed",
            "failure_stage": "all_parts_filtered",
            "status_note": "FAILED: all normalized parts were filtered",
            "polygon_type": polygon_type,
            "build_would_isolate_row": False,
            "row_degenerated": False,
            "part_results": [],
        }

    build_would_isolate_row = bool(
        build_module._should_isolate_row(
            safe_mode=str(safe_mode),
            filtered_parts=[info["normalized_poly"] for info in kept_infos],
            part_safe=int(part_safe),
            node_safe=int(node_safe),
            hole_safe=int(hole_safe),
            edge_safe=float(edge_safe),
        )
    )

    part_results: list[dict[str, Any]] = []
    for info in kept_infos:
        tri_result = _triangulate_part_with_timeout(
            candidate_poly=info["normalized_poly"],
            timeout_safe=float(timeout_safe),
            min_triangle_area=float(min_triangle_area),
            min_triangle_height=float(min_triangle_height),
        )
        tri_result.update(
            {
                "part_index": int(info["part_index"]),
                "color": str(info["color"]),
                "normalized_poly": info["normalized_poly"],
            }
        )
        part_results.append(tri_result)

    failed_results = [result for result in part_results if result.get("status") != "success"]
    had_part_filter = any(bool(info["filtered"]) for info in part_infos)
    filtered_triangle_total = int(
        sum(int(dict(result.get("filter_stats", {})).get("filtered_total", 0)) for result in part_results)
    )

    if failed_results:
        failure_stage = _select_overall_failure_stage(failed_results)
        return {
            "expanded_parts": expanded_parts,
            "part_infos": part_infos,
            "triangulation_status": "failed",
            "failure_stage": failure_stage,
            "status_note": f"FAILED: {failure_stage}",
            "polygon_type": polygon_type,
            "build_would_isolate_row": bool(build_would_isolate_row),
            "row_degenerated": False,
            "part_results": part_results,
            "filtered_triangle_total": filtered_triangle_total,
            "had_part_filter": had_part_filter,
        }

    row_degenerated = bool(had_part_filter or filtered_triangle_total > 0)
    return {
        "expanded_parts": expanded_parts,
        "part_infos": part_infos,
        "triangulation_status": "success",
        "failure_stage": None,
        "status_note": f"SUCCESS: {len(part_results)}/{len(kept_infos)} kept part(s) triangulated",
        "polygon_type": polygon_type,
        "build_would_isolate_row": bool(build_would_isolate_row),
        "row_degenerated": bool(row_degenerated),
        "part_results": part_results,
        "filtered_triangle_total": filtered_triangle_total,
        "had_part_filter": had_part_filter,
    }


def _add_polygon_patch(
    ax,
    poly: Polygon,
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float = 0.45,
    hatch: str | None = None,
) -> None:
    """Draw one polygon with holes using matplotlib patches."""
    ext = np.asarray(poly.exterior.coords, dtype=np.float64)
    if ext.ndim != 2 or ext.shape[0] < 3:
        return

    ax.add_patch(
        MplPolygonPatch(
            ext[:, :2],
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.2,
            alpha=alpha,
            hatch=hatch,
        )
    )
    for interior in poly.interiors:
        hole = np.asarray(interior.coords, dtype=np.float64)
        if hole.ndim != 2 or hole.shape[0] < 3:
            continue
        ax.add_patch(
            MplPolygonPatch(
                hole[:, :2],
                closed=True,
                facecolor="white",
                edgecolor=edgecolor,
                linewidth=1.0,
                alpha=1.0,
            )
        )


def _plot_geometry_panel(
    ax,
    geoms: list[Polygon],
    title: str,
    *,
    annotate_ids: bool = False,
    colors: list[str] | None = None,
) -> None:
    """Plot polygon list on one axis."""
    for idx, poly in enumerate(geoms):
        color = colors[idx % len(colors)] if colors else _PLOT_COLORS[idx % len(_PLOT_COLORS)]
        _add_polygon_patch(ax, poly, facecolor=color, edgecolor="#1d3557", alpha=0.45)
        if annotate_ids:
            rep = poly.representative_point()
            ax.text(float(rep.x), float(rep.y), str(idx + 1), fontsize=8, color="#111827", ha="center", va="center")

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    if geoms:
        minx = min(poly.bounds[0] for poly in geoms)
        miny = min(poly.bounds[1] for poly in geoms)
        maxx = max(poly.bounds[2] for poly in geoms)
        maxy = max(poly.bounds[3] for poly in geoms)
        dx = max(maxx - minx, 1e-9)
        dy = max(maxy - miny, 1e-9)
        pad_x = max(dx * _GEOM_PANEL_PAD_RATIO, 1e-9)
        pad_y = max(dy * _GEOM_PANEL_PAD_RATIO, 1e-9)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
    _compactify_axis(ax)


def _plot_normalized_parts_panel(ax, part_infos: list[dict[str, Any]], title: str, norm_max: float) -> None:
    """Plot all row-normalized parts, including those that later get filtered."""
    geoms = [info["normalized_poly"] for info in part_infos if info["normalized_poly"] is not None]
    colors = [str(info["color"]) for info in part_infos if info["normalized_poly"] is not None]
    _plot_geometry_panel(ax, geoms, title=title, annotate_ids=True, colors=colors)
    square = np.array(
        [[-norm_max, -norm_max], [norm_max, -norm_max], [norm_max, norm_max], [-norm_max, norm_max], [-norm_max, -norm_max]],
        dtype=np.float64,
    )
    ax.plot(square[:, 0], square[:, 1], "--", color="#d90429", linewidth=1.0, alpha=0.85)
    pad = max(float(norm_max) * _NORM_PANEL_PAD_RATIO, 1e-3)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    _compactify_axis(ax)


def _plot_filtered_parts_panel(ax, part_infos: list[dict[str, Any]], title: str, norm_max: float) -> None:
    """Plot filtered-vs-kept normalized parts."""
    for info in part_infos:
        poly = info["normalized_poly"]
        if poly is None:
            continue
        if info["filtered"]:
            facecolor = "#d1d5db"
            edgecolor = "#b91c1c"
            alpha = 0.30
            hatch = "//"
        else:
            facecolor = str(info["color"])
            edgecolor = "#1d3557"
            alpha = 0.50
            hatch = None
        _add_polygon_patch(ax, poly, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, hatch=hatch)
        rep = poly.representative_point()
        suffix = "F" if info["filtered"] else "K"
        ax.text(float(rep.x), float(rep.y), f"{info['part_index']}{suffix}", fontsize=8, ha="center", va="center")

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    square = np.array(
        [[-norm_max, -norm_max], [norm_max, -norm_max], [norm_max, norm_max], [-norm_max, norm_max], [-norm_max, -norm_max]],
        dtype=np.float64,
    )
    ax.plot(square[:, 0], square[:, 1], "--", color="#d90429", linewidth=1.0, alpha=0.85)
    pad = max(float(norm_max) * _NORM_PANEL_PAD_RATIO, 1e-3)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    _compactify_axis(ax)


def _build_triangle_segments(tris: np.ndarray) -> np.ndarray:
    """Convert `[T,3,2]` triangles into line segments for `LineCollection`."""
    tri_np = np.asarray(tris, dtype=np.float64)
    if tri_np.ndim != 3 or tri_np.shape[0] == 0:
        return np.zeros((0, 2, 2), dtype=np.float64)
    seg_ab = tri_np[:, [0, 1], :]
    seg_bc = tri_np[:, [1, 2], :]
    seg_ca = tri_np[:, [2, 0], :]
    return np.concatenate([seg_ab, seg_bc, seg_ca], axis=0)


def _plot_triangle_panel(
    ax,
    triangle_groups: list[dict[str, Any]],
    title: str,
    *,
    norm_max: float,
    status_text: str | None = None,
) -> None:
    """Plot one triangulation panel in normalized coordinates."""
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    square = np.array(
        [[-norm_max, -norm_max], [norm_max, -norm_max], [norm_max, norm_max], [-norm_max, norm_max], [-norm_max, -norm_max]],
        dtype=np.float64,
    )
    ax.plot(square[:, 0], square[:, 1], "--", color="#d90429", linewidth=1.0, alpha=0.85)
    pad = max(float(norm_max) * _NORM_PANEL_PAD_RATIO, 1e-3)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)

    has_triangles = False
    for group in triangle_groups:
        tris = np.asarray(group.get("triangles", _empty_triangles()), dtype=np.float64)
        if tris.ndim != 3 or tris.shape[0] == 0:
            continue
        has_triangles = True
        color = str(group.get("color", "#4c78a8"))
        segments = _build_triangle_segments(tris)
        ax.add_collection(LineCollection(segments, colors=color, linewidths=0.7, alpha=0.9))
        poly_norm = group.get("normalized_poly")
        if poly_norm is not None:
            rep = poly_norm.representative_point()
            ax.text(
                float(rep.x),
                float(rep.y),
                f"P{int(group.get('part_index', 0))}",
                fontsize=8,
                color=color,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.5},
            )

    if not has_triangles:
        ax.text(0.5, 0.5, "No triangles", ha="center", va="center", fontsize=11, transform=ax.transAxes)

    if status_text:
        ax.text(
            0.02,
            0.98,
            status_text,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db", "pad": 2.0},
        )
    _compactify_axis(ax)


def _format_multiline_json(data: dict[str, Any]) -> str:
    """Format metadata as readable multiline text."""
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, (list, dict)):
            lines.append(f"{key}: {json.dumps(_to_jsonable(value), ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _format_info_panel_text(info_payload: dict[str, Any], part_summary_payload: dict[str, Any]) -> str:
    """Build a concise text block for the right-side overview panel.

    The PNG should prioritize geometry panels over verbose metadata. We therefore
    keep the on-figure text concise, while the companion JSON file still stores
    the complete payload.
    """
    selected = info_payload.get("selected_part_summary") or {}
    lines: list[str] = [
        "Row Overview",
        f"row_index: {info_payload.get('row_index')}",
        f"part_index: {info_payload.get('part_index')} (requested={info_payload.get('requested_part_index')})",
        f"polygon_type: {info_payload.get('polygon_type')}",
        f"raw_part_count: {info_payload.get('raw_part_count')}",
        f"normalized_part_count: {info_payload.get('normalized_part_count')}",
        f"filtered_part_count: {info_payload.get('filtered_part_count')}",
        f"kept_part_count: {info_payload.get('kept_part_count')}",
        "",
        "Execution",
        f"safe_mode: {info_payload.get('safe_mode')}",
        f"build_would_isolate_row: {info_payload.get('build_would_isolate_row')}",
        f"triangulation_status: {info_payload.get('triangulation_status')}",
        f"failure_stage: {info_payload.get('failure_stage')}",
        f"row_degenerated: {info_payload.get('row_degenerated')}",
        f"status_note: {info_payload.get('status_note')}",
        "",
        "Thresholds",
        f"part_safe: {info_payload.get('part_safe')}",
        f"node_safe: {info_payload.get('node_safe')}",
        f"hole_safe: {info_payload.get('hole_safe')}",
        f"edge_safe: {info_payload.get('edge_safe')}",
        f"timeout_safe: {info_payload.get('timeout_safe')}",
        f"norm_max: {info_payload.get('norm_max')}",
        f"min_triangle_area: {info_payload.get('min_triangle_area')}",
        f"min_triangle_height: {info_payload.get('min_triangle_height')}",
        "",
        "Selected Part",
        f"filtered: {selected.get('filtered')}",
        f"filter_reason: {selected.get('filter_reason')}",
        f"is_valid: {selected.get('is_valid')}",
        f"shell_hole_touching: {selected.get('shell_hole_touching')}",
        f"hole_count: {selected.get('hole_count')}",
        f"node_count: {selected.get('node_count')}",
        f"min_edge: {selected.get('min_edge')}",
        "",
        "Part Statuses",
    ]

    part_statuses = list(part_summary_payload.get("part_statuses", []))
    max_lines = 12
    for idx, part in enumerate(part_statuses[:max_lines], start=1):
        lines.append(
            "P{part_index}: filtered={filtered}, reason={reason}, valid={valid}, "
            "touch={touch}, holes={holes}, nodes={nodes}, min_edge={edge}".format(
                part_index=part.get("part_index"),
                filtered=part.get("filtered"),
                reason=part.get("filter_reason"),
                valid=part.get("is_valid"),
                touch=part.get("shell_hole_touching"),
                holes=part.get("hole_count"),
                nodes=part.get("node_count"),
                edge=part.get("min_edge"),
            )
        )
    if len(part_statuses) > max_lines:
        lines.append(f"... {len(part_statuses) - max_lines} more parts omitted in PNG; see JSON for full details")
    return "\n".join(lines)


def main() -> None:
    """CLI main function."""
    project_root = _inject_repo_root()
    build_module = _load_build_module()

    parser = argparse.ArgumentParser(description="Visualize one row under the row-level triangulation pipeline.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory that contains exactly one .shp dataset.")
    parser.add_argument("--row_index", type=int, required=True, help="Original geometry row index in the source .shp.")
    parser.add_argument(
        "--part_index",
        type=int,
        default=1,
        help="1-based raw polygon part index within the selected row. Values above part count fall back to the last part.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "outputs" / "polygon_viz"),
        help="Directory for generated PNG and metadata JSON.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    parser.add_argument("--safe_mode", type=str, default="risky", choices=["all", "risky", "off"])
    parser.add_argument("--part_safe", type=int, default=1)
    parser.add_argument("--node_safe", type=int, default=2048)
    parser.add_argument("--hole_safe", type=int, default=1)
    parser.add_argument("--edge_safe", type=float, default=1e-5)
    parser.add_argument(
        "--timeout_safe",
        "--timeout",
        dest="timeout_safe",
        type=float,
        default=_DEFAULT_TIMEOUT_SAFE,
        help="Maximum seconds to wait for one kept part during guarded triangulation visualization.",
    )
    parser.add_argument("--norm_max", type=float, default=1.0)
    parser.add_argument("--min_triangle_area", type=float, default=_DEFAULT_MIN_TRIANGLE_AREA)
    parser.add_argument("--min_triangle_height", type=float, default=_DEFAULT_MIN_TRIANGLE_HEIGHT)
    args = parser.parse_args()

    if args.timeout_safe <= 0:
        raise ValueError(f"`timeout_safe` must be > 0, got {args.timeout_safe}")
    if args.norm_max <= 0:
        raise ValueError(f"`norm_max` must be > 0, got {args.norm_max}")

    input_dir = Path(args.input_dir)
    input_path = _resolve_single_shp(input_dir)
    output_dir = Path(args.output_dir) / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(input_path)
    target = _resolve_row_part_target(build_module, gdf, int(args.row_index), int(args.part_index))
    file_stem = f"row_{int(target['row_index']):06d}_part_{int(target['part_index']):02d}"

    row_eval = _evaluate_row(
        build_module,
        target["row_geom"],
        safe_mode=str(args.safe_mode),
        part_safe=int(args.part_safe),
        node_safe=int(args.node_safe),
        hole_safe=int(args.hole_safe),
        edge_safe=float(args.edge_safe),
        timeout_safe=float(args.timeout_safe),
        norm_max=float(args.norm_max),
        min_triangle_area=float(args.min_triangle_area),
        min_triangle_height=float(args.min_triangle_height),
    )

    raw_triangle_groups = []
    filtered_triangle_groups = []
    part_summaries: list[dict[str, Any]] = []
    for result in row_eval["part_results"]:
        normalized_poly = result.get("normalized_poly")
        raw_triangle_groups.append(
            {
                "part_index": int(result.get("part_index", 0)),
                "color": str(result.get("color", "#4c78a8")),
                "normalized_poly": normalized_poly,
                "triangles": result.get("raw_triangles", _empty_triangles()),
            }
        )
        filtered_triangle_groups.append(
            {
                "part_index": int(result.get("part_index", 0)),
                "color": str(result.get("color", "#4c78a8")),
                "normalized_poly": normalized_poly,
                "triangles": result.get("filtered_triangles", _empty_triangles()),
            }
        )
        filter_stats = dict(result.get("filter_stats", {}))
        part_summaries.append(
            {
                "part_index": int(result.get("part_index", 0)),
                "status": str(result.get("status")),
                "failure_stage": result.get("failure_stage"),
                "raw_triangle_count": int(np.asarray(result.get("raw_triangles", _empty_triangles())).shape[0]),
                "kept_triangle_count": int(np.asarray(result.get("filtered_triangles", _empty_triangles())).shape[0]),
                "filtered_total": int(filter_stats.get("filtered_total", 0)),
                "filtered_by_area_small": int(filter_stats.get("filtered_by_area_small", 0)),
                "filtered_by_near_collinear": int(filter_stats.get("filtered_by_near_collinear", 0)),
                "elapsed_sec": float(result.get("elapsed_sec", 0.0)),
                "child_exitcode": result.get("child_exitcode"),
                "error": str(result.get("error", "")),
            }
        )

    selected_part_summary = next(
        (info for info in row_eval["part_infos"] if int(info["part_index"]) == int(target["part_index"])),
        None,
    )
    if selected_part_summary is not None:
        selected_part_summary_json = {
            "part_index": int(selected_part_summary["part_index"]),
            "filtered": bool(selected_part_summary["filtered"]),
            "filter_reason": str(selected_part_summary["filter_reason"]),
            "is_valid": bool(selected_part_summary["is_valid"]),
            "shell_hole_touching": bool(selected_part_summary["shell_hole_touching"]),
            "hole_count": int(selected_part_summary["hole_count"]),
            "node_count": int(selected_part_summary["node_count"]),
            "min_edge": selected_part_summary["min_edge"],
        }
    else:
        selected_part_summary_json = None

    info_payload = {
        "input_path": str(input_path),
        "row_index": int(target["row_index"]),
        "part_index": int(target["part_index"]),
        "requested_part_index": int(target["requested_part_index"]),
        "part_count": int(target["part_count"]),
        "from_multipolygon": bool(target["from_multipolygon"]),
        "polygon_type": str(row_eval["polygon_type"]),
        "safe_mode": str(args.safe_mode),
        "part_safe": int(args.part_safe),
        "node_safe": int(args.node_safe),
        "hole_safe": int(args.hole_safe),
        "edge_safe": float(args.edge_safe),
        "timeout_safe": float(args.timeout_safe),
        "norm_max": float(args.norm_max),
        "min_triangle_area": float(args.min_triangle_area),
        "min_triangle_height": float(args.min_triangle_height),
        "triangulation_status": str(row_eval["triangulation_status"]),
        "failure_stage": row_eval["failure_stage"],
        "status_note": str(row_eval["status_note"]),
        "build_would_isolate_row": bool(row_eval["build_would_isolate_row"]),
        "visualization_execution_mode": "guarded_subprocess_per_kept_part",
        "row_degenerated": bool(row_eval["row_degenerated"]),
        "raw_part_count": int(len(row_eval["expanded_parts"])),
        "normalized_part_count": int(sum(1 for info in row_eval["part_infos"] if info["normalized_poly"] is not None)),
        "filtered_part_count": int(sum(1 for info in row_eval["part_infos"] if info["filtered"])),
        "kept_part_count": int(sum(1 for info in row_eval["part_infos"] if info["normalized_poly"] is not None and not info["filtered"])),
        "selected_part_summary": selected_part_summary_json,
        "part_summaries": part_summaries,
    }

    part_summary_payload = {
        "part_statuses": [
            {
                "part_index": int(info["part_index"]),
                "filtered": bool(info["filtered"]),
                "filter_reason": str(info["filter_reason"]),
                "is_valid": bool(info["is_valid"]),
                "shell_hole_touching": bool(info["shell_hole_touching"]),
                "hole_count": int(info["hole_count"]),
                "node_count": int(info["node_count"]),
                "min_edge": info["min_edge"],
            }
            for info in row_eval["part_infos"]
        ]
    }

    figure_text = _format_info_panel_text(info_payload, part_summary_payload)

    fig = plt.figure(figsize=(25, 12), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.12, 1.12, 1.12, 0.92],
        height_ratios=[1.0, 1.0],
        wspace=0.04,
        hspace=0.06,
    )
    ax_raw_row = fig.add_subplot(gs[0, 0])
    ax_selected = fig.add_subplot(gs[0, 1])
    ax_norm = fig.add_subplot(gs[0, 2])
    ax_filtered = fig.add_subplot(gs[1, 0])
    ax_raw_tri = fig.add_subplot(gs[1, 1])
    ax_filtered_tri = fig.add_subplot(gs[1, 2])
    ax_info = fig.add_subplot(gs[:, 3])

    _plot_geometry_panel(ax_raw_row, list(row_eval["expanded_parts"]), title="Raw Row Geometry", annotate_ids=True)
    _plot_geometry_panel(ax_selected, [target["selected_poly"]], title="Selected Raw Part", annotate_ids=False)
    _plot_normalized_parts_panel(ax_norm, row_eval["part_infos"], title="Row-Normalized Parts", norm_max=float(args.norm_max))
    _plot_filtered_parts_panel(ax_filtered, row_eval["part_infos"], title="Filtered Parts", norm_max=float(args.norm_max))

    _plot_triangle_panel(
        ax_raw_tri,
        raw_triangle_groups,
        title="Raw Triangulation",
        norm_max=float(args.norm_max),
        status_text=str(row_eval["status_note"]),
    )
    _plot_triangle_panel(
        ax_filtered_tri,
        filtered_triangle_groups,
        title="Filtered Triangulation",
        norm_max=float(args.norm_max),
        status_text=str(row_eval["status_note"]),
    )
    ax_info.axis("off")
    ax_info.set_title("Overview", fontsize=13, loc="left", pad=10)
    ax_info.text(
        0.0,
        1.0,
        figure_text,
        va="top",
        ha="left",
        fontsize=8.8,
        family="monospace",
        wrap=True,
    )

    fig.suptitle(f"Row Triangulation Visualization: {file_stem}", fontsize=16)

    png_path = output_dir / f"{file_stem}.png"
    json_path = output_dir / f"{file_stem}.json"
    fig.savefig(png_path, dpi=max(72, int(args.dpi)), bbox_inches="tight")
    plt.close(fig)

    json_path.write_text(json.dumps(_to_jsonable(info_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Saved visualization to {png_path}")
    print(f"[INFO] Saved metadata to {json_path}")


if __name__ == "__main__":
    main()
