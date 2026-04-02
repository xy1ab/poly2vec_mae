"""Visualize one polygon part together with triangulation diagnostics."""

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

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CURRENT_DIR.parent
    _REPO_ROOT = _PROJECT_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    ensure_cuda_runtime_libs = importlib.import_module(
        "mae_pretrain.scripts.runtime_bootstrap"
    ).ensure_cuda_runtime_libs
else:
    from .runtime_bootstrap import ensure_cuda_runtime_libs


_DEFAULT_TIMEOUT_SEC = 20.0
_DEFAULT_MIN_TRIANGLE_AREA = 1e-8
_DEFAULT_MIN_TRIANGLE_HEIGHT = 1e-5
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


def _inject_repo_root() -> Path:
    """Inject repository root into `sys.path` for direct script execution."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def _load_build_module():
    """Load build_dataset_triangle lazily."""
    if __package__ in {None, ""}:
        return importlib.import_module("mae_pretrain.src.datasets.build_dataset_triangle")
    from ..src.datasets import build_dataset_triangle as build_module

    return build_module


def _load_diag_module():
    """Load run_polygon_diagnosis lazily for shared static-analysis helpers."""
    if __package__ in {None, ""}:
        return importlib.import_module("mae_pretrain.scripts.run_polygon_diagnosis")
    from . import run_polygon_diagnosis as diag_module

    return diag_module


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
    """Flatten a geometry into polygon parts."""
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


def _add_polygon_patch(ax, poly: Polygon, facecolor: str, edgecolor: str, alpha: float = 0.45) -> None:
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
    annotate_ids: bool = False,
    colors: list[str] | None = None,
) -> None:
    """Plot polygon list on one axis."""
    edgecolor = "#1d3557"

    for idx, poly in enumerate(geoms):
        color = colors[idx % len(colors)] if colors else _PLOT_COLORS[idx % len(_PLOT_COLORS)]
        _add_polygon_patch(ax, poly, facecolor=color, edgecolor=edgecolor)
        if annotate_ids:
            rep = poly.representative_point()
            ax.text(
                float(rep.x),
                float(rep.y),
                str(idx + 1),
                fontsize=8,
                color="#111827",
                ha="center",
                va="center",
            )

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
        pad_x = dx * 0.08
        pad_y = dy * 0.08
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)


def _plot_normalized_panel(ax, geoms: list[Polygon], title: str, colors: list[str] | None = None) -> None:
    """Plot normalized candidates and the reference [-1,1] box."""
    _plot_geometry_panel(ax, geoms, title=title, annotate_ids=True, colors=colors)
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=np.float64)
    ax.plot(square[:, 0], square[:, 1], "--", color="#d90429", linewidth=1.0, alpha=0.85)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)


def _format_multiline_json(data: dict[str, Any]) -> str:
    """Format metadata as compact readable text for the info panel."""
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, (list, dict)):
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _resolve_row_part_target(build_module, gdf: gpd.GeoDataFrame, row_index: int, part_index: int) -> dict[str, Any]:
    """Resolve one source row plus one selected polygon part for visualization."""
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
        "selection_mode": "row_part",
        "row_index": int(row_index),
        "part_index": int(resolved_part_index),
        "requested_part_index": int(requested_part_index),
        "part_count": int(len(expanded_parts)),
        "row_geom": row_geom,
        "expanded_parts": [poly for poly, _ in expanded_parts],
        "selected_poly": selected_poly,
        "from_multipolygon": bool(from_multi),
    }


def _triangulate_candidate_subprocess_entry(
    conn,
    polygon_wkb: bytes,
    min_triangle_area: float,
    min_triangle_height: float,
) -> None:
    """Run one candidate triangulation in a child process for timeout isolation."""
    try:
        build_module = _load_build_module()
        poly_norm = shapely_wkb.loads(polygon_wkb)
        tris_raw = build_module._triangulate_polygon_with_holes(poly_norm)
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


def _empty_triangles() -> np.ndarray:
    """Create one empty triangle array in the expected schema."""
    return np.zeros((0, 3, 2), dtype=np.float32)


def _triangulate_candidate_with_timeout(
    candidate_poly: Polygon,
    timeout_sec: float,
    min_triangle_area: float,
    min_triangle_height: float,
) -> dict[str, Any]:
    """Triangulate one candidate polygon with a hard timeout guard."""
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_triangulate_candidate_subprocess_entry,
        args=(send_conn, bytes(candidate_poly.wkb), float(min_triangle_area), float(min_triangle_height)),
        daemon=True,
    )

    start_time = time.perf_counter()
    proc.start()
    send_conn.close()

    payload: dict[str, Any] | None = None
    timed_out = False
    try:
        if recv_conn.poll(timeout=max(0.01, float(timeout_sec))):
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
            "failure_stage": "triangulate_timeout",
            "error": f"Triangulation exceeded timeout_sec={timeout_sec}",
            "raw_triangles": _empty_triangles(),
            "filtered_triangles": _empty_triangles(),
            "filter_stats": {
                "filtered_total": 0,
                "filtered_by_area_small": 0,
                "filtered_by_near_collinear": 0,
                "kept_count": 0,
            },
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
            "filter_stats": {
                "filtered_total": 0,
                "filtered_by_area_small": 0,
                "filtered_by_near_collinear": 0,
                "kept_count": 0,
            },
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
            "filter_stats": {
                "filtered_total": 0,
                "filtered_by_area_small": 0,
                "filtered_by_near_collinear": 0,
                "kept_count": 0,
            },
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


def _select_overall_failure_stage(candidate_results: list[dict[str, Any]]) -> str | None:
    """Select one representative failure stage for the full sample."""
    priority = [
        "triangulate_timeout",
        "triangulate_crash",
        "triangulate_error",
        "all_triangles_filtered",
        "triangulate_empty",
        "normalize_failed",
    ]
    observed = [result.get("failure_stage") for result in candidate_results if result.get("failure_stage")]
    for stage in priority:
        if stage in observed:
            return stage
    return observed[0] if observed else None


def _evaluate_selected_part(build_module, poly_raw: Polygon, timeout_sec: float) -> dict[str, Any]:
    """Evaluate triangulation pipeline for one selected polygon part."""
    repaired = poly_raw.buffer(0)
    repaired_polys = _iter_polygons(repaired)
    candidate_polys: list[Polygon] = []
    normalized_polys: list[Polygon] = []
    candidate_results: list[dict[str, Any]] = []

    if repaired.is_empty:
        return {
            "repaired": repaired,
            "repaired_polys": repaired_polys,
            "candidate_polys": candidate_polys,
            "normalized_polys": normalized_polys,
            "candidate_results": candidate_results,
            "triangulation_status": "failed",
            "failure_stage": "repair_empty",
            "status_note": "FAILED: repaired geometry is empty",
        }

    if repaired.geom_type != "Polygon":
        return {
            "repaired": repaired,
            "repaired_polys": repaired_polys,
            "candidate_polys": candidate_polys,
            "normalized_polys": normalized_polys,
            "candidate_results": candidate_results,
            "triangulation_status": "failed",
            "failure_stage": "repair_non_polygon",
            "status_note": f"FAILED: repaired geometry type is {repaired.geom_type}",
        }

    candidate_polys = build_module._prepare_polygon_candidates(repaired)
    if not candidate_polys:
        return {
            "repaired": repaired,
            "repaired_polys": repaired_polys,
            "candidate_polys": candidate_polys,
            "normalized_polys": normalized_polys,
            "candidate_results": candidate_results,
            "triangulation_status": "failed",
            "failure_stage": "no_candidate_polygons",
            "status_note": "FAILED: no candidate polygons after repair/preparation",
        }

    for candidate_index, candidate_poly in enumerate(candidate_polys, start=1):
        color = _PLOT_COLORS[(candidate_index - 1) % len(_PLOT_COLORS)]
        poly_norm = build_module._normalize_polygon_to_unit_box(candidate_poly)
        if poly_norm is None:
            candidate_results.append(
                {
                    "candidate_index": int(candidate_index),
                    "color": color,
                    "normalized_poly": None,
                    "status": "failed",
                    "failure_stage": "normalize_failed",
                    "error": "Normalization returned None.",
                    "raw_triangles": _empty_triangles(),
                    "filtered_triangles": _empty_triangles(),
                    "filter_stats": {
                        "filtered_total": 0,
                        "filtered_by_area_small": 0,
                        "filtered_by_near_collinear": 0,
                        "kept_count": 0,
                    },
                    "elapsed_sec": 0.0,
                    "child_exitcode": None,
                }
            )
            continue

        normalized_polys.append(poly_norm)
        tri_result = _triangulate_candidate_with_timeout(
            candidate_poly=poly_norm,
            timeout_sec=timeout_sec,
            min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
            min_triangle_height=_DEFAULT_MIN_TRIANGLE_HEIGHT,
        )
        tri_result.update(
            {
                "candidate_index": int(candidate_index),
                "color": color,
                "normalized_poly": poly_norm,
            }
        )
        candidate_results.append(tri_result)

    successful_results = [result for result in candidate_results if result.get("status") == "success"]
    if not successful_results:
        failure_stage = _select_overall_failure_stage(candidate_results)
        return {
            "repaired": repaired,
            "repaired_polys": repaired_polys,
            "candidate_polys": candidate_polys,
            "normalized_polys": normalized_polys,
            "candidate_results": candidate_results,
            "triangulation_status": "failed",
            "failure_stage": failure_stage,
            "status_note": f"FAILED: {failure_stage}",
        }

    failed_results = [result for result in candidate_results if result.get("status") != "success"]
    if failed_results:
        return {
            "repaired": repaired,
            "repaired_polys": repaired_polys,
            "candidate_polys": candidate_polys,
            "normalized_polys": normalized_polys,
            "candidate_results": candidate_results,
            "triangulation_status": "partial_success",
            "failure_stage": _select_overall_failure_stage(failed_results),
            "status_note": (
                f"PARTIAL SUCCESS: {len(successful_results)}/{len(candidate_results)} "
                "candidate(s) produced kept triangles"
            ),
        }

    return {
        "repaired": repaired,
        "repaired_polys": repaired_polys,
        "candidate_polys": candidate_polys,
        "normalized_polys": normalized_polys,
        "candidate_results": candidate_results,
        "triangulation_status": "success",
        "failure_stage": None,
        "status_note": f"SUCCESS: {len(successful_results)}/{len(candidate_results)} candidate(s) succeeded",
    }


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
    status_text: str | None = None,
) -> None:
    """Plot one triangulation panel in normalized coordinates."""
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=np.float64)
    ax.plot(square[:, 0], square[:, 1], "--", color="#d90429", linewidth=1.0, alpha=0.85)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)

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
                f"C{int(group.get('candidate_index', 0))}",
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


def main() -> None:
    """CLI main function."""
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()
    build_module = _load_build_module()
    diag_module = _load_diag_module()

    parser = argparse.ArgumentParser(description="Visualize one polygon part together with triangulation results.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory that contains exactly one .shp dataset.",
    )
    parser.add_argument("--row_index", type=int, required=True, help="Original geometry row index in the source .shp.")
    parser.add_argument(
        "--part_index",
        type=int,
        default=1,
        help="1-based polygon part index within the selected row. Values above part count fall back to the last part.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "outputs" / "polygon_viz"),
        help="Directory for generated PNG and metadata JSON.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_SEC,
        help="Triangulation timeout in seconds. Timeout is treated as triangulation failure.",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        raise ValueError(f"`timeout` must be > 0, got {args.timeout}")

    input_dir = Path(args.input_dir)
    input_path = _resolve_single_shp(input_dir)
    output_dir = Path(args.output_dir) / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(input_path)
    target = _resolve_row_part_target(build_module, gdf, int(args.row_index), int(args.part_index))
    file_stem = f"row_{int(target['row_index']):06d}_part_{int(target['part_index']):02d}"

    poly_raw = target["selected_poly"]
    assert poly_raw is not None

    triangulation_eval = _evaluate_selected_part(build_module, poly_raw, timeout_sec=float(args.timeout))
    diag = diag_module._static_diagnose_polygon(build_module, poly_raw)
    risk = diag_module._static_assess_risk(diag)

    raw_triangle_groups = []
    filtered_triangle_groups = []
    candidate_summaries: list[dict[str, Any]] = []
    for result in triangulation_eval["candidate_results"]:
        normalized_poly = result.get("normalized_poly")
        raw_triangle_groups.append(
            {
                "candidate_index": int(result.get("candidate_index", 0)),
                "color": str(result.get("color", "#4c78a8")),
                "normalized_poly": normalized_poly,
                "triangles": result.get("raw_triangles", _empty_triangles()),
            }
        )
        filtered_triangle_groups.append(
            {
                "candidate_index": int(result.get("candidate_index", 0)),
                "color": str(result.get("color", "#4c78a8")),
                "normalized_poly": normalized_poly,
                "triangles": result.get("filtered_triangles", _empty_triangles()),
            }
        )
        filter_stats = dict(result.get("filter_stats", {}))
        candidate_summaries.append(
            {
                "candidate_index": int(result.get("candidate_index", 0)),
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

    raw_panel_geoms = _iter_polygons(target["row_geom"])
    selected_panel_geoms = [poly_raw]
    repaired_polys = list(triangulation_eval["repaired_polys"])
    candidate_polys = list(triangulation_eval["candidate_polys"])
    normalized_polys = list(triangulation_eval["normalized_polys"])
    candidate_colors = [summary["color"] for summary in raw_triangle_groups[: len(candidate_polys)]]

    info_payload = {
        "selection_mode": target["selection_mode"],
        "row_index": target["row_index"],
        "part_index": target["part_index"],
        "requested_part_index": target["requested_part_index"],
        "part_count": target["part_count"],
        "from_multipolygon": target["from_multipolygon"],
        "input_path": str(input_path),
        "timeout_sec": float(args.timeout),
        "triangulation_status": str(triangulation_eval["triangulation_status"]),
        "failure_stage": triangulation_eval["failure_stage"],
        "status_note": str(triangulation_eval["status_note"]),
        "risk_level": str(risk.get("risk_level")),
        "risk_score": int(risk.get("risk_score", 0)),
        "triangulation_path": str(risk.get("triangulation_path")),
        "reason_tags": list(diag.get("reason_tags", [])),
        "raw_is_valid": bool(diag.get("raw_is_valid")),
        "raw_hole_count": int(diag.get("raw_hole_count", 0)),
        "raw_shell_vertex_count": int(diag.get("raw_shell_vertex_count", 0)),
        "candidate_count": int(diag.get("candidate_count", 0)),
        "max_triangle_vertices": int(diag.get("max_triangle_vertices", 0)),
        "max_triangle_segments": int(diag.get("max_triangle_segments", 0)),
        "max_triangle_holes": int(diag.get("max_triangle_holes", 0)),
        "min_normalized_edge_length": diag.get("min_normalized_edge_length"),
        "risk_notes": list(risk.get("risk_notes", [])),
        "triangulation_candidates": candidate_summaries,
    }

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    _plot_geometry_panel(axes[0, 0], raw_panel_geoms, title="Raw Row Geometry", annotate_ids=False)
    _plot_geometry_panel(axes[0, 1], selected_panel_geoms, title="Selected Part", annotate_ids=False)
    _plot_geometry_panel(axes[0, 2], repaired_polys, title="buffer(0) Repaired", annotate_ids=True)
    _plot_geometry_panel(axes[0, 3], candidate_polys, title="Prepared Candidates", annotate_ids=True, colors=candidate_colors)

    _plot_normalized_panel(axes[1, 0], normalized_polys, title="Normalized Candidates", colors=candidate_colors)
    _plot_triangle_panel(
        axes[1, 1],
        raw_triangle_groups,
        title="Raw Triangulation",
        status_text=str(triangulation_eval["status_note"]),
    )
    _plot_triangle_panel(
        axes[1, 2],
        filtered_triangle_groups,
        title="Filtered Triangulation",
        status_text=str(triangulation_eval["status_note"]),
    )
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.0,
        1.0,
        _format_multiline_json(info_payload),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        wrap=True,
    )

    fig.suptitle(f"Polygon Triangulation Visualization: {file_stem}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = output_dir / f"{file_stem}.png"
    json_path = output_dir / f"{file_stem}.json"
    fig.savefig(png_path, dpi=max(72, int(args.dpi)), bbox_inches="tight")
    plt.close(fig)

    json_path.write_text(json.dumps(info_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Saved visualization to {png_path}")
    print(f"[INFO] Saved metadata to {json_path}")


if __name__ == "__main__":
    main()
