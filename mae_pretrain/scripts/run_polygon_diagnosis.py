"""Polygon native-triangulation diagnosis script.

This script scans polygon samples from one or more shapefile directories and
diagnoses which samples are most likely to destabilize the current native
triangulation pipeline used by ``build_dataset_triangle.py``.

Static mode is the default and does not call ``triangle.triangulate(...)``.
Instead, it reuses the current repair, normalization, and triangle-input
construction logic to attach interpretable topology/complexity tags and a
heuristic risk level to each sample. For very targeted slow-path validation,
probe mode can still run isolated real triangulation one sample at a time.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import math
import multiprocessing as mp
import os
import signal
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely import wkb as shapely_wkb
from shapely.geometry import Polygon
from tqdm import tqdm

try:
    from shapely.validation import explain_validity
except Exception:  # pragma: no cover - compatibility fallback
    explain_validity = None

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


_BUILD_MODULE = None
_TRIANGLE_COMPLEXITY_THRESHOLD = 2048
_TINY_NORMALIZED_EDGE_THRESHOLD = 1e-6
_NODE_COUNT_BUCKET_LABELS = ("<3", "3~1E3", "1E3~1E4", "1E4~1E5", ">1E5")
_MIN_EDGE_BUCKET_LABELS = (">=1E-2", "1E-3~1E-2", "1E-4~1E-3", "1E-5~1E-4", "<1E-5")
_CONNECTIVITY_BUCKET_LABELS = ("simple", "multi", "donut", "porous", "complex")


def _inject_repo_root() -> Path:
    """Inject repository root into `sys.path` for direct script execution."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def _load_build_module():
    """Load the triangulation helper module lazily."""
    global _BUILD_MODULE
    if _BUILD_MODULE is None:
        if __package__ in {None, ""}:
            _BUILD_MODULE = importlib.import_module("mae_pretrain.src.datasets.build_dataset_triangle")
        else:
            from ..src.datasets import build_dataset_triangle as build_module

            _BUILD_MODULE = build_module
    return _BUILD_MODULE


def _resolve_single_shp(input_dir: Path) -> Path:
    """Resolve exactly one `.shp` file from the given directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Each --input_dirs item must be a directory that contains one .shp file: {input_dir}")

    shp_files = sorted(path for path in input_dir.glob("*.shp") if path.is_file())
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found under input directory: {input_dir}")
    if len(shp_files) > 1:
        shp_list = ", ".join(path.name for path in shp_files[:10])
        raise ValueError(
            f"Expected exactly one .shp file under {input_dir}, but found {len(shp_files)}: {shp_list}"
        )
    return shp_files[0]


def _resolve_input_tasks(input_dirs: list[str]) -> list[tuple[Path, Path]]:
    """Resolve one shapefile task from each provided input directory."""
    tasks: list[tuple[Path, Path]] = []
    for raw_dir in input_dirs:
        input_dir = Path(raw_dir)
        input_path = _resolve_single_shp(input_dir)
        tasks.append((input_dir, input_path))
    return tasks


def _build_task_output_names(tasks: list[tuple[Path, Path]]) -> list[str]:
    """Create stable output subdirectory names for multiple input tasks."""
    stem_counts: dict[str, int] = {}
    for _, input_path in tasks:
        stem = input_path.stem
        stem_counts[stem] = stem_counts.get(stem, 0) + 1

    assigned: list[str] = []
    used_names: set[str] = set()
    for task_index, (input_dir, input_path) in enumerate(tasks):
        stem = input_path.stem
        if stem_counts.get(stem, 0) <= 1:
            candidate = stem
        else:
            candidate = f"{input_dir.name}__{stem}"
        if candidate in used_names:
            candidate = f"{candidate}_{task_index + 1:02d}"
        used_names.add(candidate)
        assigned.append(candidate)
    return assigned


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


def _signal_name_from_exitcode(exitcode: int | None) -> str | None:
    """Convert negative multiprocessing exitcode into signal name."""
    if exitcode is None or exitcode >= 0:
        return None
    try:
        return signal.Signals(-int(exitcode)).name
    except Exception:
        return None


def _init_bucket_counts(labels: tuple[str, ...]) -> dict[str, int]:
    """Create an ordered zero-initialized bucket counter."""
    return {label: 0 for label in labels}


def _bucket_node_count(node_count: int) -> str:
    """Bucket one sample by triangle-input node count.

    The metric is `max_triangle_vertices`, i.e. the maximum triangle-input
    vertex count among repaired candidate polygons of the current sample.
    """
    value = int(node_count)
    if value < 3:
        return "<3"
    if value < 1_000:
        return "3~1E3"
    if value < 10_000:
        return "1E3~1E4"
    if value <= 100_000:
        return "1E4~1E5"
    return ">1E5"


def _bucket_min_edge(min_edge: float) -> str:
    """Bucket one sample by normalized minimum edge length."""
    value = float(min_edge)
    if value >= 1e-2:
        return ">=1E-2"
    if value >= 1e-3:
        return "1E-3~1E-2"
    if value >= 1e-4:
        return "1E-4~1E-3"
    if value >= 1e-5:
        return "1E-5~1E-4"
    return "<1E-5"


def _bucket_connectivity(polygon_type: str) -> str:
    """Map polygon type into the 5 connectivity buckets used by charts."""
    if polygon_type in {"simple", "multi", "donut", "porous"}:
        return polygon_type
    return "complex"


def _draw_pie_panel(
    ax,
    title: str,
    counts: dict[str, int],
    subtitle: str = "",
) -> None:
    """Draw one compact pie chart panel with a legend placed below the pie."""
    labels = list(counts.keys())
    values = [int(counts[label]) for label in labels]
    total = int(sum(values))
    if total <= 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=15, fontweight="bold")
        ax.axis("off")
    else:
        def _autopct(pct: float) -> str:
            count = int(round(pct * total / 100.0))
            if pct < 1.0 or count <= 0:
                return ""
            return f"{pct:.1f}%\n{count}"

        wedges, _, _ = ax.pie(
            values,
            autopct=_autopct,
            startangle=90,
            counterclock=False,
            radius=1.08,
            pctdistance=0.72,
            wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
            textprops={"fontsize": 12, "fontweight": "bold"},
        )
        ax.axis("equal")

        legend_labels = [
            f"{label}: N={value}, {(value / total) * 100.0:.1f}%"
            for label, value in zip(labels, values)
        ]
        ax.legend(
            wedges,
            legend_labels,
            title="Legend",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.045),
            frameon=False,
            ncol=1,
            fontsize=11.2,
            title_fontsize=12.2,
            labelspacing=0.3,
            handlelength=1.2,
            handletextpad=0.6,
        )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=8)
    if subtitle:
        ax.text(0.5, 0.015, subtitle, ha="center", va="top", fontsize=11.5, transform=ax.transAxes)

def _write_combined_pie_chart(
    output_path: Path,
    panels: list[dict[str, Any]],
) -> None:
    """Render a single PNG containing three horizontally-arranged pie charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(16.8, 5.1))
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        _draw_pie_panel(
            ax=ax,
            title=str(panel["title"]),
            counts=dict(panel["counts"]),
            subtitle=str(panel.get("subtitle", "")),
        )

    fig.subplots_adjust(left=0.02, right=0.995, top=0.88, bottom=0.11, wspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _classify_polygon_type(geom) -> str:
    """Classify one source geometry into diagnosis connectivity categories."""
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
        any_holes = any(int(len(getattr(poly, "interiors", []))) > 0 for poly in parts if poly is not None and not poly.is_empty)
        return "complex" if any_holes else "multi"
    return "complex"


def _ring_quality(build_module, ring_coords: Any) -> tuple[float | None, bool]:
    """Measure minimum edge length and whether duplicate vertices were cleaned."""
    arr = np.asarray(ring_coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None, False

    raw_vertex_count = int(arr.shape[0])
    if np.allclose(arr[0, :2], arr[-1, :2], atol=1e-12, rtol=0.0):
        raw_vertex_count -= 1

    cleaned = build_module._clean_ring_coords(arr)
    if cleaned is None or cleaned.shape[0] < 3:
        return None, True

    had_near_duplicate = int(cleaned.shape[0]) < raw_vertex_count
    closed = np.concatenate([cleaned, cleaned[:1]], axis=0)
    diffs = closed[1:] - closed[:-1]
    edge_lengths = np.linalg.norm(diffs, axis=1)
    if edge_lengths.size == 0:
        return None, had_near_duplicate
    return float(edge_lengths.min()), had_near_duplicate


def _scan_polygon_edge_quality(build_module, poly: Polygon) -> tuple[float | None, bool]:
    """Inspect shell and holes for tiny edges or cleaned duplicate vertices."""
    min_edge: float | None = None
    had_near_duplicate = False

    rings = [np.asarray(poly.exterior.coords)]
    rings.extend(np.asarray(interior.coords) for interior in poly.interiors)
    for ring in rings:
        ring_min_edge, ring_dup = _ring_quality(build_module, ring)
        if ring_min_edge is not None:
            min_edge = ring_min_edge if min_edge is None else min(min_edge, ring_min_edge)
        had_near_duplicate = had_near_duplicate or ring_dup

    return min_edge, had_near_duplicate


def _static_diagnose_polygon(build_module, poly_raw: Polygon) -> dict[str, Any]:
    """Collect lightweight risk tags and metrics without running triangle."""
    tags: set[str] = set()
    diag: dict[str, Any] = {
        "raw_geom_type": str(poly_raw.geom_type),
        "raw_is_valid": bool(poly_raw.is_valid),
        "validity_reason": "",
        "raw_hole_count": int(len(poly_raw.interiors)),
        "raw_shell_vertex_count": int(max(0, len(poly_raw.exterior.coords) - 1)),
        "bounds": [float(v) for v in poly_raw.bounds],
        "buffer_geom_type": "",
        "candidate_count": 0,
        "repair_changed_topology": False,
        "would_use_isolated_triangle": False,
        "would_call_triangle_directly_in_worker": False,
        "max_triangle_vertices": 0,
        "max_triangle_segments": 0,
        "max_triangle_holes": 0,
        "min_normalized_edge_length": None,
        "static_analysis_error": "",
    }

    if not diag["raw_is_valid"]:
        tags.add("invalid_geometry")
        if explain_validity is not None:
            try:
                diag["validity_reason"] = str(explain_validity(poly_raw))
            except Exception:
                diag["validity_reason"] = ""

    has_holes = diag["raw_hole_count"] > 0
    if has_holes:
        tags.add("has_holes")

    shell_hole_touching = False
    if has_holes:
        try:
            shell_hole_touching = bool(build_module._polygon_has_shell_hole_intersection(poly_raw))
        except Exception:
            shell_hole_touching = False
    diag["shell_hole_touching"] = shell_hole_touching
    if shell_hole_touching:
        tags.add("shell_hole_touching")

    try:
        poly_fixed = poly_raw.buffer(0)
        if poly_fixed.is_empty:
            diag["buffer_geom_type"] = "EMPTY"
            tags.add("buffer0_empty")
            diag["reason_tags"] = sorted(tags)
            return diag

        diag["buffer_geom_type"] = str(poly_fixed.geom_type)
        if poly_fixed.geom_type != "Polygon":
            tags.add("buffer0_non_polygon")
            diag["reason_tags"] = sorted(tags)
            return diag

        candidate_polys = build_module._prepare_polygon_candidates(poly_fixed)
        diag["candidate_count"] = int(len(candidate_polys))
        if len(candidate_polys) > 1:
            diag["repair_changed_topology"] = True
            tags.add("repair_changed_topology")
            tags.add("candidate_count_gt_1")
        elif len(candidate_polys) == 1:
            diag["repair_changed_topology"] = bytes(candidate_polys[0].wkb) != bytes(poly_fixed.wkb)
            if diag["repair_changed_topology"]:
                tags.add("repair_changed_topology")

        min_edge_length: float | None = None
        for candidate_poly in candidate_polys:
            poly_norm = build_module._normalize_polygon_to_unit_box(candidate_poly)
            if poly_norm is None:
                tags.add("normalize_failed")
                continue

            if len(poly_norm.interiors) > 0 and build_module._polygon_has_shell_hole_intersection(poly_norm):
                diag["would_use_isolated_triangle"] = True
            else:
                diag["would_call_triangle_directly_in_worker"] = True

            tri_input = build_module._build_triangle_input(poly_norm)
            if tri_input is None:
                tags.add("triangle_input_failed")
                continue

            tri_vertices = int(np.asarray(tri_input["vertices"]).shape[0])
            tri_segments = int(np.asarray(tri_input["segments"]).shape[0])
            tri_holes = int(np.asarray(tri_input.get("holes", np.zeros((0, 2), dtype=np.float64))).shape[0])

            diag["max_triangle_vertices"] = max(diag["max_triangle_vertices"], tri_vertices)
            diag["max_triangle_segments"] = max(diag["max_triangle_segments"], tri_segments)
            diag["max_triangle_holes"] = max(diag["max_triangle_holes"], tri_holes)

            if tri_vertices >= _TRIANGLE_COMPLEXITY_THRESHOLD:
                tags.add("triangle_vertices_too_many")
            if tri_segments >= _TRIANGLE_COMPLEXITY_THRESHOLD:
                tags.add("triangle_segments_too_many")

            candidate_min_edge, has_near_duplicate = _scan_polygon_edge_quality(build_module, poly_norm)
            if candidate_min_edge is not None:
                min_edge_length = (
                    candidate_min_edge if min_edge_length is None else min(min_edge_length, candidate_min_edge)
                )
            if has_near_duplicate or (
                candidate_min_edge is not None and candidate_min_edge < _TINY_NORMALIZED_EDGE_THRESHOLD
            ):
                tags.add("tiny_edges_or_near_duplicate_vertices")

        diag["min_normalized_edge_length"] = min_edge_length
    except Exception as exc:
        diag["static_analysis_error"] = f"{type(exc).__name__}: {exc}"

    diag["reason_tags"] = sorted(tags)
    return diag


def _classify_triangulation_path(diag: dict[str, Any]) -> str:
    """Classify how current build code would reach native triangulation."""
    use_isolated = bool(diag.get("would_use_isolated_triangle", False))
    use_direct = bool(diag.get("would_call_triangle_directly_in_worker", False))
    if use_direct and use_isolated:
        return "mixed"
    if use_direct:
        return "direct_worker"
    if use_isolated:
        return "isolated_subprocess"
    return "not_reachable"


def _static_assess_risk(diag: dict[str, Any]) -> dict[str, Any]:
    """Estimate native-crash risk from static topology and path signals.

    Scoring philosophy
    ------------------
    This function does *not* predict an exact native-crash probability. Instead,
    it produces a practical triage score for debugging: samples that are more
    likely to be worth manual inspection, forced isolation, or defensive
    fallback should receive a higher score.

    The score is intentionally heuristic and is built from two signal groups:

    1. Execution-path risk
       We first ask how the current build pipeline would reach native triangle:

       - ``direct_worker``:
         ``triangle.triangulate(...)`` would run directly inside the outer chunk
         worker process. If the native library crashes here, it can break the
         whole process pool. This is why it contributes a positive base score.

       - ``isolated_subprocess``:
         the sample would only reach triangle through an extra isolated child
         process. A native crash still loses that sample, but its blast radius
         is much smaller, so this path gets no base penalty.

       - ``mixed``:
         different repaired candidates from the same source sample could take
         different paths. This is treated as more dangerous than pure direct
         execution because the sample is both topologically unstable and not
         uniformly protected.

       - ``not_reachable``:
         static preprocessing never reached a valid triangle-input stage. In
         that case we currently classify the sample as ``low`` risk with
         respect to *native* triangle crash, because triangle itself would not
         be entered on this path.

    2. Geometry/topology fragility
       We then accumulate signals that historically correlate with unstable
       triangulation inputs:

       - large triangle-input vertex count / segment count
       - tiny normalized edges or near-duplicate vertices
       - raw invalid geometry
       - holes
       - shell-hole touching topology
       - topology-changing repair
       - candidate splitting
       - normalization / triangle-input construction failure
       - internal exceptions during static preprocessing

    Current score table
    -------------------
    Base path contribution:
    - ``mixed``: +2
    - ``direct_worker``: +1
    - ``isolated_subprocess``: +0
    - ``not_reachable``: short-circuit to ``low``

    Additional signals:
    - ``static_analysis_error``: +3
    - ``triangle_vertices_too_many``: +2
    - ``triangle_segments_too_many``: +2
    - ``tiny_edges_or_near_duplicate_vertices``: +2
    - ``repair_changed_topology``: +1
    - ``candidate_count_gt_1``: +1
    - ``invalid_geometry``: +1
    - ``has_holes``: +1
    - ``shell_hole_touching``: +1
    - ``triangle_input_failed`` or ``normalize_failed``: +1

    Risk-level thresholds
    ---------------------
    - ``score >= 6`` -> ``critical``
    - ``score >= 4`` -> ``high``
    - ``score >= 2`` -> ``medium``
    - otherwise -> ``low``

    Interpretation
    --------------
    - ``low`` means "few static warnings" rather than "guaranteed safe".
    - ``medium`` means the sample is non-trivial and worth keeping in the risk
      report, commonly because it has holes or required repair.
    - ``high`` means the sample combines direct-worker execution with large or
      fragile triangle input and should be prioritized for inspection.
    - ``critical`` means the strongest warning set currently observed in static
      mode; these samples are the best candidates for forced isolation,
      skipping, or dedicated reproduction.
    """
    tags = set(str(tag) for tag in diag.get("reason_tags", []))
    path_kind = _classify_triangulation_path(diag)
    score = 0
    notes: list[str] = []

    if path_kind == "mixed":
        score += 2
        notes.append("contains direct-worker and isolated triangulation paths")
    elif path_kind == "direct_worker":
        score += 1
        notes.append("would call triangle directly inside chunk worker")
    elif path_kind == "isolated_subprocess":
        score += 0
        notes.append("would reach isolated triangulation subprocess only")
    else:
        notes.append("triangle path is not reachable after static preprocessing")

    if diag.get("static_analysis_error"):
        score += 3
        notes.append("static preprocessing raised an internal exception")
    if "triangle_vertices_too_many" in tags:
        score += 2
        notes.append("triangle input has a large vertex count")
    if "triangle_segments_too_many" in tags:
        score += 2
        notes.append("triangle input has a large segment count")
    if "tiny_edges_or_near_duplicate_vertices" in tags:
        score += 2
        notes.append("normalized polygon contains tiny edges or near-duplicate vertices")
    if "repair_changed_topology" in tags:
        score += 1
        notes.append("buffer/make_valid changed polygon topology before triangulation")
    if "candidate_count_gt_1" in tags:
        score += 1
        notes.append("one source polygon expands into multiple triangulation candidates")
    if "invalid_geometry" in tags:
        score += 1
        notes.append("raw geometry is invalid before repair")
    if "has_holes" in tags:
        score += 1
        notes.append("polygon contains holes")
    if "shell_hole_touching" in tags:
        score += 1
        notes.append("shell-hole touching topology is a known triangulation instability")
    if "triangle_input_failed" in tags or "normalize_failed" in tags:
        score += 1
        notes.append("pre-triangulation normalization/input building is fragile")

    if path_kind == "not_reachable":
        risk_level = "low"
    elif score >= 6:
        risk_level = "critical"
    elif score >= 4:
        risk_level = "high"
    elif score >= 2:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "analysis_mode": "static",
        "triangulation_path": path_kind,
        "risk_score": int(score),
        "risk_level": risk_level,
        "risk_notes": notes,
    }


def _probe_polygon_pipeline(build_module, poly_raw: Polygon) -> dict[str, Any]:
    """Run the current triangulation path for one polygon sample."""
    poly_fixed = poly_raw.buffer(0)
    if poly_fixed.is_empty:
        return {
            "probe_status": "bad_input",
            "probe_stage": "buffer0_empty",
            "triangle_count": 0,
        }
    if poly_fixed.geom_type != "Polygon":
        return {
            "probe_status": "bad_input",
            "probe_stage": "buffer0_non_polygon",
            "triangle_count": 0,
            "buffer_geom_type": str(poly_fixed.geom_type),
        }

    candidate_polys = build_module._prepare_polygon_candidates(poly_fixed)
    if not candidate_polys:
        return {
            "probe_status": "bad_input",
            "probe_stage": "prepare_candidates_empty",
            "triangle_count": 0,
        }

    candidate_count = int(len(candidate_polys))
    attempted_candidate_count = 0
    total_triangle_count = 0
    for candidate_poly in candidate_polys:
        poly_norm = build_module._normalize_polygon_to_unit_box(candidate_poly)
        if poly_norm is None:
            continue
        attempted_candidate_count += 1
        tris = build_module._triangulate_polygon_with_holes(poly_norm)
        if isinstance(tris, np.ndarray) and tris.ndim == 3 and tris.shape[1:] == (3, 2):
            total_triangle_count += int(tris.shape[0])

    if total_triangle_count > 0:
        return {
            "probe_status": "ok",
            "probe_stage": "triangulated",
            "triangle_count": int(total_triangle_count),
            "candidate_count": candidate_count,
            "attempted_candidate_count": int(attempted_candidate_count),
        }

    if attempted_candidate_count == 0:
        return {
            "probe_status": "bad_input",
            "probe_stage": "normalize_failed",
            "triangle_count": 0,
            "candidate_count": candidate_count,
            "attempted_candidate_count": 0,
        }

    return {
        "probe_status": "bad_result",
        "probe_stage": "empty_triangle_result",
        "triangle_count": 0,
        "candidate_count": candidate_count,
        "attempted_candidate_count": int(attempted_candidate_count),
    }


def _probe_subprocess_entry(conn, poly_wkb: bytes) -> None:
    """Child entry that runs the probe and reports back through a pipe."""
    try:
        build_module = _load_build_module()
        poly_raw = shapely_wkb.loads(poly_wkb)
        payload = _probe_polygon_pipeline(build_module, poly_raw)
        try:
            conn.send({"ok": True, "payload": payload})
        except BrokenPipeError:
            pass
    except Exception as exc:
        try:
            conn.send(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        except BrokenPipeError:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _run_probe(poly_wkb: bytes, timeout_sec: float) -> dict[str, Any]:
    """Probe one polygon sample in an isolated subprocess."""
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_probe_subprocess_entry,
        args=(send_conn, poly_wkb),
    )
    proc.start()
    send_conn.close()

    payload: dict[str, Any] | None = None
    timed_out = False
    try:
        if recv_conn.poll(timeout=max(0.1, float(timeout_sec))):
            payload = recv_conn.recv()
        else:
            timed_out = True
    except EOFError:
        payload = None
    except Exception:
        payload = None
    finally:
        try:
            recv_conn.close()
        except Exception:
            pass

        if timed_out and proc.is_alive():
            proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)

    exitcode = proc.exitcode
    signal_name = _signal_name_from_exitcode(exitcode)

    if timed_out:
        return {
            "probe_status": "timeout",
            "probe_stage": "probe_timeout",
            "child_exitcode": exitcode,
            "child_signal": signal_name,
        }

    if not payload:
        return {
            "probe_status": "child_crash",
            "probe_stage": "child_exit",
            "child_exitcode": exitcode,
            "child_signal": signal_name,
        }

    if bool(payload.get("ok", False)):
        probe_payload = dict(payload.get("payload") or {})
        probe_payload["child_exitcode"] = exitcode
        probe_payload["child_signal"] = signal_name
        return probe_payload

    return {
        "probe_status": "python_exception",
        "probe_stage": "probe_exception",
        "error_type": str(payload.get("error_type", "")),
        "error_message": str(payload.get("error_message", "")),
        "child_exitcode": exitcode,
        "child_signal": signal_name,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a `.jsonl` file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")


def _resolve_num_workers(num_workers: int, unit_count: int) -> int:
    """Resolve an effective worker count for chunk-parallel diagnosis."""
    if int(num_workers) <= 0:
        resolved = max(1, (os.cpu_count() or 1) - 1)
    else:
        resolved = max(1, int(num_workers))
    if unit_count <= 0:
        return resolved
    return max(1, min(resolved, int(unit_count)))


def _build_static_chunks(
    row_payloads: list[tuple[int, int, bytes | None]],
    rows_per_chunk: int,
    num_workers: int,
) -> tuple[list[list[tuple[int, int, bytes | None]]], int]:
    """Split one shapefile workload into enough row chunks for real parallelism.

    `rows_per_chunk` is treated as the preferred maximum chunk size, but when it
    would otherwise collapse the whole shapefile into too few chunks, we still
    split the workload so that multiple workers can be used on the same `.shp`.
    """
    row_count = int(len(row_payloads))
    if row_count <= 0:
        return [], 1

    preferred_chunk_size = max(1, int(rows_per_chunk))
    requested_workers = _resolve_num_workers(num_workers=num_workers, unit_count=row_count)

    chunk_count_by_size = max(1, math.ceil(row_count / preferred_chunk_size))
    target_chunk_count = max(chunk_count_by_size, min(requested_workers, row_count))
    target_chunk_count = max(1, min(target_chunk_count, row_count))

    base_chunk_size, remainder = divmod(row_count, target_chunk_count)
    chunks: list[list[tuple[int, int, bytes | None]]] = []
    start = 0
    for chunk_index in range(target_chunk_count):
        current_size = base_chunk_size + (1 if chunk_index < remainder else 0)
        end = start + current_size
        chunks.append(row_payloads[start:end])
        start = end

    effective_workers = _resolve_num_workers(num_workers=num_workers, unit_count=len(chunks))
    return chunks, effective_workers


def _serialize_geometry(geom) -> bytes | None:
    """Serialize one shapely geometry into WKB for worker transfer."""
    if geom is None or getattr(geom, "is_empty", True):
        return None
    return bytes(geom.wkb)


def _deserialize_geometry(geom_wkb: bytes | None):
    """Deserialize one WKB payload back into a shapely geometry."""
    if geom_wkb is None:
        return None
    return shapely_wkb.loads(geom_wkb)


def _init_diagnosis_state() -> dict[str, Any]:
    """Create an empty diagnosis-aggregation state."""
    return {
        "status_counts": {},
        "reason_tag_counts": {},
        "row_polygon_type_counts": _init_bucket_counts(_CONNECTIVITY_BUCKET_LABELS),
        "risk_row_polygon_type_counts": _init_bucket_counts(_CONNECTIVITY_BUCKET_LABELS),
        "node_count_bucket_counts": _init_bucket_counts(_NODE_COUNT_BUCKET_LABELS),
        "min_edge_bucket_counts": _init_bucket_counts(_MIN_EDGE_BUCKET_LABELS),
        "triangulation_path_counts": {},
        "scanned_sample_count": 0,
        "scanned_polygon_row_count": 0,
        "skipped_non_polygon_row_count": 0,
        "focus_sample_count": 0,
        "min_edge_missing_sample_count": 0,
        "risk_records": [],
    }


def _merge_counter_dict(target: dict[str, int], source: dict[str, int]) -> None:
    """In-place sum two flat integer dictionaries."""
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def _merge_diagnosis_state(target: dict[str, Any], partial: dict[str, Any]) -> None:
    """Merge one partial diagnosis state into the aggregate state."""
    for key in (
        "status_counts",
        "reason_tag_counts",
        "row_polygon_type_counts",
        "risk_row_polygon_type_counts",
        "node_count_bucket_counts",
        "min_edge_bucket_counts",
        "triangulation_path_counts",
    ):
        _merge_counter_dict(target[key], partial.get(key, {}))

    for key in (
        "scanned_sample_count",
        "scanned_polygon_row_count",
        "skipped_non_polygon_row_count",
        "focus_sample_count",
        "min_edge_missing_sample_count",
    ):
        target[key] += int(partial.get(key, 0))

    target["risk_records"].extend(list(partial.get("risk_records", [])))


def _diagnose_one_row(
    build_module,
    input_dir: str,
    input_path: str,
    row_idx: int,
    sample_index_base: int,
    geom,
    mode: str,
    timeout_sec: float,
) -> dict[str, Any]:
    """Diagnose one source row and return row-local counters and records."""
    row_state = _init_diagnosis_state()

    polygon_parts = build_module._expand_geometry_to_polygons(geom)
    row_polygon_type = _classify_polygon_type(geom)
    if not polygon_parts:
        row_state["skipped_non_polygon_row_count"] += 1
        return row_state

    row_state["scanned_polygon_row_count"] += 1
    connectivity_bucket = _bucket_connectivity(row_polygon_type)
    row_state["row_polygon_type_counts"][connectivity_bucket] += 1

    row_has_risk_sample = False
    count_key = "probe_status" if mode == "probe" else "risk_level"
    for part_idx, (poly_raw, from_multi) in enumerate(polygon_parts):
        sample_index = int(sample_index_base + part_idx)
        row_state["scanned_sample_count"] += 1

        static_diag = _static_diagnose_polygon(build_module, poly_raw)
        node_bucket = _bucket_node_count(int(static_diag.get("max_triangle_vertices", 0) or 0))
        row_state["node_count_bucket_counts"][node_bucket] += 1
        min_edge_value = static_diag.get("min_normalized_edge_length")
        if min_edge_value is None:
            row_state["min_edge_missing_sample_count"] += 1
        else:
            min_edge_bucket = _bucket_min_edge(float(min_edge_value))
            row_state["min_edge_bucket_counts"][min_edge_bucket] += 1

        if mode == "probe":
            analysis_result = _run_probe(bytes(poly_raw.wkb), timeout_sec=timeout_sec)
        else:
            analysis_result = _static_assess_risk(static_diag)

        record = {
            "input_dir": str(input_dir),
            "input_path": str(input_path),
            "row_idx": int(row_idx),
            "part_idx": int(part_idx),
            "sample_index": sample_index,
            "polygon_type": row_polygon_type,
            "from_multipolygon": bool(from_multi),
            **static_diag,
            **analysis_result,
        }

        triangulation_path = str(record.get("triangulation_path", _classify_triangulation_path(static_diag)))
        row_state["triangulation_path_counts"][triangulation_path] = (
            row_state["triangulation_path_counts"].get(triangulation_path, 0) + 1
        )

        status_value = str(record.get(count_key, "unknown"))
        row_state["status_counts"][status_value] = row_state["status_counts"].get(status_value, 0) + 1

        if mode == "probe":
            should_write_problem = status_value != "ok"
            should_write_focus = status_value in {"child_crash", "timeout"}
        else:
            should_write_problem = status_value in {"medium", "high", "critical"}
            should_write_focus = status_value in {"high", "critical"}

        if should_write_problem:
            row_has_risk_sample = True
            for tag in record.get("reason_tags", []):
                row_state["reason_tag_counts"][str(tag)] = row_state["reason_tag_counts"].get(str(tag), 0) + 1
            row_state["risk_records"].append(record)

        if should_write_focus:
            row_state["focus_sample_count"] += 1

    if row_has_risk_sample:
        row_state["risk_row_polygon_type_counts"][connectivity_bucket] += 1
    return row_state


def _diagnose_static_chunk_worker(
    chunk_index: int,
    row_payloads: list[tuple[int, int, bytes | None]],
    input_dir: str,
    input_path: str,
) -> dict[str, Any]:
    """Static-diagnosis worker for one row chunk."""
    try:
        build_module = _load_build_module()
        chunk_state = _init_diagnosis_state()
        for row_idx, sample_index_base, geom_wkb in row_payloads:
            geom = _deserialize_geometry(geom_wkb)
            row_state = _diagnose_one_row(
                build_module=build_module,
                input_dir=input_dir,
                input_path=input_path,
                row_idx=int(row_idx),
                sample_index_base=int(sample_index_base),
                geom=geom,
                mode="static",
                timeout_sec=0.0,
            )
            _merge_diagnosis_state(chunk_state, row_state)
        return {
            "ok": True,
            "chunk_index": int(chunk_index),
            "row_count": int(len(row_payloads)),
            **chunk_state,
        }
    except Exception as exc:
        return {
            "ok": False,
            "chunk_index": int(chunk_index),
            "row_count": int(len(row_payloads)),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _build_summary_field_descriptions(mode: str) -> dict[str, Any]:
    """Create human-readable field descriptions for `summary.json`."""
    status_key = "risk_level" if mode == "static" else "probe_status"
    status_descriptions = (
        {
            "low": "Static mode: low-risk sample with no prominent topology/complexity warnings.",
            "medium": "Static mode: sample has moderate warning tags, commonly holes or repaired topology.",
            "high": "Static mode: sample combines direct-worker triangle path with large/fragile triangle input.",
            "critical": "Static mode: strongest warning set; most worth manual inspection or forced isolation.",
        }
        if mode == "static"
        else {
            "ok": "Probe mode: isolated child finished triangulation successfully.",
            "bad_input": "Probe mode: preprocessing failed before a valid triangle input could be built.",
            "bad_result": "Probe mode: triangle path returned no usable triangles.",
            "python_exception": "Probe mode: child raised a Python exception and reported it back.",
            "child_crash": "Probe mode: child exited abruptly with no payload, often indicating native crash.",
            "timeout": "Probe mode: child exceeded timeout and was terminated.",
        }
    )
    return {
        "row_idx": "Original GeoDataFrame row index inside the source .shp file. One row may expand to multiple samples when the geometry is a MultiPolygon.",
        "sample_index": "Expanded polygon-sample index after splitting each MultiPolygon row into multiple polygon parts. This is the index used by diagnosis records.",
        "polygon_type": "Source-feature connectivity type. simple=single Polygon with no holes; multi=MultiPolygon and all parts have no holes; donut=single Polygon with exactly one hole; porous=single Polygon with more than one hole; complex=all remaining polygon complexities, mainly MultiPolygon with holes.",
        "input_dirs": "All input directories passed into the diagnosis script. Each directory must contain exactly one .shp dataset.",
        "input_dir": "Source directory of the current summary/output shard. This is one element from input_dirs.",
        "input_path": "Resolved .shp file path actually read by the diagnosis script.",
        "output_dir": "Directory where diagnosis outputs are written.",
        "analysis_mode": "Diagnosis mode. 'static' means no real triangulation call; 'probe' means isolated per-sample triangulation.",
        "num_workers": "Requested diagnosis worker count. In static mode, >1 enables chunk-parallel row processing; in probe mode the script falls back to sequential diagnosis.",
        "effective_num_workers": "Actual worker count used for the current task. In static mode this is min(resolved workers, chunk count); in probe mode it is always 1.",
        "rows_per_chunk": "Row count per diagnosis chunk when static-mode parallel processing is enabled.",
        "total_rows": "Total geometry row count read from the source .shp.",
        "row_start": "Inclusive start row index used in this diagnosis run.",
        "row_end": "Exclusive end row index used in this diagnosis run.",
        "scanned_polygon_row_count": "Number of scanned rows that contained Polygon or MultiPolygon geometry.",
        "skipped_non_polygon_row_count": "Number of scanned rows that contained no Polygon/MultiPolygon geometry.",
        "scanned_sample_count": "Total number of expanded polygon samples scanned in this run.",
        "focus_sample_count": "Count of most important samples. In static mode this means high/critical risk; in probe mode this means child_crash/timeout.",
        "status_counts": f"Counts grouped by {status_key}. See status_value_descriptions for meanings.",
        "status_value_descriptions": status_descriptions,
        "row_polygon_type_counts": "Counts of polygon_type over scanned polygon source rows.",
        "risk_row_polygon_type_counts": "Counts of polygon_type over polygon source rows that produced at least one record in risk_samples.jsonl.",
        "connectivity_bucket_counts": "Counts of 5 connectivity categories over scanned polygon source rows. This field is the numeric source for the connectivity pie chart.",
        "node_count_bucket_counts": "Counts of scanned samples grouped by max_triangle_vertices buckets: <3, 3~1E3, 1E3~1E4, 1E4~1E5, >1E5.",
        "min_edge_bucket_counts": "Counts of scanned samples grouped by min_normalized_edge_length buckets: >=1E-2, 1E-3~1E-2, 1E-4~1E-3, 1E-5~1E-4, <1E-5.",
        "min_edge_missing_sample_count": "Number of scanned samples whose min_normalized_edge_length could not be computed during static preprocessing.",
        "polygon_type_descriptions": {
            "simple": "Single Polygon with no holes.",
            "multi": "MultiPolygon and no part has holes.",
            "donut": "Single Polygon with exactly one hole.",
            "porous": "Single Polygon with more than one hole.",
            "complex": "All remaining polygon complexities, mainly MultiPolygon with holes.",
        },
        "triangulation_path_counts": "Counts grouped by how current build_dataset_triangle.py would reach triangle: direct_worker, isolated_subprocess, mixed, or not_reachable.",
        "triangulation_path_descriptions": {
            "direct_worker": "Would call triangle directly inside the chunk worker process. A native crash here can break the whole process pool.",
            "isolated_subprocess": "Would only call triangle inside an extra isolated subprocess. Native crash radius is limited to one sample attempt.",
            "mixed": "Different prepared candidates from the same sample could take both direct and isolated paths.",
            "not_reachable": "Static preprocessing never reached a valid triangle-input stage.",
        },
        "reason_tag_counts": "Counts of reason_tags among records written to the problem/risk JSONL file.",
        "risk_samples_jsonl": "Path to the JSONL file containing all non-trivial samples: medium/high/critical in static mode, or probe_status != ok in probe mode.",
        "pie_chart_path": "Path to the combined PNG that contains the node-count, min-edge, and connectivity pie charts in one row.",
    }


def _run_diagnosis_for_task(
    build_module,
    input_dir: Path,
    input_path: Path,
    input_dirs: list[str],
    output_dir: Path,
    mode: str,
    timeout_sec: float,
    row_start_arg: int,
    row_end_arg: int,
    num_workers: int,
    rows_per_chunk: int,
) -> None:
    """Run diagnosis for one shapefile task and write outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(input_path)
    total_rows = int(len(gdf))
    row_start = max(0, int(row_start_arg))
    row_end = total_rows if int(row_end_arg) < 0 else min(total_rows, int(row_end_arg))
    if row_start >= row_end:
        raise ValueError(f"Empty scan range: row_start={row_start}, row_end={row_end}, total_rows={total_rows}")

    risk_jsonl = output_dir / "risk_samples.jsonl"
    summary_path = output_dir / "summary.json"
    pie_overview_path = output_dir / "pie_overview.png"

    for stale_path in (
        risk_jsonl,
        output_dir / "high_risk_samples.jsonl",
        output_dir / "native_crash_samples.jsonl",
        output_dir / "problem_samples.jsonl",
        output_dir / "focus_samples.jsonl",
        pie_overview_path,
        output_dir / "pie_node_count.png",
        output_dir / "pie_min_edge.png",
        output_dir / "pie_connectivity.png",
    ):
        if stale_path.exists():
            stale_path.unlink()

    row_payloads: list[tuple[int, int, bytes | None]] = []
    sample_index_cursor = 0
    for row_idx, geom in enumerate(gdf.geometry):
        part_count = int(build_module._count_geometry_samples(geom))
        if row_start <= row_idx < row_end:
            row_payloads.append((int(row_idx), int(sample_index_cursor), _serialize_geometry(geom)))
        sample_index_cursor += part_count

    diag_state = _init_diagnosis_state()
    count_key = "probe_status" if mode == "probe" else "risk_level"
    effective_workers = 1
    if mode == "static":
        chunks, effective_workers = _build_static_chunks(
            row_payloads=row_payloads,
            rows_per_chunk=int(rows_per_chunk),
            num_workers=int(num_workers),
        )
    else:
        chunks = []
        if int(num_workers) > 1:
            print("[WARN] Probe mode currently runs sequentially; --num_workers is ignored.")

    def _consume_partial(partial: dict[str, Any]) -> None:
        risk_records = list(partial.get("risk_records", []))
        partial_for_merge = dict(partial)
        partial_for_merge["risk_records"] = []
        _merge_diagnosis_state(diag_state, partial_for_merge)
        for record in risk_records:
            _append_jsonl(risk_jsonl, record)

    progress_desc = f"Diagnosing rows ({input_path.stem})"
    with tqdm(total=row_end - row_start, desc=progress_desc, unit="row") as pbar:
        if mode == "static" and effective_workers > 1 and chunks:
            mp_context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=effective_workers, mp_context=mp_context) as executor:
                future_to_meta = {
                    executor.submit(
                        _diagnose_static_chunk_worker,
                        chunk_index,
                        chunk_payload,
                        str(input_dir),
                        str(input_path),
                    ): (chunk_index, int(len(chunk_payload)))
                    for chunk_index, chunk_payload in enumerate(chunks)
                }
                pending_results: dict[int, dict[str, Any]] = {}
                next_chunk_index = 0
                for future in as_completed(future_to_meta):
                    chunk_index, row_count = future_to_meta[future]
                    try:
                        chunk_result = future.result()
                    except Exception as exc:
                        chunk_result = {
                            "ok": False,
                            "chunk_index": int(chunk_index),
                            "row_count": int(row_count),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    pbar.update(int(chunk_result.get("row_count", row_count)))
                    pending_results[int(chunk_result.get("chunk_index", chunk_index))] = chunk_result
                    while next_chunk_index in pending_results:
                        ordered_chunk = pending_results.pop(next_chunk_index)
                        if not bool(ordered_chunk.get("ok", False)):
                            raise RuntimeError(
                                f"Static diagnosis chunk failed in {input_path}: {ordered_chunk.get('error', 'unknown error')}"
                            )
                        _consume_partial(ordered_chunk)
                        next_chunk_index += 1
        else:
            for row_idx, sample_index_base, geom_wkb in row_payloads:
                geom = _deserialize_geometry(geom_wkb)
                row_state = _diagnose_one_row(
                    build_module=build_module,
                    input_dir=str(input_dir),
                    input_path=str(input_path),
                    row_idx=int(row_idx),
                    sample_index_base=int(sample_index_base),
                    geom=geom,
                    mode=str(mode),
                    timeout_sec=float(timeout_sec),
                )
                _consume_partial(row_state)
                pbar.update(1)

    _write_combined_pie_chart(
        pie_overview_path,
        panels=[
            {
                "title": "Node Count Pie",
                "counts": diag_state["node_count_bucket_counts"],
                "subtitle": "Metric: max_triangle_vertices per sample",
            },
            {
                "title": "Minimum Edge Pie",
                "counts": diag_state["min_edge_bucket_counts"],
                "subtitle": (
                    "Metric: min_normalized_edge_length per sample"
                    if int(diag_state["min_edge_missing_sample_count"]) == 0
                    else (
                        "Metric: min_normalized_edge_length per sample; "
                        f"missing={int(diag_state['min_edge_missing_sample_count'])}"
                    )
                ),
            },
            {
                "title": "Connectivity Pie",
                "counts": diag_state["row_polygon_type_counts"],
                "subtitle": "Metric: source polygon rows",
            },
        ],
    )

    summary = {
        "field_descriptions": _build_summary_field_descriptions(str(mode)),
        "input_dirs": [str(Path(p)) for p in input_dirs],
        "input_dir": str(input_dir),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "analysis_mode": str(mode),
        "num_workers": int(num_workers),
        "effective_num_workers": int(effective_workers),
        "rows_per_chunk": int(rows_per_chunk),
        "total_rows": total_rows,
        "row_start": row_start,
        "row_end": row_end,
        "scanned_polygon_row_count": int(diag_state["scanned_polygon_row_count"]),
        "skipped_non_polygon_row_count": int(diag_state["skipped_non_polygon_row_count"]),
        "scanned_sample_count": int(diag_state["scanned_sample_count"]),
        "focus_sample_count": int(diag_state["focus_sample_count"]),
        "status_counts": dict(sorted(diag_state["status_counts"].items())),
        "row_polygon_type_counts": dict(diag_state["row_polygon_type_counts"]),
        "risk_row_polygon_type_counts": dict(diag_state["risk_row_polygon_type_counts"]),
        "connectivity_bucket_counts": dict(diag_state["row_polygon_type_counts"]),
        "node_count_bucket_counts": dict(diag_state["node_count_bucket_counts"]),
        "min_edge_bucket_counts": dict(diag_state["min_edge_bucket_counts"]),
        "min_edge_missing_sample_count": int(diag_state["min_edge_missing_sample_count"]),
        "triangulation_path_counts": dict(sorted(diag_state["triangulation_path_counts"].items())),
        "reason_tag_counts": dict(sorted(diag_state["reason_tag_counts"].items())),
        "risk_samples_jsonl": str(risk_jsonl),
        "pie_chart_path": str(pie_overview_path),
    }
    summary_path.write_text(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Diagnosis finished for {input_path}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Scanned polygon samples: {diag_state['scanned_sample_count']}")
    if mode == "probe":
        print(f"[INFO] Native-crash-like samples (child_crash/timeout): {diag_state['focus_sample_count']}")
    else:
        print(f"[INFO] High-risk samples (high/critical): {diag_state['focus_sample_count']}")
    print(f"[INFO] Effective workers: {effective_workers}")
    print(f"[INFO] {count_key} counts: {json.dumps(diag_state['status_counts'], ensure_ascii=False)}")
    print(f"[INFO] Summary file: {summary_path}")


def main() -> None:
    """CLI main function for polygon diagnosis."""
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()
    build_module = _load_build_module()

    parser = argparse.ArgumentParser(
        description="Diagnose polygon samples that can crash native triangulation."
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="One or more directories, each containing exactly one .shp dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "outputs" / "polygon_diagnosis"),
        help="Directory for summary and JSONL outputs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="static",
        choices=["static", "probe"],
        help="Diagnosis mode. 'static' does not run triangle; 'probe' runs isolated per-sample triangulation.",
    )
    parser.add_argument(
        "--timeout_sec",
        type=float,
        default=20.0,
        help="Per-sample probe timeout in seconds. Only used when --mode=probe.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Static-mode diagnosis worker count. <=0 means auto.",
    )
    parser.add_argument(
        "--rows_per_chunk",
        type=int,
        default=2000,
        help="Row count per static-diagnosis chunk.",
    )
    parser.add_argument(
        "--row_start",
        type=int,
        default=0,
        help="Inclusive start row index for scanning. Applied to each input file.",
    )
    parser.add_argument(
        "--row_end",
        type=int,
        default=-1,
        help="Exclusive end row index. <0 means scan to the end. Applied to each input file.",
    )
    args = parser.parse_args()

    tasks = _resolve_input_tasks(list(args.input_dirs))
    task_output_names = _build_task_output_names(tasks)
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Diagnosis mode        : {args.mode}")
    print(f"[INFO] Input task count      : {len(tasks)}")
    if str(args.mode) == "static":
        print(f"[INFO] Static num_workers    : {args.num_workers}")
        print(f"[INFO] Static rows_per_chunk : {args.rows_per_chunk}")

    for task_index, (input_dir, input_path) in enumerate(tasks, start=1):
        output_name = task_output_names[task_index - 1]
        output_dir = base_output_dir / output_name
        print(f"[INFO] Diagnosis task {task_index}/{len(tasks)}: {input_path}")
        _run_diagnosis_for_task(
            build_module=build_module,
            input_dir=input_dir,
            input_path=input_path,
            input_dirs=list(args.input_dirs),
            output_dir=output_dir,
            mode=str(args.mode),
            timeout_sec=float(args.timeout_sec),
            row_start_arg=int(args.row_start),
            row_end_arg=int(args.row_end),
            num_workers=int(args.num_workers),
            rows_per_chunk=int(args.rows_per_chunk),
        )


if __name__ == "__main__":
    main()
