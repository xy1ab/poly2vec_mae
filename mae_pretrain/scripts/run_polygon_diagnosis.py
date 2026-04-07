"""Row-level polygon diagnosis for the current dataset-build pipeline.

This script scans one or more shapefiles and diagnoses rows under the current
row-level triangulation flow used by `build_dataset_triangle.py`.

Key design choices:
- Output is strictly row-based, matching the new dataset semantics.
- `risk_samples.jsonl` records only dropped rows.
- The script depends only on `src/` helpers and does not import other scripts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely import wkb as shapely_wkb
from tqdm import tqdm


_BUILD_MODULE = None
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
    """Load build_dataset_triangle lazily from `src/` only."""
    global _BUILD_MODULE
    if _BUILD_MODULE is None:
        _BUILD_MODULE = importlib.import_module("mae_pretrain.src.datasets.build_dataset_triangle")
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


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a `.jsonl` file."""
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")


def _init_bucket_counts(labels: tuple[str, ...]) -> dict[str, int]:
    """Create an ordered zero-initialized bucket counter."""
    return {label: 0 for label in labels}


def _bucket_node_count(node_count: int) -> str:
    """Bucket one row by total kept-node count."""
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
    """Bucket one row by normalized minimum edge length."""
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


def _draw_pie_panel(ax, title: str, counts: dict[str, int], subtitle: str = "") -> None:
    """Draw one compact pie chart panel with legend below."""
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


def _write_combined_pie_chart(output_path: Path, panels: list[dict[str, Any]]) -> None:
    """Render a single PNG containing three horizontally arranged pie charts."""
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


def _init_diagnosis_state() -> dict[str, Any]:
    """Create an empty diagnosis-aggregation state."""
    return {
        "row_polygon_type_counts": _init_bucket_counts(_CONNECTIVITY_BUCKET_LABELS),
        "risk_row_polygon_type_counts": _init_bucket_counts(_CONNECTIVITY_BUCKET_LABELS),
        "node_count_bucket_counts": _init_bucket_counts(_NODE_COUNT_BUCKET_LABELS),
        "min_edge_bucket_counts": _init_bucket_counts(_MIN_EDGE_BUCKET_LABELS),
        "dropped_reason_counts": {},
        "scanned_polygon_row_count": 0,
        "skipped_non_polygon_row_count": 0,
        "triangulated_row_count": 0,
        "dropped_row_count": 0,
        "degenerated_row_count": 0,
        "isolated_row_count": 0,
        "min_edge_missing_row_count": 0,
        "risk_records": [],
    }


def _merge_counter_dict(target: dict[str, int], source: dict[str, int]) -> None:
    """In-place sum two flat integer dictionaries."""
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def _merge_diagnosis_state(target: dict[str, Any], partial: dict[str, Any]) -> None:
    """Merge one partial diagnosis state into the aggregate state."""
    for key in (
        "row_polygon_type_counts",
        "risk_row_polygon_type_counts",
        "node_count_bucket_counts",
        "min_edge_bucket_counts",
        "dropped_reason_counts",
    ):
        _merge_counter_dict(target[key], partial.get(key, {}))

    for key in (
        "scanned_polygon_row_count",
        "skipped_non_polygon_row_count",
        "triangulated_row_count",
        "dropped_row_count",
        "degenerated_row_count",
        "isolated_row_count",
        "min_edge_missing_row_count",
    ):
        target[key] += int(partial.get(key, 0))

    target["risk_records"].extend(list(partial.get("risk_records", [])))


def _collect_row_metrics(build_module, normalized_parts: list[Any], filtered_parts: list[Any]) -> tuple[int, float | None]:
    """Compute row-level node count and minimum edge metrics."""
    metric_parts = filtered_parts if filtered_parts else normalized_parts
    if not metric_parts:
        return 0, None

    node_count = int(sum(build_module._polygon_node_count(poly) for poly in metric_parts))
    min_edge_values = [
        float(build_module._polygon_min_edge(poly))
        for poly in metric_parts
        if np.isfinite(float(build_module._polygon_min_edge(poly)))
    ]
    min_edge = min(min_edge_values) if min_edge_values else None
    return node_count, min_edge


def _collect_part_diagnostics(build_module, normalized_parts: list[Any]) -> tuple[list[Any], list[dict[str, Any]], int]:
    """Filter normalized row parts and collect per-part diagnostic flags."""
    kept_parts: list[Any] = []
    part_records: list[dict[str, Any]] = []
    filtered_count = 0

    for part_index, poly in enumerate(normalized_parts, start=1):
        is_valid = bool(poly.is_valid)
        shell_hole_touching = bool(build_module._polygon_has_shell_hole_intersection(poly))
        filtered = (not is_valid) or shell_hole_touching
        if filtered:
            filtered_count += 1
        else:
            kept_parts.append(poly)

        part_records.append(
            {
                "part_index": int(part_index),
                "is_valid": bool(is_valid),
                "shell_hole_touching": bool(shell_hole_touching),
                "filtered": bool(filtered),
                "hole_count": int(len(poly.interiors)),
                "node_count": int(build_module._polygon_node_count(poly)),
                "min_edge": float(build_module._polygon_min_edge(poly)),
            }
        )

    return kept_parts, part_records, int(filtered_count)


def _build_drop_record(
    input_dir: str,
    input_path: str,
    row_idx: int,
    polygon_type: str,
    raw_part_count: int,
    normalized_part_count: int,
    filtered_part_count: int,
    kept_part_count: int,
    node_count: int,
    min_edge: float | None,
    isolate_row: bool,
    dropped_reason: str,
    row_result: dict[str, Any] | None,
    part_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one dropped-row record for risk_samples.jsonl."""
    filtered_triangle_count = int((row_result or {}).get("filtered_triangle_count", 0))
    return {
        "input_dir": str(input_dir),
        "input_path": str(input_path),
        "row_idx": int(row_idx),
        "polygon_type": str(polygon_type),
        "raw_part_count": int(raw_part_count),
        "normalized_part_count": int(normalized_part_count),
        "filtered_part_count": int(filtered_part_count),
        "kept_part_count": int(kept_part_count),
        "row_node_count": int(node_count),
        "row_min_normalized_edge": min_edge,
        "isolated": bool(isolate_row),
        "dropped_reason": str(dropped_reason),
        "filtered_triangle_count": int(filtered_triangle_count),
        "part_records": part_records,
    }


def _diagnose_one_row(
    build_module,
    input_dir: str,
    input_path: str,
    row_idx: int,
    geom,
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
    """Diagnose one source row under the current row-level build flow."""
    row_state = _init_diagnosis_state()

    if geom is None or geom.is_empty or geom.geom_type not in {"Polygon", "MultiPolygon"}:
        row_state["skipped_non_polygon_row_count"] += 1
        return row_state

    row_state["scanned_polygon_row_count"] += 1
    polygon_type = _classify_polygon_type(geom)
    row_state["row_polygon_type_counts"][polygon_type] += 1

    raw_parts = build_module._expand_geometry_to_polygons(geom)
    raw_part_count = int(len(raw_parts))
    normalized_parts = build_module._normalize_row_parts(geom=geom, norm_max=float(norm_max))
    normalized_part_count = int(len(normalized_parts))

    kept_parts, part_records, filtered_part_count = _collect_part_diagnostics(build_module, normalized_parts)
    kept_part_count = int(len(kept_parts))

    row_node_count, row_min_edge = _collect_row_metrics(build_module, normalized_parts, kept_parts)
    row_state["node_count_bucket_counts"][_bucket_node_count(row_node_count)] += 1
    if row_min_edge is None:
        row_state["min_edge_missing_row_count"] += 1
    else:
        row_state["min_edge_bucket_counts"][_bucket_min_edge(float(row_min_edge))] += 1

    isolate_row = False
    row_result: dict[str, Any] | None = None
    dropped_reason: str | None = None

    if normalized_part_count == 0:
        dropped_reason = "normalize_row_failed"
    elif kept_part_count == 0:
        dropped_reason = "all_parts_filtered"
    else:
        isolate_row = bool(
            build_module._should_isolate_row(
                safe_mode=str(safe_mode),
                filtered_parts=kept_parts,
                part_safe=int(part_safe),
                node_safe=int(node_safe),
                hole_safe=int(hole_safe),
                edge_safe=float(edge_safe),
            )
        )
        if isolate_row:
            row_state["isolated_row_count"] += 1
            row_result = build_module._process_row_isolated(
                normalized_parts=normalized_parts,
                min_triangle_area=float(min_triangle_area),
                min_triangle_height=float(min_triangle_height),
                timeout_safe=float(timeout_safe),
            )
        else:
            row_result = build_module._process_normalized_row_parts(
                normalized_parts=normalized_parts,
                min_triangle_area=float(min_triangle_area),
                min_triangle_height=float(min_triangle_height),
            )

        if not bool(row_result.get("ok", False)):
            dropped_reason = str(row_result.get("failure_reason", "row_processing_failed"))

    if dropped_reason is not None:
        row_state["dropped_row_count"] += 1
        row_state["risk_row_polygon_type_counts"][polygon_type] += 1
        row_state["dropped_reason_counts"][dropped_reason] = row_state["dropped_reason_counts"].get(dropped_reason, 0) + 1
        row_state["risk_records"].append(
            _build_drop_record(
                input_dir=input_dir,
                input_path=input_path,
                row_idx=int(row_idx),
                polygon_type=polygon_type,
                raw_part_count=raw_part_count,
                normalized_part_count=normalized_part_count,
                filtered_part_count=filtered_part_count,
                kept_part_count=kept_part_count,
                node_count=row_node_count,
                min_edge=row_min_edge,
                isolate_row=bool(isolate_row),
                dropped_reason=dropped_reason,
                row_result=row_result,
                part_records=part_records,
            )
        )
        return row_state

    row_state["triangulated_row_count"] += 1
    had_part_filter = bool((row_result or {}).get("had_part_filter", False))
    filtered_triangle_count = int((row_result or {}).get("filtered_triangle_count", 0))
    if had_part_filter or filtered_triangle_count > 0:
        row_state["degenerated_row_count"] += 1
    return row_state


def _diagnose_static_chunk_worker(
    chunk_index: int,
    row_payloads: list[tuple[int, bytes | None]],
    input_dir: str,
    input_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Static-diagnosis worker for one row chunk."""
    try:
        build_module = _load_build_module()
        chunk_state = _init_diagnosis_state()
        for row_idx, geom_wkb in row_payloads:
            geom = _deserialize_geometry(geom_wkb)
            row_state = _diagnose_one_row(
                build_module=build_module,
                input_dir=input_dir,
                input_path=input_path,
                row_idx=int(row_idx),
                geom=geom,
                safe_mode=str(config["safe_mode"]),
                part_safe=int(config["part_safe"]),
                node_safe=int(config["node_safe"]),
                hole_safe=int(config["hole_safe"]),
                edge_safe=float(config["edge_safe"]),
                timeout_safe=float(config["timeout_safe"]),
                norm_max=float(config["norm_max"]),
                min_triangle_area=float(config["min_triangle_area"]),
                min_triangle_height=float(config["min_triangle_height"]),
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


def _resolve_num_workers(num_workers: int, unit_count: int) -> int:
    """Resolve an effective worker count for chunk-parallel diagnosis."""
    if int(num_workers) <= 0:
        resolved = max(1, (os.cpu_count() or 1) - 1)
    else:
        resolved = max(1, int(num_workers))
    if unit_count <= 0:
        return resolved
    return max(1, min(resolved, int(unit_count)))


def _build_row_chunks(
    row_payloads: list[tuple[int, bytes | None]],
    rows_per_chunk: int,
    num_workers: int,
) -> tuple[list[list[tuple[int, bytes | None]]], int]:
    """Split one shapefile workload into enough row chunks for true parallelism."""
    row_count = int(len(row_payloads))
    if row_count <= 0:
        return [], 1

    preferred_chunk_size = max(1, int(rows_per_chunk))
    requested_workers = _resolve_num_workers(num_workers=num_workers, unit_count=row_count)

    chunk_count_by_size = max(1, math.ceil(row_count / preferred_chunk_size))
    target_chunk_count = max(chunk_count_by_size, min(requested_workers, row_count))
    target_chunk_count = max(1, min(target_chunk_count, row_count))

    base_chunk_size, remainder = divmod(row_count, target_chunk_count)
    chunks: list[list[tuple[int, bytes | None]]] = []
    start = 0
    for chunk_index in range(target_chunk_count):
        current_size = base_chunk_size + (1 if chunk_index < remainder else 0)
        end = start + current_size
        chunks.append(row_payloads[start:end])
        start = end

    effective_workers = _resolve_num_workers(num_workers=num_workers, unit_count=len(chunks))
    return chunks, effective_workers


def _build_summary_field_descriptions() -> dict[str, Any]:
    """Create human-readable field descriptions for `summary.json`."""
    return {
        "input_dirs": "All input directories passed to this diagnosis run. Each directory must contain exactly one .shp file.",
        "input_dir": "Source directory of the current summary/output subfolder.",
        "input_path": "Resolved .shp file path actually read by the diagnosis script.",
        "output_dir": "Directory where summary, risk JSONL, and pie PNG are written.",
        "safe_mode": "Isolation trigger mode copied from the new row-level dataset builder: all, risky, or off.",
        "part_safe": "Enter row isolation when kept part count is strictly greater than this threshold.",
        "node_safe": "Enter row isolation when any kept part node count is strictly greater than this threshold.",
        "hole_safe": "Enter row isolation when any kept part hole count is strictly greater than this threshold.",
        "edge_safe": "Enter row isolation when any kept part minimum normalized edge is strictly smaller than this threshold.",
        "timeout_safe": "Maximum seconds allowed for one isolated row before it is treated as dropped.",
        "norm_max": "Maximum absolute coordinate used by row-level normalization.",
        "min_triangle_area": "Minimum triangle area used by degenerate-triangle filtering.",
        "min_triangle_height": "Minimum triangle height proxy used by degenerate-triangle filtering.",
        "total_rows": "Total geometry row count read from the source .shp.",
        "row_start": "Inclusive start row index used in this diagnosis run.",
        "row_end": "Exclusive end row index used in this diagnosis run.",
        "scanned_polygon_row_count": "Number of scanned rows whose geometry type is Polygon or MultiPolygon.",
        "skipped_non_polygon_row_count": "Number of scanned rows skipped because they were empty or not Polygon/MultiPolygon.",
        "triangulated_rows": "Rows that would successfully produce one merged training sample under the current build flow.",
        "dropped_rows": "Rows that would be dropped under the current build flow. These rows are the only records written to risk_samples.jsonl.",
        "degenerated_rows": "Rows that still triangulate successfully but experienced part filtering or triangle filtering. This is a subset of triangulated_rows.",
        "isolated_rows": "Rows that would enter safe subprocess mode under the current thresholds.",
        "row_polygon_type_counts": "Connectivity-category counts over scanned polygon rows.",
        "risk_row_polygon_type_counts": "Connectivity-category counts over dropped rows only.",
        "connectivity_bucket_counts": "Numeric source of the connectivity pie chart; same counts as row_polygon_type_counts.",
        "node_count_bucket_counts": "Counts of scanned polygon rows grouped by row_node_count buckets.",
        "min_edge_bucket_counts": "Counts of scanned polygon rows grouped by row_min_normalized_edge buckets.",
        "min_edge_missing_row_count": "Rows whose normalized minimum edge could not be computed.",
        "dropped_reason_counts": "Counts of drop reasons observed under the current build flow.",
        "risk_samples_jsonl": "Path to the JSONL file containing dropped rows only.",
        "pie_chart_path": "Path to the combined PNG containing the node-count, minimum-edge, and connectivity pie charts.",
        "polygon_type_descriptions": {
            "simple": "Single Polygon with no holes.",
            "multi": "MultiPolygon and no part has holes.",
            "donut": "Single Polygon with exactly one hole.",
            "porous": "Single Polygon with more than one hole.",
            "complex": "All remaining polygon complexities, mainly MultiPolygon with holes.",
        },
    }


def _run_diagnosis_for_task(
    build_module,
    input_dir: Path,
    input_path: Path,
    input_dirs: list[str],
    output_dir: Path,
    *,
    row_start_arg: int,
    row_end_arg: int,
    num_workers: int,
    rows_per_chunk: int,
    safe_mode: str,
    part_safe: int,
    node_safe: int,
    hole_safe: int,
    edge_safe: float,
    timeout_safe: float,
    norm_max: float,
    min_triangle_area: float,
    min_triangle_height: float,
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

    for stale_path in (risk_jsonl, summary_path, pie_overview_path):
        if stale_path.exists():
            stale_path.unlink()
    risk_jsonl.touch()

    row_payloads: list[tuple[int, bytes | None]] = []
    for row_idx, geom in enumerate(gdf.geometry):
        if row_start <= row_idx < row_end:
            row_payloads.append((int(row_idx), _serialize_geometry(geom)))

    config = {
        "safe_mode": str(safe_mode),
        "part_safe": int(part_safe),
        "node_safe": int(node_safe),
        "hole_safe": int(hole_safe),
        "edge_safe": float(edge_safe),
        "timeout_safe": float(timeout_safe),
        "norm_max": float(norm_max),
        "min_triangle_area": float(min_triangle_area),
        "min_triangle_height": float(min_triangle_height),
    }

    chunks, effective_workers = _build_row_chunks(
        row_payloads=row_payloads,
        rows_per_chunk=int(rows_per_chunk),
        num_workers=int(num_workers),
    )

    diag_state = _init_diagnosis_state()

    def _consume_partial(partial: dict[str, Any]) -> None:
        risk_records = list(partial.get("risk_records", []))
        partial_for_merge = dict(partial)
        partial_for_merge["risk_records"] = []
        _merge_diagnosis_state(diag_state, partial_for_merge)
        for record in risk_records:
            _append_jsonl(risk_jsonl, record)

    progress_desc = f"Diagnosing rows ({input_path.stem})"
    with tqdm(total=row_end - row_start, desc=progress_desc, unit="row") as pbar:
        if effective_workers > 1 and chunks:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                future_to_meta = {
                    executor.submit(
                        _diagnose_static_chunk_worker,
                        chunk_index,
                        chunk_payload,
                        str(input_dir),
                        str(input_path),
                        config,
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
                                f"Diagnosis chunk failed in {input_path}: {ordered_chunk.get('error', 'unknown error')}"
                            )
                        _consume_partial(ordered_chunk)
                        next_chunk_index += 1
        else:
            for row_idx, geom_wkb in row_payloads:
                geom = _deserialize_geometry(geom_wkb)
                row_state = _diagnose_one_row(
                    build_module=build_module,
                    input_dir=str(input_dir),
                    input_path=str(input_path),
                    row_idx=int(row_idx),
                    geom=geom,
                    **config,
                )
                _consume_partial(row_state)
                pbar.update(1)

    _write_combined_pie_chart(
        pie_overview_path,
        panels=[
            {
                "title": "Node Count Pie",
                "counts": diag_state["node_count_bucket_counts"],
                "subtitle": "Metric: row_node_count",
            },
            {
                "title": "Minimum Edge Pie",
                "counts": diag_state["min_edge_bucket_counts"],
                "subtitle": (
                    "Metric: row_min_normalized_edge"
                    if int(diag_state["min_edge_missing_row_count"]) == 0
                    else (
                        "Metric: row_min_normalized_edge; "
                        f"missing={int(diag_state['min_edge_missing_row_count'])}"
                    )
                ),
            },
            {
                "title": "Connectivity Pie",
                "counts": diag_state["row_polygon_type_counts"],
                "subtitle": "Metric: polygon rows",
            },
        ],
    )

    summary = {
        "field_descriptions": _build_summary_field_descriptions(),
        "input_dirs": [str(Path(p)) for p in input_dirs],
        "input_dir": str(input_dir),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "num_workers": int(num_workers),
        "effective_num_workers": int(effective_workers),
        "rows_per_chunk": int(rows_per_chunk),
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
        "row_start": int(row_start),
        "row_end": int(row_end),
        "scanned_polygon_row_count": int(diag_state["scanned_polygon_row_count"]),
        "skipped_non_polygon_row_count": int(diag_state["skipped_non_polygon_row_count"]),
        "triangulated_rows": int(diag_state["triangulated_row_count"]),
        "dropped_rows": int(diag_state["dropped_row_count"]),
        "degenerated_rows": int(diag_state["degenerated_row_count"]),
        "isolated_rows": int(diag_state["isolated_row_count"]),
        "row_polygon_type_counts": dict(diag_state["row_polygon_type_counts"]),
        "risk_row_polygon_type_counts": dict(diag_state["risk_row_polygon_type_counts"]),
        "connectivity_bucket_counts": dict(diag_state["row_polygon_type_counts"]),
        "node_count_bucket_counts": dict(diag_state["node_count_bucket_counts"]),
        "min_edge_bucket_counts": dict(diag_state["min_edge_bucket_counts"]),
        "min_edge_missing_row_count": int(diag_state["min_edge_missing_row_count"]),
        "dropped_reason_counts": dict(sorted(diag_state["dropped_reason_counts"].items())),
        "risk_samples_jsonl": str(risk_jsonl),
        "pie_chart_path": str(pie_overview_path),
    }
    summary_path.write_text(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] Diagnosis finished for {input_path}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Triangulated rows: {diag_state['triangulated_row_count']}")
    print(f"[INFO] Dropped rows: {diag_state['dropped_row_count']}")
    print(f"[INFO] Degenerated rows: {diag_state['degenerated_row_count']}")
    print(f"[INFO] Effective workers: {effective_workers}")
    print(f"[INFO] Summary file: {summary_path}")


def main() -> None:
    """CLI main function for polygon diagnosis."""
    project_root = _inject_repo_root()
    build_module = _load_build_module()

    parser = argparse.ArgumentParser(
        description="Diagnose dropped rows under the current row-level triangulation pipeline."
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
        help="Directory for summary, risk JSONL, and pie outputs.",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Diagnosis worker count. <=0 means auto.")
    parser.add_argument("--rows_per_chunk", type=int, default=2000, help="Row count per diagnosis chunk.")
    parser.add_argument("--row_start", type=int, default=0, help="Inclusive start row index. Applied to each input file.")
    parser.add_argument("--row_end", type=int, default=-1, help="Exclusive end row index. <0 means scan to the end.")
    parser.add_argument("--safe_mode", type=str, default="risky", choices=["all", "risky", "off"])
    parser.add_argument("--part_safe", type=int, default=1)
    parser.add_argument("--node_safe", type=int, default=2048)
    parser.add_argument("--hole_safe", type=int, default=1)
    parser.add_argument("--edge_safe", type=float, default=1e-5)
    parser.add_argument("--timeout_safe", type=float, default=20.0)
    parser.add_argument("--norm_max", type=float, default=1.0)
    parser.add_argument("--min_triangle_area", type=float, default=1e-8)
    parser.add_argument("--min_triangle_height", type=float, default=1e-5)
    args = parser.parse_args()

    tasks = _resolve_input_tasks(list(args.input_dirs))
    task_output_names = _build_task_output_names(tasks)
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input task count      : {len(tasks)}")
    print(f"[INFO] Diagnosis workers    : {args.num_workers}")
    print(f"[INFO] Diagnosis chunk size : {args.rows_per_chunk}")
    print(
        "[INFO] Safe mode / thresholds : "
        f"{args.safe_mode} | part>{args.part_safe}, node>{args.node_safe}, "
        f"hole>{args.hole_safe}, edge<{args.edge_safe:.3e}, timeout={args.timeout_safe:.1f}s"
    )
    print(f"[INFO] Row normalization max  : {args.norm_max:.3f}")

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
            row_start_arg=int(args.row_start),
            row_end_arg=int(args.row_end),
            num_workers=int(args.num_workers),
            rows_per_chunk=int(args.rows_per_chunk),
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


if __name__ == "__main__":
    main()
