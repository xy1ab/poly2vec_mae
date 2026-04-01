"""Visualize one polygon row or one expanded sample from a shapefile."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygonPatch
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


def _add_polygon_patch(ax, poly: Polygon, facecolor: str, edgecolor: str, alpha: float = 0.55) -> None:
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


def _plot_geometry_panel(ax, geoms: list[Polygon], title: str, annotate_ids: bool = False) -> None:
    """Plot polygon list on one axis."""
    colors = [
        "#8ecae6",
        "#ffb703",
        "#bde0fe",
        "#fb8500",
        "#90be6d",
        "#cdb4db",
        "#ffafcc",
        "#a8dadc",
    ]
    edgecolor = "#1d3557"

    for idx, poly in enumerate(geoms):
        color = colors[idx % len(colors)]
        _add_polygon_patch(ax, poly, facecolor=color, edgecolor=edgecolor)
        if annotate_ids:
            rep = poly.representative_point()
            ax.text(float(rep.x), float(rep.y), str(idx), fontsize=8, color="#111827", ha="center", va="center")

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


def _plot_normalized_panel(ax, geoms: list[Polygon], title: str) -> None:
    """Plot normalized candidates and the reference [-1,1] box."""
    _plot_geometry_panel(ax, geoms, title=title, annotate_ids=True)
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


def _resolve_row_target(build_module, gdf: gpd.GeoDataFrame, row_idx: int) -> dict[str, Any]:
    """Resolve one source row for visualization."""
    if row_idx < 0 or row_idx >= len(gdf):
        raise IndexError(f"row_idx out of range: {row_idx} (total_rows={len(gdf)})")

    row_geom = gdf.geometry.iloc[row_idx]
    expanded_parts = build_module._expand_geometry_to_polygons(row_geom)
    if not expanded_parts:
        raise ValueError(f"row_idx={row_idx} does not contain a Polygon or MultiPolygon geometry")

    return {
        "selection_mode": "row_idx",
        "row_idx": int(row_idx),
        "sample_index": None,
        "part_idx": None,
        "row_geom": row_geom,
        "expanded_parts": [poly for poly, _ in expanded_parts],
        "selected_poly": None,
        "from_multipolygon": bool(getattr(row_geom, "geom_type", "") == "MultiPolygon"),
    }


def _resolve_sample_target(build_module, gdf: gpd.GeoDataFrame, sample_index: int) -> dict[str, Any]:
    """Resolve one expanded polygon sample by sample_index."""
    if sample_index < 0:
        raise ValueError(f"sample_index must be >= 0, got {sample_index}")

    cursor = 0
    for row_idx, geom in enumerate(gdf.geometry):
        expanded_parts = build_module._expand_geometry_to_polygons(geom)
        part_count = len(expanded_parts)
        if sample_index < cursor + part_count:
            part_idx = sample_index - cursor
            selected_poly, from_multi = expanded_parts[part_idx]
            return {
                "selection_mode": "sample_index",
                "row_idx": int(row_idx),
                "sample_index": int(sample_index),
                "part_idx": int(part_idx),
                "row_geom": geom,
                "expanded_parts": [poly for poly, _ in expanded_parts],
                "selected_poly": selected_poly,
                "from_multipolygon": bool(from_multi),
            }
        cursor += part_count

    raise IndexError(f"sample_index out of range: {sample_index}")


def main() -> None:
    """CLI main function."""
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()
    build_module = _load_build_module()
    diag_module = _load_diag_module()

    parser = argparse.ArgumentParser(description="Visualize one polygon row or one expanded polygon sample.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory that contains exactly one .shp dataset.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--row_idx", type=int, default=None, help="Original geometry row index in the source .shp.")
    group.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Expanded polygon-sample index after splitting MultiPolygon rows into multiple parts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "outputs" / "polygon_viz"),
        help="Directory for generated PNG and metadata JSON.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_path = _resolve_single_shp(input_dir)
    output_dir = Path(args.output_dir) / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(input_path)
    if args.row_idx is not None:
        target = _resolve_row_target(build_module, gdf, int(args.row_idx))
        file_stem = f"row_{int(args.row_idx):06d}"
        raw_panel_geoms = _iter_polygons(target["row_geom"])
        selected_panel_geoms = list(target["expanded_parts"])
        repaired_polys: list[Polygon] = []
        candidate_polys: list[Polygon] = []
        normalized_polys: list[Polygon] = []
        part_summaries: list[dict[str, Any]] = []
        for part_idx, poly_raw in enumerate(target["expanded_parts"]):
            diag = diag_module._static_diagnose_polygon(build_module, poly_raw)
            risk = diag_module._static_assess_risk(diag)
            repaired = poly_raw.buffer(0)
            repaired_polys.extend(_iter_polygons(repaired))
            candidates = build_module._prepare_polygon_candidates(repaired) if repaired.geom_type == "Polygon" else []
            candidate_polys.extend(candidates)
            normalized_polys.extend(
                [poly_norm for poly_norm in (build_module._normalize_polygon_to_unit_box(pg) for pg in candidates) if poly_norm]
            )
            part_summaries.append(
                {
                    "part_idx": int(part_idx),
                    "risk_level": str(risk.get("risk_level")),
                    "risk_score": int(risk.get("risk_score", 0)),
                    "reason_tags": list(diag.get("reason_tags", [])),
                    "max_triangle_vertices": int(diag.get("max_triangle_vertices", 0)),
                    "max_triangle_holes": int(diag.get("max_triangle_holes", 0)),
                }
            )

        info_payload = {
            "selection_mode": target["selection_mode"],
            "row_idx": target["row_idx"],
            "part_count": len(target["expanded_parts"]),
            "input_path": str(input_path),
            "note": "Use --sample_index for per-part diagnosis when this row is a MultiPolygon.",
            "parts": part_summaries,
        }
    else:
        target = _resolve_sample_target(build_module, gdf, int(args.sample_index))
        file_stem = f"sample_{int(args.sample_index):06d}"
        poly_raw = target["selected_poly"]
        assert poly_raw is not None
        raw_panel_geoms = _iter_polygons(target["row_geom"])
        selected_panel_geoms = [poly_raw]
        repaired = poly_raw.buffer(0)
        repaired_polys = _iter_polygons(repaired)
        candidate_polys = build_module._prepare_polygon_candidates(repaired) if repaired.geom_type == "Polygon" else []
        normalized_polys = [
            poly_norm for poly_norm in (build_module._normalize_polygon_to_unit_box(pg) for pg in candidate_polys) if poly_norm
        ]
        diag = diag_module._static_diagnose_polygon(build_module, poly_raw)
        risk = diag_module._static_assess_risk(diag)
        info_payload = {
            "selection_mode": target["selection_mode"],
            "row_idx": target["row_idx"],
            "sample_index": target["sample_index"],
            "part_idx": target["part_idx"],
            "from_multipolygon": target["from_multipolygon"],
            "input_path": str(input_path),
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
        }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _plot_geometry_panel(axes[0, 0], raw_panel_geoms, title="Raw Row Geometry", annotate_ids=False)
    _plot_geometry_panel(axes[0, 1], selected_panel_geoms, title="Expanded Part(s)", annotate_ids=True)
    _plot_geometry_panel(axes[0, 2], repaired_polys, title="buffer(0) Repaired", annotate_ids=True)
    _plot_geometry_panel(axes[1, 0], candidate_polys, title="Prepared Candidates", annotate_ids=True)
    _plot_normalized_panel(axes[1, 1], normalized_polys, title="Normalized Candidates")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        1.0,
        _format_multiline_json(info_payload),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        wrap=True,
    )

    fig.suptitle(f"Polygon Visualization: {file_stem}", fontsize=14)
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
