"""Dataset build launcher script.

This script scans vector files, triangulates polygons, and saves processed
triangle tensors into `data/processed`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from runtime_bootstrap import ensure_cuda_runtime_libs
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def _inject_src_path() -> Path:
    """Inject local `src` directory into `sys.path`.

    Returns:
        Project root path.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return project_root


def main() -> None:
    """CLI main function for dataset building."""
    # ensure_cuda_runtime_libs()
    project_root = _inject_src_path()

    from datasets.build_dataset_triangle import process_and_save

    parser = argparse.ArgumentParser(description="Build triangulated polygon dataset")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="One or more raw vector directories.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(project_root / "data" / "processed" / "polygon_triangles_normalized.pt"),
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="shp",
        choices=["shp", "gdb", "geojs"],
        help="Input vector file type. One of: shp, gdb, geojs.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="all",
        help="Layer selector for gdb input. Use 'all' to read every layer.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="File-level process count. <=0 means auto.",
    )
    parser.add_argument(
        "--shard_size_mb",
        type=float,
        default=0.0,
        help="Target shard size in MB. <=0 means single .pt output.",
    )
    parser.add_argument(
        "--min_triangle_area",
        type=float,
        default=1e-8,
        help="Minimum triangle area in normalized [-1,1] space.",
    )
    parser.add_argument(
        "--min_triangle_height",
        type=float,
        default=1e-5,
        help="Minimum triangle altitude proxy in normalized [-1,1] space.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Save triangulation quality log JSON beside output .pt.",
    )
    args = parser.parse_args()
    if args.file_type != "gdb" and str(args.layer).strip().lower() != "all":
        print(f"[WARN] --layer is ignored when --file_type={args.file_type}.")

    process_and_save(
        input_dirs=args.input_dirs,
        output_path=args.output_path,
        file_type=args.file_type,
        layer=args.layer,
        num_workers=args.num_workers,
        shard_size_mb=args.shard_size_mb,
        min_triangle_area=args.min_triangle_area,
        min_triangle_height=args.min_triangle_height,
        log=args.log,
    )


if __name__ == "__main__":
    main()
