"""Dataset build launcher script.

This script scans vector files, triangulates polygons, and saves processed
triangle tensors into `data/processed`.

Design note:
This entrypoint intentionally avoids importing the `datasets` package namespace
to prevent unrelated deep-learning dependencies (for example, torch/cuda) from
being loaded during pure data preprocessing.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import sys
from pathlib import Path

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
    """Inject repository root into `sys.path` for direct script execution.

    Returns:
        `mae_pretrain` project root path.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def main() -> None:
    """CLI main function for dataset building."""
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()
    if __package__ in {None, ""}:
        build_module = importlib.import_module("mae_pretrain.src.datasets.build_dataset_triangle")
    else:
        from ..src.datasets import build_dataset_triangle as build_module
    process_and_save = build_module.process_and_save

    parser = argparse.ArgumentParser(description="Build triangulated polygon dataset")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="One or more raw vector directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "data" / "processed"),
        help="Output directory for generated .pt shards.",
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
        "--num_worker",
        dest="num_workers",
        type=int,
        default=0,
        help="Intra-task process count. <=0 means auto.",
    )
    parser.add_argument(
        "--rows_per_chunk",
        type=int,
        default=2000,
        help="Number of source rows in one intra-task chunk.",
    )
    parser.add_argument(
        "--progress_every_chunks",
        type=int,
        default=10,
        help="Print chunk merge summary every N chunks. <=0 disables it.",
    )
    parser.add_argument(
        "--shard_size_mb",
        type=float,
        default=0.0,
        help="Target shard size in MB. >0 enables incremental shard flush; <=0 means single .pt output buffered until finalize.",
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

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output_dir must be a directory path, got file: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_type = str(args.file_type).strip().lower()
    if file_type == "shp":
        shp_files: list[Path] = []
        for input_dir in args.input_dirs:
            shp_files.extend(Path(p) for p in glob.glob(str(Path(input_dir) / "**" / "*.shp"), recursive=True))
        shp_files = sorted(set(shp_files))
        if not shp_files:
            raise FileNotFoundError("No .shp files found under --input_dirs.")
        if len(shp_files) > 1:
            print(f"[WARN] Found {len(shp_files)} .shp files; output prefix uses first file name: {shp_files[0].name}")
        output_stem = f"{shp_files[0].stem}_tri"
    else:
        # Keep compatibility for gdb/geojs modes when output prefix cannot map to one shp name.
        output_stem = "polygon_triangles_normalized"

    output_path = output_dir / f"{output_stem}.pt"

    process_and_save(
        input_dirs=args.input_dirs,
        output_path=output_path,
        file_type=args.file_type,
        layer=args.layer,
        num_workers=args.num_workers,
        rows_per_chunk=args.rows_per_chunk,
        progress_every_chunks=args.progress_every_chunks,
        shard_size_mb=args.shard_size_mb,
        min_triangle_area=args.min_triangle_area,
        min_triangle_height=args.min_triangle_height,
        log=args.log,
    )


if __name__ == "__main__":
    main()
