"""Add sequential gid field to shapefiles that do not already have one."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd


def _inject_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def _discover_shapefiles(input_dir: Path) -> list[Path]:
    return sorted(path.resolve() for path in input_dir.rglob("*.shp") if path.is_file())


def _replace_with_temp_outputs(original_shp_path: Path, temp_shp_path: Path) -> None:
    original_stem = original_shp_path.stem
    temp_stem = temp_shp_path.stem

    for temp_file in temp_shp_path.parent.glob(f"{temp_stem}.*"):
        target_path = original_shp_path.with_name(f"{original_stem}{temp_file.suffix}")
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(temp_file), str(target_path))


def _process_one_shapefile(shp_path: Path) -> str:
    gdf = gpd.read_file(shp_path)
    if "gid" in gdf.columns:
        return "skipped"

    gdf = gdf.copy()
    gdf["gid"] = list(range(1, len(gdf) + 1))

    temp_dir = Path(tempfile.mkdtemp(prefix=f".fake_gid_{shp_path.stem}_", dir=str(shp_path.parent)))
    temp_shp_path = temp_dir / shp_path.name

    try:
        gdf.to_file(temp_shp_path, driver="ESRI Shapefile", index=False)
        _replace_with_temp_outputs(shp_path, temp_shp_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return "modified"


def main() -> None:
    _inject_repo_root()

    parser = argparse.ArgumentParser(description="Add sequential gid field to shapefiles that do not already have one.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory to recursively scan for .shp files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"`input_dir` is not a directory: {input_dir}")

    shp_files = _discover_shapefiles(input_dir)
    if not shp_files:
        print(f"[WARN] No .shp files found under: {input_dir}")
        return

    modified_count = 0
    skipped_count = 0
    failed_paths: list[tuple[Path, str]] = []

    print(f"[INFO] Discovered shapefiles: {len(shp_files)}")
    for index, shp_path in enumerate(shp_files, start=1):
        try:
            status = _process_one_shapefile(shp_path)
            if status == "modified":
                modified_count += 1
                print(f"[INFO] [{index}/{len(shp_files)}] Added gid: {shp_path}")
            else:
                skipped_count += 1
                print(f"[INFO] [{index}/{len(shp_files)}] Skipped (already has gid): {shp_path}")
        except Exception as exc:
            failed_paths.append((shp_path, f"{type(exc).__name__}: {exc}"))
            print(f"[WARN] [{index}/{len(shp_files)}] Failed: {shp_path} | {type(exc).__name__}: {exc}")

    print("[INFO] fake_gid_shp completed.")
    print(f"[INFO] Total shapefiles : {len(shp_files)}")
    print(f"[INFO] Modified        : {modified_count}")
    print(f"[INFO] Skipped         : {skipped_count}")
    print(f"[INFO] Failed          : {len(failed_paths)}")
    if failed_paths:
        print("[INFO] Failed file list:")
        for shp_path, error_message in failed_paths:
            print(f"[INFO]   - {shp_path} | {error_message}")


if __name__ == "__main__":
    main()
