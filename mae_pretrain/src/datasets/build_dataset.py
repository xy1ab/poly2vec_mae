"""Dataset building utilities for polygon triangulation.

This module reads vector files, normalizes polygons, triangulates each polygon,
and saves triangle tensors into one or multiple `.pt` shards.

Key features:
1) File-level parallel triangulation with process pool.
2) Stable file-order merge even when futures complete out-of-order.
3) Size-based output sharding for large datasets.
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
from tqdm import tqdm

from mae_pretrain.src.datasets.geometry_polygon import PolyFourierConverter


_PER_SAMPLE_OVERHEAD_BYTES = 1024


def _collect_vector_files(input_dirs: Iterable[str | Path]) -> list[Path]:
    """Collect shapefile and geojson files recursively.

    Args:
        input_dirs: Directory list to scan.

    Returns:
        Sorted unique file path list.
    """
    files: list[Path] = []
    for directory in input_dirs:
        directory = Path(directory)
        files.extend(Path(p) for p in glob.glob(str(directory / "**" / "*.shp"), recursive=True))
        files.extend(Path(p) for p in glob.glob(str(directory / "**" / "*.geojson"), recursive=True))

    return sorted(set(files))


def _triangulate_file_worker(file_index: int, file_path: str) -> dict[str, Any]:
    """Triangulate polygons from one vector file in a worker process.

    Args:
        file_index: Stable index of source file in sorted file list.
        file_path: Source vector path.

    Returns:
        Dictionary containing worker status and triangle outputs.
    """
    converter = PolyFourierConverter(device="cpu")
    out_triangles: list[np.ndarray] = []
    skipped_count = 0

    try:
        gdf = gpd.read_file(file_path)
    except Exception as exc:
        return {
            "index": file_index,
            "file_path": file_path,
            "ok": False,
            "error": str(exc),
            "source_geometry_count": 0,
            "skipped_count": 0,
            "triangles": [],
        }

    source_geometry_count = int(len(gdf.geometry))

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            skipped_count += 1
            continue

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            skipped_count += 1
            continue

        for poly in polygons:
            coords = np.asarray(poly.exterior.coords, dtype=np.float32)
            if coords.shape[0] < 4:
                skipped_count += 1
                continue

            minx, miny = float(coords[:, 0].min()), float(coords[:, 1].min())
            maxx, maxy = float(coords[:, 0].max()), float(coords[:, 1].max())
            cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
            max_range = max(maxx - minx, maxy - miny) / 2.0
            if max_range < 1e-6:
                skipped_count += 1
                continue

            norm_coords = (coords - np.array([cx, cy], dtype=np.float32)) / float(max_range)
            tris = converter.triangulate_polygon(norm_coords)
            if tris.shape[0] > 0:
                out_triangles.append(tris.astype(np.float32))
            else:
                skipped_count += 1

    return {
        "index": file_index,
        "file_path": file_path,
        "ok": True,
        "error": "",
        "source_geometry_count": source_geometry_count,
        "skipped_count": skipped_count,
        "triangles": out_triangles,
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


def _consume_worker_result(result: dict[str, Any], writer: _ShardWriter) -> tuple[bool, int, int, int]:
    """Consume one worker result and append triangles to shard writer.

    Args:
        result: Worker result dictionary.
        writer: Shard writer instance.

    Returns:
        Tuple `(ok, source_geometry_count, skipped_count, triangulated_count)`.
    """
    ok = bool(result.get("ok", False))
    source_geometry_count = int(result.get("source_geometry_count", 0))
    skipped_count = int(result.get("skipped_count", 0))

    if not ok:
        file_path = result.get("file_path", "<unknown>")
        error = result.get("error", "unknown error")
        tqdm.write(f"[WARN] Failed to read/process {file_path}: {error}")
        return False, source_geometry_count, skipped_count, 0

    triangles = result.get("triangles", [])
    triangulated_count = len(triangles)
    writer.add_many(triangles)
    return True, source_geometry_count, skipped_count, triangulated_count


def process_and_save(
    input_dirs: Iterable[str | Path],
    output_path: str | Path,
    num_workers: int = 0,
    shard_size_mb: float = 0.0,
) -> None:
    """Build triangulated polygon dataset and save to disk.

    Args:
        input_dirs: Iterable of source directories containing vector files.
        output_path: Output `.pt` base path.
        num_workers: File-level process count. `<=0` means auto.
        shard_size_mb: Target shard size in MB. `<=0` means single `.pt` output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_list = _collect_vector_files(input_dirs)
    if not file_list:
        print("[WARN] No .shp/.geojson files found under given input_dirs.")
        return

    worker_count = _resolve_num_workers(num_workers=num_workers, file_count=len(file_list))
    shard_size_mb = float(shard_size_mb)

    print(f"[INFO] Discovered vector files: {len(file_list)}")
    print(f"[INFO] File-level workers: {worker_count}")
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

    if worker_count <= 1:
        with tqdm(total=len(file_list), desc="Triangulating files", unit="file") as pbar:
            for file_index, file_path in enumerate(file_list):
                result = _triangulate_file_worker(file_index=file_index, file_path=str(file_path))
                ok, src_count, skipped_count, tri_count = _consume_worker_result(result, writer)
                files_ok += int(ok)
                files_failed += int(not ok)
                source_geometries_total += src_count
                skipped_total += skipped_count
                triangulated_total += tri_count
                pbar.update(1)
    else:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
            future_to_meta = {
                executor.submit(_triangulate_file_worker, file_index, str(file_path)): (file_index, file_path)
                for file_index, file_path in enumerate(file_list)
            }

            pending_results: dict[int, dict[str, Any]] = {}
            next_index = 0

            with tqdm(total=len(file_list), desc="Triangulating files", unit="file") as pbar:
                for future in as_completed(future_to_meta):
                    file_index, file_path = future_to_meta[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "index": file_index,
                            "file_path": str(file_path),
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "source_geometry_count": 0,
                            "skipped_count": 0,
                            "triangles": [],
                        }

                    pending_results[int(result["index"])] = result
                    pbar.update(1)

                    while next_index in pending_results:
                        ordered_result = pending_results.pop(next_index)
                        ok, src_count, skipped_count, tri_count = _consume_worker_result(ordered_result, writer)
                        files_ok += int(ok)
                        files_failed += int(not ok)
                        source_geometries_total += src_count
                        skipped_total += skipped_count
                        triangulated_total += tri_count
                        next_index += 1

    if triangulated_total == 0:
        print("[WARN] No valid polygons were triangulated. Nothing to save.")
        return

    shard_paths, manifest_path = writer.finalize()
    if not shard_paths:
        print("[WARN] No output shard was written.")
        return

    print("[INFO] Triangulation completed.")
    print(f"[INFO] Files succeeded/failed : {files_ok}/{files_failed}")
    print(f"[INFO] Source geometries       : {source_geometries_total}")
    print(f"[INFO] Triangulated samples    : {triangulated_total}")
    print(f"[INFO] Skipped geometries      : {skipped_total}")

    if shard_size_mb > 0:
        print(f"[INFO] Output shards           : {len(shard_paths)}")
        for shard_path in shard_paths:
            size_mb = shard_path.stat().st_size / (1024.0 * 1024.0)
            print(f"[INFO]   - {shard_path} ({size_mb:.2f} MB)")
        if manifest_path is not None:
            print(f"[INFO] Shard manifest         : {manifest_path}")
    else:
        print(f"[INFO] Saved output           : {shard_paths[0]}")

