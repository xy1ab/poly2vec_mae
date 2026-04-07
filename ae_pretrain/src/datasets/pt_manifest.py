"""Shard manifest utilities for AE triangle datasets.

This module implements the metadata layer used by eager and lazy dataset
readers. It is responsible for:
1) Discovering `.pt` shard files from a user-specified data directory.
2) Counting samples stored in each shard.
3) Building a stable global-index view over multiple shard files.
4) Estimating a conservative shard-cache size for lazy loading.

The manifest intentionally contains only lightweight metadata so the training
engine can switch between eager and lazy reading strategies without changing
the training loop itself.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Callable

from .shard_io import load_triangle_shard, resolve_triangle_shard_paths


@dataclass(frozen=True)
class PtShardInfo:
    """Metadata describing one serialized `.pt` shard file.

    Args:
        path: Absolute path of the shard file.
        num_samples: Number of polygon samples stored in this shard.
        size_bytes: On-disk file size in bytes.
        start_index: Inclusive global sample offset of this shard.
    """

    path: Path
    num_samples: int
    size_bytes: int
    start_index: int


def _read_mem_available_bytes() -> int:
    """Read best-effort available system memory in bytes.

    Returns:
        Estimated available host memory in bytes. Returns `0` when the runtime
        cannot provide a reliable value.
    """
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.is_file():
        try:
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
        except Exception:
            return 0

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (ValueError, OSError, AttributeError):
        return 0


class PtShardManifest:
    """Global index manifest for multiple `.pt` dataset shards.

    The manifest exposes a stable sample index space `[0, total_samples)` so
    downstream datasets can map any global sample id back to a `(shard_id,
    local_index)` pair.
    """

    def __init__(self, shard_infos: list[PtShardInfo]):
        """Store validated shard metadata.

        Args:
            shard_infos: Ordered shard descriptors with non-overlapping global
                index ranges.
        """
        if not shard_infos:
            raise ValueError("PtShardManifest requires at least one non-empty shard.")

        self.shards = shard_infos
        self.num_shards = len(self.shards)
        self.total_samples = sum(info.num_samples for info in self.shards)
        self.total_size_bytes = sum(info.size_bytes for info in self.shards)
        self.avg_shard_bytes = self.total_size_bytes / max(1, self.num_shards)
        self._start_offsets = [info.start_index for info in self.shards]

    @classmethod
    def from_pt_files(cls, pt_files: list[str | Path]) -> "PtShardManifest":
        """Build a manifest from an explicit shard-file list.

        Args:
            pt_files: Input `.pt` paths. The provided order is preserved.

        Returns:
            Manifest instance with global offsets for each non-empty shard.
        """
        normalized_paths = [Path(path).expanduser().resolve() for path in pt_files]
        if not normalized_paths:
            raise FileNotFoundError("No .pt files were provided for manifest construction.")

        shard_infos: list[PtShardInfo] = []
        global_start = 0
        for path in normalized_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Shard file does not exist: {path}")

            shard_data = load_triangle_shard(path)
            num_samples = len(shard_data)
            if num_samples <= 0:
                continue

            shard_infos.append(
                PtShardInfo(
                    path=path,
                    num_samples=num_samples,
                    size_bytes=int(path.stat().st_size),
                    start_index=global_start,
                )
            )
            global_start += num_samples

        if not shard_infos:
            raise ValueError("All discovered shard files were empty.")

        return cls(shard_infos)

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        warn_fn: Callable[[str], None] | None = None,
    ) -> "PtShardManifest":
        """Discover `.pt` shards from one data directory.

        Args:
            data_dir: Directory that contains one or more shard `.pt` files.
            warn_fn: Optional warning sink used during manifest fallback.

        Returns:
            Manifest instance over all discovered shard files.
        """
        pt_files = resolve_triangle_shard_paths(data_dir, warn_fn=warn_fn)
        return cls.from_pt_files(pt_files)

    def locate_sample(self, global_index: int) -> tuple[int, int]:
        """Map one global sample index to `(shard_id, local_index)`.

        Args:
            global_index: Zero-based sample index in manifest-global space.

        Returns:
            Tuple containing shard position and sample offset inside that shard.
        """
        if global_index < 0 or global_index >= self.total_samples:
            raise IndexError(
                f"Global sample index out of range: {global_index} not in [0, {self.total_samples})"
            )

        shard_id = bisect_right(self._start_offsets, global_index) - 1
        shard_info = self.shards[shard_id]
        local_index = global_index - shard_info.start_index
        return shard_id, local_index

    def recommend_cache_shards(
        self,
        world_size: int,
        num_workers: int,
        cache_ratio: float = 0.05,
        max_shards: int = 2,
    ) -> int:
        """Estimate a conservative shard-cache size for lazy loading.

        Args:
            world_size: Number of distributed training ranks.
            num_workers: DataLoader worker count per rank.
            cache_ratio: Fraction of available host memory allowed for shard
                caches across all ranks and workers.
            max_shards: Hard upper bound of cached shards per dataset instance.

        Returns:
            Recommended shard-cache size for one dataset instance.
        """
        max_shards = max(1, int(max_shards))
        available_bytes = _read_mem_available_bytes()
        if available_bytes <= 0:
            return 1

        total_budget = int(available_bytes * max(0.01, float(cache_ratio)))
        process_factor = max(1, int(world_size)) * max(1, int(num_workers))
        bytes_per_dataset = total_budget // process_factor
        cache_by_memory = bytes_per_dataset // max(1, int(self.avg_shard_bytes))
        return max(1, min(max_shards, int(cache_by_memory)))
