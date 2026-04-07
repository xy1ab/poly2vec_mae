"""Shard-aware eager and lazy datasets for polygon MAE pretraining.

This module implements the mature map-style dataset pattern used in many
PyTorch projects:
1) Eager mode loads all shard files into host memory once.
2) Lazy mode keeps only shard metadata and loads individual shards on demand.
3) Train/validation split is expressed with stable sample-index lists, so both
   eager and lazy readers share identical sampling semantics.

The dataset classes only handle sample retrieval and optional online
augmentation. DDP shuffling remains the responsibility of the standard
DistributedSampler used by the training engine.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .pt_manifest import PtShardManifest
from .shard_io import load_triangle_shard
from .transforms import augment_triangles


def _ensure_numpy_float32(sample) -> np.ndarray:
    """Normalize one loaded polygon sample to a float32 NumPy array.

    Args:
        sample: One polygon sample loaded from a shard file.

    Returns:
        NumPy array shaped `[T, 3, 2]` with `float32` dtype.
    """
    if isinstance(sample, np.ndarray):
        return sample.astype(np.float32, copy=False)
    if torch.is_tensor(sample):
        return sample.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(sample, dtype=np.float32)


def load_all_samples_from_manifest(manifest: PtShardManifest) -> list[np.ndarray]:
    """Load and normalize all samples referenced by one manifest.

    Args:
        manifest: Shard manifest describing the dataset.

    Returns:
        Concatenated sample list in manifest-global order.
    """
    all_samples: list[np.ndarray] = []
    for shard_info in manifest.shards:
        shard_data = load_triangle_shard(shard_info.path)
        all_samples.extend(_ensure_numpy_float32(sample) for sample in shard_data)
    return all_samples


class _BaseIndexedPolyDataset(Dataset):
    """Shared index-based dataset behavior for eager and lazy readers.

    Args:
        sample_indices: Global sample ids that belong to this dataset split.
        augment_times: Number of virtual repeats used for online augmentation.
    """

    def __init__(self, sample_indices: Sequence[int], augment_times: int = 1):
        """Store split indices and augmentation settings."""
        super().__init__()
        self.sample_indices = [int(index) for index in sample_indices]
        if not self.sample_indices:
            raise ValueError("Dataset split must contain at least one sample.")

        self.augment_times = self._validate_augment_times(augment_times)
        self.base_len = len(self.sample_indices)
        self.total_len = self.base_len * self.augment_times

    @staticmethod
    def _validate_augment_times(augment_times: int) -> int:
        """Validate dataset virtual-repeat factor.

        Args:
            augment_times: Requested repeat count.

        Returns:
            Validated integer repeat count.
        """
        try:
            augment_float = float(augment_times)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"`augment_times` must be an integer >= 1, got {augment_times!r}") from exc

        if not augment_float.is_integer():
            raise ValueError(f"`augment_times` must be an integer >= 1, got {augment_times!r}")

        augment_value = int(augment_float)
        if augment_value < 1:
            raise ValueError(f"`augment_times` must be >= 1, got {augment_value}")
        return augment_value

    def __len__(self) -> int:
        """Return virtual dataset length after augmentation expansion."""
        return self.total_len

    def apply_augmentation(self, tris: np.ndarray) -> np.ndarray:
        """Apply online triangle augmentation.

        Args:
            tris: Triangle coordinates shaped `[T, 3, 2]`.

        Returns:
            Augmented triangle coordinates.
        """
        return augment_triangles(tris)

    def get_base_sample(self, base_index: int) -> np.ndarray:
        """Fetch one non-augmented sample from the dataset split.

        Args:
            base_index: Split-local sample index in `[0, base_len)`.

        Returns:
            Triangle array shaped `[T, 3, 2]`.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> torch.Tensor:
        """Fetch one sample tensor with optional augmentation applied.

        Args:
            index: Dataset index over the virtual expanded length.

        Returns:
            Float32 tensor shaped `[T, 3, 2]`.
        """
        if index < 0 or index >= self.total_len:
            raise IndexError(f"Dataset index out of range: {index}")

        base_index = index % self.base_len
        tris = self.get_base_sample(base_index)

        if index >= self.base_len:
            tris = self.apply_augmentation(tris)

        return torch.as_tensor(tris, dtype=torch.float32)


class EagerShardedPolyDataset(_BaseIndexedPolyDataset):
    """Dataset backed by one in-memory sample list.

    Args:
        all_samples: All dataset samples already loaded into host memory.
        sample_indices: Split-local mapping into `all_samples`.
        augment_times: Number of virtual repeats used for train augmentation.
    """

    def __init__(
        self,
        all_samples: Sequence[np.ndarray],
        sample_indices: Sequence[int],
        augment_times: int = 1,
    ):
        """Initialize eager dataset storage."""
        self.all_samples = list(all_samples)
        super().__init__(sample_indices=sample_indices, augment_times=augment_times)

    def get_base_sample(self, base_index: int) -> np.ndarray:
        """Return one split-local non-augmented sample."""
        global_index = self.sample_indices[base_index]
        return self.all_samples[global_index]


class LazyShardedPolyDataset(_BaseIndexedPolyDataset):
    """Dataset that loads `.pt` shards on demand and caches a small subset.

    Args:
        manifest: Shard manifest describing the dataset.
        sample_indices: Split-local global sample ids.
        augment_times: Number of virtual repeats used for train augmentation.
        max_cached_shards: Maximum number of loaded shards retained in memory
            per dataset instance.
    """

    def __init__(
        self,
        manifest: PtShardManifest,
        sample_indices: Sequence[int],
        augment_times: int = 1,
        max_cached_shards: int = 1,
    ):
        """Initialize lazy dataset state and LRU shard cache."""
        self.manifest = manifest
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._shard_cache: OrderedDict[int, list[np.ndarray]] = OrderedDict()
        super().__init__(sample_indices=sample_indices, augment_times=augment_times)

    def _load_shard(self, shard_id: int) -> list[np.ndarray]:
        """Load one shard into the local LRU cache when needed.

        Args:
            shard_id: Manifest-local shard index.

        Returns:
            Normalized list of polygon samples stored in that shard.
        """
        cached = self._shard_cache.get(shard_id)
        if cached is not None:
            self._shard_cache.move_to_end(shard_id)
            return cached

        shard_info = self.manifest.shards[shard_id]
        shard_data = load_triangle_shard(shard_info.path)
        normalized_shard = [_ensure_numpy_float32(sample) for sample in shard_data]
        self._shard_cache[shard_id] = normalized_shard
        self._shard_cache.move_to_end(shard_id)

        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)

        return normalized_shard

    def get_base_sample(self, base_index: int) -> np.ndarray:
        """Resolve one split-local sample through the manifest and shard cache."""
        global_index = self.sample_indices[base_index]
        shard_id, local_index = self.manifest.locate_sample(global_index)
        shard_samples = self._load_shard(shard_id)
        return shard_samples[local_index]
