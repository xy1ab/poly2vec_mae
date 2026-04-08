"""Polygon dataset implementation for VQAE training.

This dataset wraps pre-triangulated polygon samples and supports online
augmentation through configurable augmentation repeats.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import augment_triangles


class PolyDataset(Dataset):
    """Dataset of pre-triangulated polygons for VQAE pretraining.

    Args:
        data_list: List of triangle arrays, each shaped `[T, 3, 2]`.
        augment_times: Number of virtual repeats per sample.
    """

    def __init__(self, data_list: list[np.ndarray], augment_times: int = 1):
        """Initialize dataset state."""
        super().__init__()
        self.data_list = data_list
        self.augment_times = int(max(1, augment_times))
        self.total_len = len(self.data_list) * self.augment_times

    def __len__(self) -> int:
        """Return virtual dataset length after augmentation expansion."""
        return self.total_len

    def apply_augmentation(self, tris: np.ndarray) -> np.ndarray:
        """Apply online triangle augmentation.

        Args:
            tris: Input triangles shaped `[T,3,2]`.

        Returns:
            Augmented triangles.
        """
        return augment_triangles(tris)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Fetch a sample tensor.

        Args:
            idx: Sample index over expanded virtual length.

        Returns:
            Triangle tensor with shape `[T,3,2]`.
        """
        real_idx = idx % len(self.data_list)
        tris = self.data_list[real_idx]

        if idx >= len(self.data_list):
            tris = self.apply_augmentation(tris)

        return torch.tensor(tris, dtype=torch.float32)
