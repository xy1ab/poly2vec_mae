"""Abstract geometry codec contract.

This module defines a small interface that decouples engine/model code from
geometry-specific preprocessing and Fourier conversions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class GeometryCodec(ABC):
    """Abstract geometry codec interface for OCP-style extension.

    Concrete implementations (polygon, point, line) should provide methods for
    preprocessing raw vectors, batch CFT conversion, and optional inverse CFT.
    """

    @abstractmethod
    def preprocess_geometry(self, geom: np.ndarray):
        """Preprocess a raw geometry object into model-ready elements."""

    @abstractmethod
    def cft_batch(self, batch_elements: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert batched geometry elements into magnitude and phase tensors."""

    @abstractmethod
    def icft_2d(
        self,
        f_uv_real: torch.Tensor,
        f_uv_imag: torch.Tensor | None = None,
        spatial_size: int = 256,
    ) -> torch.Tensor:
        """Run inverse transform from real/imag frequency parts to spatial field."""
