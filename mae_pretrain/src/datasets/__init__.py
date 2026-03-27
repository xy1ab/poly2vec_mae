"""Dataset package for MAE pretraining.

This package contains dataset loading, augmentation, geometry codecs, and codec
registry utilities for extensible geometry support.
"""

from .collate import mae_collate_fn
from .polygon_dataset import PolyMAEDataset
from .registry import get_geometry_codec
from .transforms import augment_triangles
__all__ = ["PolyMAEDataset", "mae_collate_fn", "get_geometry_codec"]
