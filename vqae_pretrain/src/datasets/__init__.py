"""Dataset package for VQAE pretraining.

This package intentionally keeps its top-level import surface lazy so
submodule-oriented tools and scripts can import `build_dataset_triangle`
without pulling in unrelated deep-learning dependencies during module import.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PolyDataset",
    "PtShardManifest",
    "EagerShardedPolyDataset",
    "LazyShardedPolyDataset",
    "triangle_collate_fn",
    "get_geometry_codec",
    "augment_triangles",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "PolyDataset": (".polygon_dataset", "PolyDataset"),
    "PtShardManifest": (".pt_manifest", "PtShardManifest"),
    "EagerShardedPolyDataset": (".sharded_pt_dataset", "EagerShardedPolyDataset"),
    "LazyShardedPolyDataset": (".sharded_pt_dataset", "LazyShardedPolyDataset"),
    "triangle_collate_fn": (".collate", "triangle_collate_fn"),
    "get_geometry_codec": (".registry", "get_geometry_codec"),
    "augment_triangles": (".transforms", "augment_triangles"),
}


def __getattr__(name: str) -> Any:
    """Resolve dataset package exports lazily on first access.

    Args:
        name: Exported symbol name.

    Returns:
        Requested symbol object.

    Raises:
        AttributeError: If the symbol is not a declared package export.
    """
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
