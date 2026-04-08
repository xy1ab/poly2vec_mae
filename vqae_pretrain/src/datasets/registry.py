"""Geometry codec registry.

This module provides a single selection function to construct geometry-specific
codecs while keeping training and pipeline logic geometry-agnostic.
"""

from __future__ import annotations

from typing import Any

from .geometry_line import LineGeometryCodec
from .geometry_point import PointGeometryCodec
from .geometry_polygon import PolygonGeometryCodec


def get_geometry_codec(geom_type: str, config: dict[str, Any], device: str):
    """Build geometry codec from geometry type string.

    Args:
        geom_type: Geometry type name, e.g. `polygon`, `point`, `line`.
        config: Runtime configuration dictionary.
        device: Runtime device string.

    Returns:
        Geometry codec instance.

    Raises:
        NotImplementedError: If geometry type is unsupported.
    """
    key = str(geom_type).strip().lower()
    if key == "polygon":
        return PolygonGeometryCodec.from_config(config=config, device=device)
    if key == "point":
        return PointGeometryCodec()
    if key == "line":
        return LineGeometryCodec()
    raise NotImplementedError(f"Unsupported geometry type: {geom_type}")
