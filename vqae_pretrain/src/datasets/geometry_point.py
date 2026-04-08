"""Point geometry codec placeholder.

This file is intentionally a scaffold for future extension. It exists to keep
engine code closed for modification and open for extension.
"""

from __future__ import annotations

from .geometry_codec_base import GeometryCodec


class PointGeometryCodec(GeometryCodec):
    """Placeholder codec for point geometry support."""

    def preprocess_geometry(self, geom):
        """Preprocess raw point geometry.

        Raises:
            NotImplementedError: Point codec is not implemented yet.
        """
        raise NotImplementedError("Point geometry codec is not implemented yet.")

    def cft_batch(self, batch_elements, lengths):
        """Convert point batches to frequency domain.

        Raises:
            NotImplementedError: Point codec is not implemented yet.
        """
        raise NotImplementedError("Point geometry codec is not implemented yet.")

    def icft_2d(self, f_uv_real, f_uv_imag=None, spatial_size=256):
        """Inverse transform for point codec.

        Raises:
            NotImplementedError: Point codec is not implemented yet.
        """
        raise NotImplementedError("Point geometry codec is not implemented yet.")
