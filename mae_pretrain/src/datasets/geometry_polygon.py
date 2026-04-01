"""Polygon geometry codec and Fourier engine.

This module contains:
1) Polygon preprocessing helpers.
2) A high-performance polygon CFT/ICFT engine.
3) A `PolygonGeometryCodec` adapter that implements the common geometry codec
   interface for trainer and pipeline integration.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from shapely.geometry import Polygon
import triangle as tr

from .geometry_codec_base import GeometryCodec

_FOURIER_SINGULAR_EPS = 1e-6


def complex_mul(a_real: torch.Tensor, a_imag: torch.Tensor, b_real: torch.Tensor, b_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Multiply complex tensors represented by real/imag parts.

    Args:
        a_real: Real part of first operand.
        a_imag: Imaginary part of first operand.
        b_real: Real part of second operand.
        b_imag: Imaginary part of second operand.

    Returns:
        Tuple `(real, imag)` of the product.
    """
    out_real = a_real * b_real - a_imag * b_imag
    out_imag = a_real * b_imag + a_imag * b_real
    return out_real, out_imag


def ensure_polygon_array(poly: np.ndarray) -> np.ndarray:
    """Validate and cast polygon vertex array.

    Args:
        poly: Input array expected shape `[N, 2]`.

    Returns:
        `float32` polygon array.

    Raises:
        ValueError: If input shape is invalid.
    """
    poly_np = np.asarray(poly, dtype=np.float32)
    if poly_np.ndim != 2 or poly_np.shape[1] != 2:
        raise ValueError(f"Expected polygon shape [N,2], got {poly_np.shape}")
    return poly_np


def normalize_polygon_bbox(poly: np.ndarray) -> tuple[np.ndarray, float, float, float, int]:
    """Normalize polygon into `[-1, 1]` box via bounding-box scaling.

    Args:
        poly: Polygon vertex array `[N,2]`.

    Returns:
        Tuple `(poly_norm, cx, cy, side_len, node_count)`.
    """
    poly_np = ensure_polygon_array(poly)
    node_count = int(poly_np.shape[0])

    min_x, min_y = np.min(poly_np, axis=0)
    max_x, max_y = np.max(poly_np, axis=0)
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

    side_len = max(max_x - min_x, max_y - min_y)
    if side_len == 0:
        side_len = 1e-6

    poly_norm = (poly_np - np.array([cx, cy], dtype=np.float32)) / (side_len / 2.0)
    return poly_norm.astype(np.float32), float(cx), float(cy), float(side_len), node_count


def ensure_triangle_array(tris: np.ndarray) -> np.ndarray:
    """Validate and cast triangle array.

    Args:
        tris: Triangle array expected shape `[T, 3, 2]`.

    Returns:
        `float32` triangle array.

    Raises:
        ValueError: If input shape is invalid.
    """
    tri_np = np.asarray(tris, dtype=np.float32)
    if tri_np.ndim != 3 or tri_np.shape[1:] != (3, 2):
        raise ValueError(f"Expected triangles shape [T,3,2], got {tri_np.shape}")
    return tri_np


def pad_triangle_batch(triangles_list: Sequence[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length triangle arrays into a dense batch tensor.

    Args:
        triangles_list: List of arrays each shaped `[T_i,3,2]`.
        device: Target device.

    Returns:
        Tuple `(batch_tris, lengths)`.
    """
    if len(triangles_list) == 0:
        raise ValueError("triangles_list must not be empty")

    tri_arrays = [ensure_triangle_array(t) for t in triangles_list]
    lengths = torch.tensor([arr.shape[0] for arr in tri_arrays], dtype=torch.long, device=device)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0

    batch = torch.zeros((len(tri_arrays), max_len, 3, 2), dtype=torch.float32, device=device)
    for idx, arr in enumerate(tri_arrays):
        length = int(arr.shape[0])
        if length > 0:
            batch[idx, :length] = torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    return batch, lengths


def build_poly_fourier_converter_from_config(config: dict, device: str | torch.device = "cpu") -> "PolyFourierConverter":
    """Build polygon Fourier converter from config dictionary.

    Args:
        config: Runtime configuration dictionary.
        device: Target device.

    Returns:
        PolyFourierConverter instance.
    """
    return PolyFourierConverter(
        pos_freqs=int(config.get("pos_freqs", 31)),
        w_min=float(config.get("w_min", 0.1)),
        w_max=float(config.get("w_max", 100.0)),
        freq_type=str(config.get("freq_type", "geometric")),
        patch_size=int(config.get("patch_size", 2)),
        device=device,
    )


class PolyFourierConverter(nn.Module):
    """Continuous Fourier transform engine for triangulated polygons.

    The engine operates directly on triangle primitives and computes batched CFT
    and ICFT with padding-aware frequency grids.
    """

    def __init__(
        self,
        pos_freqs: int = 31,
        w_min: float = 0.1,
        w_max: float = 1.0,
        freq_type: str = "geometric",
        device: str | torch.device = "cuda",
        patch_size: int = 16,
    ):
        """Initialize converter and frequency grid buffers."""
        super().__init__()
        self.pos_freqs = pos_freqs
        self.w_min = w_min
        self.w_max = w_max
        self.freq_type = freq_type
        self.device = torch.device(device)
        self.patch_size = patch_size

        if self.pos_freqs < 1:
            raise ValueError(f"`pos_freqs` must be >= 1, got {self.pos_freqs}")
        if self.patch_size < 1:
            raise ValueError(f"`patch_size` must be >= 1, got {self.patch_size}")
        if self.w_max < self.w_min:
            raise ValueError(f"`w_max` must be >= `w_min`, got w_min={self.w_min}, w_max={self.w_max}")
        if self.freq_type == "geometric" and self.w_min <= 0:
            raise ValueError(f"`w_min` must be > 0 for geometric frequency grids, got {self.w_min}")

        self.U, self.V, self.pad_h, self.pad_w = self._build_meshgrid()

    def _build_meshgrid(self):
        """Build frequency meshgrid and compute patch-aligned paddings."""
        if self.freq_type == "geometric":
            g = (self.w_max / self.w_min) ** (1 / (self.pos_freqs - 1)) if self.pos_freqs > 1 else 1.0
            pos_w = [self.w_min * (g**u) for u in range(self.pos_freqs)]
        else:
            pos_w = np.linspace(self.w_min, self.w_max, self.pos_freqs).tolist()

        pos_w = torch.tensor(pos_w, dtype=torch.float32)

        wx = torch.cat((-torch.flip(pos_w, dims=[0]), torch.tensor([0.0]), pos_w))
        wy = torch.cat((torch.tensor([0.0]), pos_w))

        len_x, len_y = len(wx), len(wy)
        pad_h = (self.patch_size - (len_x % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (len_y % self.patch_size)) % self.patch_size

        u, v = torch.meshgrid(wx, wy, indexing="ij")
        if pad_h > 0 or pad_w > 0:
            u = torch.nn.functional.pad(u, (0, pad_w, 0, pad_h), value=0.0)
            v = torch.nn.functional.pad(v, (0, pad_w, 0, pad_h), value=0.0)

        return u.to(self.device), v.to(self.device), pad_h, pad_w

    def triangulate_polygon(self, coords: np.ndarray) -> np.ndarray:
        """Triangulate a polygon using constrained Delaunay triangulation.

        Args:
            coords: Polygon coordinates `[N,2]`.

        Returns:
            Triangle array `[T,3,2]`. Empty array when triangulation fails.
        """
        poly = Polygon(coords).buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            return np.zeros((0, 3, 2), dtype=np.float32)

        exterior = list(poly.exterior.coords)[:-1]
        if len(exterior) < 3:
            return np.zeros((0, 3, 2), dtype=np.float32)

        segments = [(i, (i + 1) % len(exterior)) for i in range(len(exterior))]
        poly_dict = {"vertices": exterior, "segments": segments}
        try:
            tri_data = tr.triangulate(poly_dict, "pq")
            triangles = tri_data["vertices"][tri_data["triangles"]]
            return triangles.astype(np.float32)
        except Exception:
            return np.zeros((0, 3, 2), dtype=np.float32)

    def _cft_single_triangle_batch(self, tris: torch.Tensor, _valid_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute triangle CFT for a batch of single triangles.

        Args:
            tris: Triangle tensor `[B,3,2]`.
            _valid_mask: Unused placeholder kept for compatibility.

        Returns:
            Tuple `(real, imag)` each shaped `[B,H,W]`.
        """
        pi = torch.pi
        u, v = self.U.unsqueeze(0), self.V.unsqueeze(0)

        q, r, s = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]
        xq, yq = q[:, 0].view(-1, 1, 1), q[:, 1].view(-1, 1, 1)
        xr, yr = r[:, 0].view(-1, 1, 1), r[:, 1].view(-1, 1, 1)
        xs, ys = s[:, 0].view(-1, 1, 1), s[:, 1].view(-1, 1, 1)

        det = xq * (yr - ys) + xr * (ys - yq) + xs * (yq - yr)
        area = torch.abs(0.5 * det)

        u_ = u * (xr - xq) + v * (yr - yq)
        v_ = u * (xs - xr) + v * (ys - yr)

        theta_shift = 2 * pi * (u * xq + v * yq)
        shift_real = torch.cos(theta_shift)
        shift_imag = -torch.sin(theta_shift)

        eps = _FOURIER_SINGULAR_EPS
        zero_mask = (u_.abs() <= eps) & (v_.abs() <= eps)
        uv_mask = ((u_ + v_).abs() <= eps) & ~zero_mask
        u_mask = (u_.abs() <= eps) & ~(zero_mask | uv_mask)
        v_mask = (v_.abs() <= eps) & ~(zero_mask | uv_mask | u_mask)
        normal_mask = ~(uv_mask | zero_mask | u_mask | v_mask)

        theta_u = 2 * pi * u_
        theta_v = 2 * pi * v_
        theta_uv = 2 * pi * (u_ + v_)

        base_u_real = torch.cos(theta_u)
        base_u_imag = -torch.sin(theta_u)
        base_v_real = torch.cos(theta_v)
        base_v_imag = -torch.sin(theta_v)
        base_uv_real = torch.cos(theta_uv)
        base_uv_imag = -torch.sin(theta_uv)

        part1 = torch.zeros_like(u_, dtype=torch.float32)
        part2_real = torch.zeros_like(u_, dtype=torch.float32)
        part2_imag = torch.zeros_like(u_, dtype=torch.float32)

        part1[normal_mask] = 1.0 / (
            4
            * pi**2
            * (u_[normal_mask] * v_[normal_mask] * (u_[normal_mask] + v_[normal_mask]))
        )
        part2_real[normal_mask] = (
            -u_[normal_mask] * base_uv_real[normal_mask]
            + (u_[normal_mask] + v_[normal_mask]) * base_u_real[normal_mask]
            - v_[normal_mask]
        )
        part2_imag[normal_mask] = (
            -u_[normal_mask] * base_uv_imag[normal_mask]
            + (u_[normal_mask] + v_[normal_mask]) * base_u_imag[normal_mask]
        )

        part1[uv_mask] = -1.0 / (4 * pi**2 * u_[uv_mask] ** 2)
        part2_real[uv_mask] = base_u_real[uv_mask] - 1.0
        part2_imag[uv_mask] = base_u_imag[uv_mask] + 2 * pi * u_[uv_mask]

        part1[u_mask] = -1.0 / (4 * pi**2 * v_[u_mask] ** 2)
        part2_real[u_mask] = base_v_real[u_mask] - 1.0
        part2_imag[u_mask] = base_v_imag[u_mask] + 2 * pi * v_[u_mask]

        part1[v_mask] = 1.0 / (4 * pi**2 * u_[v_mask] ** 2)
        a_real = torch.ones_like(u_[v_mask])
        a_imag = 2 * pi * u_[v_mask]
        mul_real = a_real * base_u_real[v_mask] - a_imag * base_u_imag[v_mask]
        mul_imag = a_real * base_u_imag[v_mask] + a_imag * base_u_real[v_mask]
        part2_real[v_mask] = mul_real - 1.0
        part2_imag[v_mask] = mul_imag

        scale = 2.0 * area * part1
        ft_real, ft_imag = complex_mul(scale * part2_real, scale * part2_imag, shift_real, shift_imag)

        ft_real = torch.where(zero_mask, area, ft_real)
        ft_imag = torch.where(zero_mask, torch.zeros_like(ft_imag), ft_imag)

        if self.pad_h > 0:
            ft_real[:, -self.pad_h :, :] = 0.0
            ft_imag[:, -self.pad_h :, :] = 0.0
        if self.pad_w > 0:
            ft_real[:, :, -self.pad_w :] = 0.0
            ft_imag[:, :, -self.pad_w :] = 0.0

        return ft_real, ft_imag

    def cft_polygon_batch(self, batch_triangles: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batched polygon CFT magnitude and phase.

        Args:
            batch_triangles: Padded triangle tensor `[B,max_T,3,2]`.
            lengths: Valid triangle counts `[B]`.

        Returns:
            Tuple `(mag_log, phase)` each shaped `[B,1,H,W]`.
        """
        batch_size, max_triangles, _, _ = batch_triangles.shape
        h, w = self.U.shape

        ft_real_total = torch.zeros((batch_size, h, w), dtype=torch.float32, device=self.device)
        ft_imag_total = torch.zeros((batch_size, h, w), dtype=torch.float32, device=self.device)
        batch_triangles = batch_triangles.to(self.device)
        lengths = lengths.to(self.device)

        valid_mask = torch.arange(max_triangles, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        valid_tris = batch_triangles[valid_mask]
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(batch_size, max_triangles)[valid_mask]

        chunk_size = 50000
        for offset in range(0, valid_tris.shape[0], chunk_size):
            tris_chunk = valid_tris[offset : offset + chunk_size]
            b_idx_chunk = batch_indices[offset : offset + chunk_size]
            ft_chunk_real, ft_chunk_imag = self._cft_single_triangle_batch(tris_chunk, None)
            ft_real_total.index_add_(0, b_idx_chunk, ft_chunk_real)
            ft_imag_total.index_add_(0, b_idx_chunk, ft_chunk_imag)

        mag = torch.log1p(torch.hypot(ft_real_total, ft_imag_total)).unsqueeze(1)
        phase = torch.atan2(ft_imag_total, ft_real_total).unsqueeze(1)
        return mag, phase

    def icft_2d(
        self,
        f_uv_real: torch.Tensor,
        f_uv_imag: torch.Tensor | None = None,
        spatial_size: int = 256,
    ) -> torch.Tensor:
        """Compute inverse non-uniform Fourier transform to spatial field.

        Args:
            f_uv_real: Real part tensor `[B,H,W]`.
            f_uv_imag: Imaginary part tensor `[B,H,W]` (optional).
            spatial_size: Output raster size.

        Returns:
            Normalized spatial field tensor `[B,S,S]` in `[0,1]`.
        """
        if f_uv_imag is None:
            if torch.is_complex(f_uv_real):
                f_uv_imag = f_uv_real.imag
                f_uv_real = f_uv_real.real
            else:
                raise ValueError("`f_uv_imag` is required when `f_uv_real` is not a complex tensor.")

        batch_size, h, w = f_uv_real.shape

        x = torch.linspace(-1, 1, spatial_size, device=self.device)
        y = torch.linspace(1, -1, spatial_size, device=self.device)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")

        x_flat = x_grid.reshape(-1)
        y_flat = y_grid.reshape(-1)
        u_flat = self.U.reshape(-1)
        v_flat = self.V.reshape(-1)

        valid_h = h - self.pad_h
        valid_w = w - self.pad_w
        wx = self.U[:valid_h, 0].clone()
        wy = self.V[0, :valid_w].clone()

        du = torch.zeros_like(wx)
        for i in range(valid_h):
            if i == 0:
                du[i] = wx[1] - wx[0] if valid_h > 1 else 1.0
            elif i == valid_h - 1:
                du[i] = wx[i] - wx[i - 1]
            else:
                du[i] = (wx[i + 1] - wx[i - 1]) / 2.0

        dv = torch.zeros_like(wy)
        for j in range(valid_w):
            if j == 0:
                dv[j] = wy[1] - wy[0] if valid_w > 1 else 1.0
            elif j == valid_w - 1:
                dv[j] = wy[j] - wy[j - 1]
            else:
                dv[j] = (wy[j + 1] - wy[j - 1]) / 2.0

        d_u, d_v = torch.meshgrid(du, dv, indexing="ij")

        weights = torch.zeros((h, w), device=self.device)
        weights[:valid_h, :valid_w] = d_u * d_v
        weights = weights.reshape(-1).unsqueeze(0)

        sym_weights = torch.where(v_flat > 1e-6, 2.0, 1.0).unsqueeze(0)
        weights = weights * sym_weights

        phase_mat = 2 * torch.pi * (u_flat.unsqueeze(1) * x_flat.unsqueeze(0) + v_flat.unsqueeze(1) * y_flat.unsqueeze(0))
        e_real = torch.cos(phase_mat)
        e_imag = torch.sin(phase_mat)

        f_flat_real = f_uv_real.reshape(batch_size, -1) * weights
        f_flat_imag = f_uv_imag.reshape(batch_size, -1) * weights

        # Re((a+ib) @ (c+id)) = a@c - b@d
        f_recon = (
            torch.matmul(f_flat_real, e_real) - torch.matmul(f_flat_imag, e_imag)
        ).reshape(batch_size, spatial_size, spatial_size)

        f_min = f_recon.amin(dim=(1, 2), keepdim=True)
        f_max = f_recon.amax(dim=(1, 2), keepdim=True)
        f_norm = (f_recon - f_min) / (f_max - f_min + 1e-8)
        return f_norm


class PolygonGeometryCodec(GeometryCodec):
    """Geometry codec implementation for polygon data."""

    def __init__(self, converter: PolyFourierConverter):
        """Initialize codec with Fourier converter backend.

        Args:
            converter: Polygon Fourier converter instance.
        """
        self.converter = converter

    @classmethod
    def from_config(cls, config: dict, device: str | torch.device) -> "PolygonGeometryCodec":
        """Construct polygon codec from runtime config.

        Args:
            config: Runtime config dictionary.
            device: Runtime device.

        Returns:
            Polygon geometry codec.
        """
        converter = build_poly_fourier_converter_from_config(config, device=device)
        return cls(converter=converter)

    def triangulate_polygon(self, poly_norm: np.ndarray) -> np.ndarray:
        """Triangulate normalized polygon vertices.

        Args:
            poly_norm: Normalized polygon coordinates `[N,2]`.

        Returns:
            Triangles `[T,3,2]`.
        """
        return self.converter.triangulate_polygon(poly_norm)

    def preprocess_geometry(self, geom: np.ndarray) -> np.ndarray:
        """Preprocess polygon vertices into triangle primitives.

        Args:
            geom: Raw polygon vertices `[N,2]`.

        Returns:
            Triangle array `[T,3,2]`.
        """
        poly_norm, _, _, _, _ = normalize_polygon_bbox(geom)
        return self.triangulate_polygon(poly_norm)

    def cft_batch(self, batch_elements: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batched CFT for polygon triangle batches.

        Args:
            batch_elements: Padded triangles `[B,max_T,3,2]`.
            lengths: Valid triangle counts `[B]`.

        Returns:
            Tuple `(mag_log, phase)` each `[B,1,H,W]`.
        """
        return self.converter.cft_polygon_batch(batch_elements, lengths)

    def icft_2d(
        self,
        f_uv_real: torch.Tensor,
        f_uv_imag: torch.Tensor | None = None,
        spatial_size: int = 256,
    ) -> torch.Tensor:
        """Run inverse CFT.

        Args:
            f_uv_real: Real Fourier tensor `[B,H,W]`.
            f_uv_imag: Imaginary Fourier tensor `[B,H,W]` (optional).
            spatial_size: Output raster size.

        Returns:
            Spatial raster `[B,S,S]`.
        """
        return self.converter.icft_2d(f_uv_real, f_uv_imag=f_uv_imag, spatial_size=spatial_size)

    def triangles_to_image_channels(self, triangles_list: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert triangle arrays to MAE input channels (mag, cos, sin).

        Args:
            triangles_list: List of triangle arrays `[T_i,3,2]`.

        Returns:
            Tensor with shape `[B,3,H,W]`.
        """
        batch_tris, lengths = pad_triangle_batch(triangles_list, device=self.converter.device)
        mag, phase = self.cft_batch(batch_tris, lengths)
        return torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)
