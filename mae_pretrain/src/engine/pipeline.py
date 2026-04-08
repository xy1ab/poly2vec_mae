"""Unified pipeline APIs for downstream consumption.

This module exposes two stable classes:
1) `PolyEncoderPipeline` for embedding extraction.
2) `PolyMaeReconstructionPipeline` for MAE reconstruction outputs.

Downstream code should only depend on this file and exported model bundles.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..datasets.geometry_polygon import ensure_polygon_array, normalize_polygon_bbox, pad_triangle_batch
from ..datasets.registry import get_geometry_codec
from ..models.factory import load_mae_model, load_pretrained_encoder
from ..utils.config import load_config_any
from ..utils.precision import autocast_context, normalize_precision


class PolyEncoderPipeline:
    """Encoder inference pipeline for polygon vectors and triangle inputs.

    Args:
        weight_path: Encoder checkpoint path.
        config_path: Config path from checkpoint directory.
        device: Runtime device string.
        precision: Runtime precision (`fp32`, `bf16`, `fp16`).
    """

    def __init__(
        self,
        weight_path: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        precision: str = "bf16",
    ):
        """Initialize encoder pipeline resources."""
        self.device = torch.device(device)
        self.precision = normalize_precision(precision)

        self.config = load_config_any(config_path)
        self.geom_type = str(self.config.get("geom_type", "polygon")).lower()

        self.codec = get_geometry_codec(self.geom_type, self.config, device=str(self.device))
        self.encoder = load_pretrained_encoder(
            weight_path=weight_path,
            config_path=config_path,
            device=self.device,
            precision=self.precision,
        )

        self.embed_dim = int(self.config.get("embed_dim", 256))
        self.final_dim = self.embed_dim + 4

    def _preprocess_polygon(self, poly: np.ndarray):
        """Normalize and triangulate polygon input.

        Args:
            poly: Raw polygon vertices `[N,2]`.

        Returns:
            Tuple `(triangles, cx, cy, side_len, node_count)`.
        """
        poly_norm, cx, cy, side_len, n_nodes = normalize_polygon_bbox(poly)
        triangles = self.codec.triangulate_polygon(poly_norm)
        return triangles, cx, cy, side_len, n_nodes

    def vector_to_triangles(self, geometries: Sequence[np.ndarray], node_count_scale: float = 100.0):
        """Convert raw polygon vectors into triangle batches and metadata.

        Args:
            geometries: Sequence of polygon arrays `[N_i,2]`.
            node_count_scale: Scaling factor for node-count metadata.

        Returns:
            Tuple `(triangles_list, meta_tensor)`.
        """
        triangles_list = []
        meta_list = []

        for geom in geometries:
            geom_np = ensure_polygon_array(geom)
            triangles, cx, cy, side_len, n_nodes = self._preprocess_polygon(geom_np)
            triangles_list.append(triangles.astype(np.float32))
            meta_list.append([cx, cy, side_len, float(n_nodes) / float(node_count_scale)])

        meta_tensor = torch.tensor(meta_list, dtype=torch.float32, device=self.device)
        return triangles_list, meta_tensor

    def triangles_to_images(self, triangles_list: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert triangle arrays to MAE-compatible input channels.

        Args:
            triangles_list: List of arrays `[T_i,3,2]`.

        Returns:
            Tensor `[B,3,H,W]`.
        """
        batch_tris, lengths = pad_triangle_batch(triangles_list, device=self.device)
        mag, phase = self.codec.cft_batch(batch_tris, lengths)
        return torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)

    @torch.no_grad()
    def image_to_embedding(self, imgs: torch.Tensor) -> torch.Tensor:
        """Extract cls-token embedding from channel images.

        Args:
            imgs: Input tensor `[B,3,H,W]`.

        Returns:
            Embedding tensor `[B,D]`.
        """
        if not torch.is_tensor(imgs):
            imgs = torch.tensor(imgs, dtype=torch.float32)
        if imgs.ndim != 4 or imgs.shape[1] != 3:
            raise ValueError(f"Expected image shape [B,3,H,W], got {tuple(imgs.shape)}")

        imgs = imgs.to(device=self.device, dtype=torch.float32)
        with autocast_context(self.device, self.precision):
            encoder_features = self.encoder(imgs)
        return encoder_features[:, :, :].float()

    @torch.no_grad()
    def triangles_to_embedding(self, triangles_list: Sequence[np.ndarray]) -> torch.Tensor:
        """Extract embeddings from triangle arrays.

        Args:
            triangles_list: List of triangle arrays.

        Returns:
            Embedding tensor `[B,D]`.
        """
        imgs = self.triangles_to_images(triangles_list)
        return self.image_to_embedding(imgs)

    @torch.no_grad()
    def vector_to_embedding(
        self,
        geometries: Sequence[np.ndarray],
        append_meta: bool = True,
        node_count_scale: float = 100.0,
    ) -> torch.Tensor:
        """End-to-end embedding extraction from raw polygons.

        Args:
            geometries: Sequence of polygon arrays.
            append_meta: Whether to append `[cx, cy, side_len, node_count_scaled]`.
            node_count_scale: Node-count scale denominator.

        Returns:
            Output tensor `[B,D]` or `[B,D+4]`.
        """
        triangles_list, meta_tensor = self.vector_to_triangles(geometries, node_count_scale=node_count_scale)
        emb = self.triangles_to_embedding(triangles_list)
        if not append_meta:
            return emb
        return torch.cat([emb, meta_tensor], dim=1)


class PolyMaeReconstructionPipeline:
    """MAE reconstruction pipeline for polygon vectors.

    Args:
        weight_path: Full MAE checkpoint path.
        config_path: Config path from checkpoint directory.
        device: Runtime device string.
        precision: Runtime precision (`fp32`, `bf16`, `fp16`).
    """

    def __init__(
        self,
        weight_path: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        precision: str = "bf16",
    ):
        """Initialize reconstruction pipeline resources."""
        self.device = torch.device(device)
        self.precision = normalize_precision(precision)

        self.model, self.config = load_mae_model(
            weight_path=weight_path,
            config_path=config_path,
            device=self.device,
            precision=self.precision,
        )

        self.geom_type = str(self.config.get("geom_type", "polygon")).lower()
        self.codec = get_geometry_codec(self.geom_type, self.config, device=str(self.device))

        self.valid_h = self.codec.converter.U.shape[0] - self.codec.converter.pad_h
        self.valid_w = self.codec.converter.U.shape[1] - self.codec.converter.pad_w

    def _preprocess_polygon(self, poly: np.ndarray) -> np.ndarray:
        """Normalize and triangulate polygon input.

        Args:
            poly: Raw polygon vertices `[N,2]`.

        Returns:
            Triangles `[T,3,2]`.
        """
        poly_norm, _, _, _, _ = normalize_polygon_bbox(poly)
        return self.codec.triangulate_polygon(poly_norm)

    def triangles_to_images(self, triangles_list: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert triangle arrays to MAE input channels.

        Args:
            triangles_list: List of arrays `[T_i,3,2]`.

        Returns:
            Tensor `[B,3,H,W]`.
        """
        batch_tris, lengths = pad_triangle_batch(triangles_list, device=self.device)
        mag, phase = self.codec.cft_batch(batch_tris, lengths)
        return torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)

    @torch.no_grad()
    def reconstruct_real_imag(self, geometries: Sequence[np.ndarray], mask_ratio: float = 0.75):
        """Run MAE reconstruction and return complex spectrum parts.

        Args:
            geometries: Sequence of polygon arrays.
            mask_ratio: MAE mask ratio.

        Returns:
            Tuple `(real_part, imag_part)` with shape `[B,valid_h,valid_w]`.
        """
        mask_ratio = self.model._validate_mask_ratio(mask_ratio)
        tri_list = []
        for geom in geometries:
            geom_np = ensure_polygon_array(geom)
            tri_list.append(self._preprocess_polygon(geom_np))

        imgs = self.triangles_to_images(tri_list)

        with autocast_context(self.device, self.precision):
            pred, mask_seq = self.model(imgs, mask_ratio=mask_ratio)

        pred = pred.float()
        mask_seq = mask_seq.float()

        patch_size = int(self.config.get("patch_size", 2))
        batch_size, _, h, w = imgs.shape
        h_p, w_p = h // patch_size, w // patch_size

        pred_img = pred.reshape(batch_size, h_p, w_p, 3, patch_size, patch_size)
        pred_img = torch.einsum("nhwcpq->nchpwq", pred_img).reshape(batch_size, 3, h, w)

        mask_map = mask_seq.reshape(batch_size, h_p, w_p, 1, 1).expand(-1, -1, -1, patch_size, patch_size)
        mask_map = mask_map.permute(0, 1, 3, 2, 4).reshape(batch_size, 1, h, w)

        loss_mode = str(self.config.get("loss_mode", "mask")).lower()
        if loss_mode == "full":
            recon_imgs = pred_img
        else:
            recon_imgs = imgs * (1 - mask_map) + pred_img * mask_map
        recon_valid = recon_imgs[:, :, : self.valid_h, : self.valid_w]

        mag_valid = recon_valid[:, 0, :, :]
        cos_valid = recon_valid[:, 1, :, :]
        sin_valid = recon_valid[:, 2, :, :]

        phase_valid = torch.atan2(sin_valid, cos_valid)
        raw_mag_valid = torch.expm1(mag_valid)
        real_part = raw_mag_valid * torch.cos(phase_valid)
        imag_part = raw_mag_valid * torch.sin(phase_valid)

        return real_part, imag_part

    @torch.no_grad()
    def vector_reconstruct(self, geometries: Sequence[np.ndarray], mask_ratio: float = 0.75):
        """Alias API for reconstruction.

        Args:
            geometries: Sequence of polygon arrays.
            mask_ratio: MAE mask ratio.

        Returns:
            Tuple `(real_part, imag_part)`.
        """
        return self.reconstruct_real_imag(geometries, mask_ratio=mask_ratio)
