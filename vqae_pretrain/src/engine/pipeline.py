"""Unified inference pipelines for polygon VQAE downstream usage."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..datasets.geometry_polygon import ensure_polygon_array, normalize_polygon_bbox, pad_triangle_batch
from ..datasets.registry import get_geometry_codec
from ..models.factory import load_decoder_from_components, load_pretrained_encoder, load_vqae_model
from ..utils.config import load_config_any
from ..utils.precision import autocast_context, normalize_precision


class PolyVqAePipeline:
    """Full-model VQAE inference pipeline."""

    def __init__(
        self,
        weight_path: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        precision: str = "bf16",
    ) -> None:
        self.device = torch.device(device)
        self.precision = normalize_precision(precision)

        self.model, self.config = load_vqae_model(
            weight_path=weight_path,
            config_path=config_path,
            device=self.device,
            precision=self.precision,
        )

        self.geom_type = str(self.config.get("geom_type", "polygon")).lower()
        self.codec = get_geometry_codec(self.geom_type, self.config, device=str(self.device))
        self.valid_h = self.codec.converter.U.shape[0] - self.codec.converter.pad_h
        self.valid_w = self.codec.converter.U.shape[1] - self.codec.converter.pad_w

    def triangles_to_images(self, triangles_list: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert triangle arrays to VQAE input channels."""
        batch_tris, lengths = pad_triangle_batch(triangles_list, device=self.device)
        mag, phase = self.codec.cft_batch(batch_tris, lengths)
        return torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)

    @torch.no_grad()
    def quantize_triangles(self, triangles_list: Sequence[np.ndarray]) -> torch.Tensor:
        """Encode one triangle batch into code-index grids `[B,H_lat,W_lat]`."""
        imgs = self.triangles_to_images(triangles_list)
        with autocast_context(self.device, self.precision):
            outputs = self.model(imgs, use_vq=True)
        if outputs.indices is None:
            raise RuntimeError("VQAE inference expected indices but received None.")
        return outputs.indices.long().cpu()

    @torch.no_grad()
    def reconstruct_triangles(
        self,
        triangles_list: Sequence[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct one triangle batch into real/imaginary maps and VQ stats."""
        imgs = self.triangles_to_images(triangles_list)
        with autocast_context(self.device, self.precision):
            outputs = self.model(imgs, use_vq=True)

        recon = outputs.recon_imgs.float()
        mag_valid = recon[:, 0, : self.valid_h, : self.valid_w]
        cos_valid = recon[:, 1, : self.valid_h, : self.valid_w]
        sin_valid = recon[:, 2, : self.valid_h, : self.valid_w]
        phase_valid = torch.atan2(sin_valid, cos_valid)
        raw_mag_valid = torch.expm1(mag_valid)
        real_part = raw_mag_valid * torch.cos(phase_valid)
        imag_part = raw_mag_valid * torch.sin(phase_valid)
        return real_part, imag_part, outputs.perplexity.float(), outputs.active_codes.float()


class PolyVqDecodePipeline:
    """Decoder-only VQ pipeline for code-index decoding."""

    def __init__(
        self,
        decoder_path: str,
        quantizer_path: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        precision: str = "bf16",
    ) -> None:
        self.device = torch.device(device)
        self.precision = normalize_precision(precision)
        self.model, self.config = load_decoder_from_components(
            decoder_path=decoder_path,
            quantizer_path=quantizer_path,
            config_path=config_path,
            device=self.device,
            precision=self.precision,
        )
        self.geom_type = str(self.config.get("geom_type", "polygon")).lower()
        self.codec = get_geometry_codec(self.geom_type, self.config, device=str(self.device))
        self.valid_h = self.codec.converter.U.shape[0] - self.codec.converter.pad_h
        self.valid_w = self.codec.converter.U.shape[1] - self.codec.converter.pad_w

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode code-index grids into real/imaginary frequency maps."""
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)
        indices = indices.to(self.device, dtype=torch.long)
        with autocast_context(self.device, self.precision):
            recon = self.model.decode_indices(indices)
        recon = recon.float()
        mag_valid = recon[:, 0, : self.valid_h, : self.valid_w]
        cos_valid = recon[:, 1, : self.valid_h, : self.valid_w]
        sin_valid = recon[:, 2, : self.valid_h, : self.valid_w]
        phase_valid = torch.atan2(sin_valid, cos_valid)
        raw_mag_valid = torch.expm1(mag_valid)
        real_part = raw_mag_valid * torch.cos(phase_valid)
        imag_part = raw_mag_valid * torch.sin(phase_valid)
        return real_part.cpu(), imag_part.cpu()


class PolyEncoderPipeline:
    """Encoder inference pipeline for continuous encoder embedding extraction."""

    def __init__(
        self,
        weight_path: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        precision: str = "bf16",
    ) -> None:
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

    def triangles_to_images(self, triangles_list: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert triangle arrays to encoder input channels."""
        batch_tris, lengths = pad_triangle_batch(triangles_list, device=self.device)
        mag, phase = self.codec.cft_batch(batch_tris, lengths)
        return torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)

    @torch.no_grad()
    def triangles_to_embedding(self, triangles_list: Sequence[np.ndarray]) -> torch.Tensor:
        """Extract one dense global embedding from triangle arrays."""
        imgs = self.triangles_to_images(triangles_list)
        with autocast_context(self.device, self.precision):
            latent = self.encoder(imgs)
        return latent.mean(dim=(2, 3)).float().cpu()
