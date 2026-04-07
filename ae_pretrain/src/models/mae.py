"""Autoencoder model wrapper for polygon frequency-domain pretraining.

The historical file name is kept for compatibility with the surrounding
project structure, but the implementation is a true autoencoder. The model uses:
1) Conv+ViT encoder.
2) Multi-stage attention+conv decoder.
3) Full-image reconstruction without masking.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .decoder import PolyDecoder
from .encoder import PolyEncoder


class PolyAutoencoder(nn.Module):
    """Polygon frequency-domain autoencoder."""

    def __init__(
        self,
        img_size: tuple[int, int] = (31, 31),
        patch_size: int = 2,
        in_chans: int = 3,
        stem_channels: Sequence[int] = (64, 128, 256),
        stem_strides: Sequence[int] = (2, 2, 2),
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        decoder_stage_channels: Sequence[int] = (256, 192, 128),
        decoder_attention_type: str = "window",
        decoder_attention_heads: Sequence[int] = (8, 4, 4),
        decoder_attention_depths: Sequence[int] = (1, 1, 0),
        decoder_conv_depths: Sequence[int] = (2, 2, 2),
        decoder_window_size: int = 8,
        decoder_upsample_mode: str = "bilinear",
        decoder_mlp_ratio: float = 4.0,
        decoder_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)

        self.encoder = PolyEncoder(
            img_size=self.img_size,
            in_chans=self.in_chans,
            stem_channels=stem_channels,
            stem_strides=stem_strides,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        upsample_scales = list(self.encoder.stem.stage_strides[::-1])
        if len(decoder_stage_channels) != len(upsample_scales):
            raise ValueError(
                "`decoder_stage_channels` must have the same length as the number of encoder stem stages "
                f"({len(upsample_scales)}), got {len(decoder_stage_channels)}"
            )

        self.decoder = PolyDecoder(
            latent_dim=embed_dim,
            out_chans=self.in_chans,
            stage_channels=decoder_stage_channels,
            upsample_scales=upsample_scales,
            attention_type=decoder_attention_type,
            attention_heads=decoder_attention_heads,
            attention_depths=decoder_attention_depths,
            conv_depths=decoder_conv_depths,
            window_size=decoder_window_size,
            mlp_ratio=decoder_mlp_ratio,
            drop_rate=decoder_drop_rate,
            upsample_mode=decoder_upsample_mode,
        )

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode input images into a latent grid."""
        return self.encoder(imgs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode one latent grid into reconstructed images."""
        recon = self.decoder(latent)
        if recon.shape[-2:] != self.img_size:
            raise RuntimeError(
                f"Decoder output spatial size {tuple(recon.shape[-2:])} does not match configured img_size={self.img_size}"
            )
        return recon

    def forward_image(self, imgs: torch.Tensor) -> torch.Tensor:
        """Run the full AE and return reconstructed images `[B,C,H,W]`."""
        latent = self.encode(imgs)
        return self.decode(latent)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert reconstructed images into flattened patch predictions."""
        batch, channels, height, width = imgs.shape
        patch_size = self.patch_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Image size {height}x{width} must be divisible by patch_size={patch_size}"
            )
        h_patch, w_patch = height // patch_size, width // patch_size
        x = imgs.reshape(batch, channels, h_patch, patch_size, w_patch, patch_size)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(batch, h_patch * w_patch, channels * patch_size**2)
        return x

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Return reconstructed patch tokens `[B,L,C*p*p]`."""
        recon_imgs = self.forward_image(imgs)
        return self.patchify(recon_imgs)


# Legacy alias kept for copied helper code during migration.
MaskedAutoencoderViTPoly = PolyAutoencoder
