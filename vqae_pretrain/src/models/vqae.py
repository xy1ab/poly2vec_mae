"""Vector-quantized autoencoder model for polygon frequency-domain pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .decoder import PolyDecoder
from .encoder import PolyEncoder
from .quantizer import EMAVectorQuantizer, QuantizerOutput


class ChannelLayerNorm2d(nn.Module):
    """Apply LayerNorm across channel dimension for each spatial location."""

    def __init__(self, num_channels: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(num_channels), eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


@dataclass
class VqAeForwardOutput:
    """Structured VQAE forward result."""

    recon_imgs: torch.Tensor
    vq_loss: torch.Tensor
    perplexity: torch.Tensor
    active_codes: torch.Tensor
    indices: torch.Tensor | None
    using_vq: bool


class PolyVqAutoencoder(nn.Module):
    """Polygon VQ-AE with conv+ViT encoder and attention+conv decoder."""

    def __init__(
        self,
        img_size: tuple[int, int] = (31, 31),
        in_chans: int = 3,
        stem_channels: Sequence[int] = (64, 128, 256),
        stem_strides: Sequence[int] = (2, 2, 2),
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        decoder_stage_channels: Sequence[int] = (256, 192, 128),
        decoder_attention_type: str = "window",
        decoder_attention_heads: Sequence[int] = (8, 4, 4),
        decoder_attention_depths: Sequence[int] = (1, 1, 0),
        decoder_conv_depths: Sequence[int] = (2, 2, 2),
        decoder_window_size: int = 8,
        decoder_upsample_mode: str = "bilinear",
        decoder_mlp_ratio: float = 4.0,
        decoder_drop_rate: float = 0.0,
        codebook_size: int = 8192,
        code_dim: int = 128,
        vq_decay: float = 0.99,
        vq_eps: float = 1.0e-5,
        vq_dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.code_dim = int(code_dim)

        self.encoder = PolyEncoder(
            img_size=self.img_size,
            in_chans=self.in_chans,
            stem_channels=stem_channels,
            stem_strides=stem_strides,
            embed_dim=self.embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )

        upsample_scales = list(self.encoder.stem.stage_strides[::-1])
        if len(decoder_stage_channels) != len(upsample_scales):
            raise ValueError(
                "`decoder_stage_channels` must have the same length as the number of encoder stem stages "
                f"({len(upsample_scales)}), got {len(decoder_stage_channels)}"
            )

        self.pre_vq_norm = ChannelLayerNorm2d(self.embed_dim)
        self.pre_vq_proj = nn.Conv2d(self.embed_dim, self.code_dim, kernel_size=1, bias=True)
        self.quantizer = EMAVectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=self.code_dim,
            decay=vq_decay,
            eps=vq_eps,
            dead_code_threshold=vq_dead_code_threshold,
        )
        self.post_vq_proj = nn.Conv2d(self.code_dim, self.embed_dim, kernel_size=1, bias=True)

        self.decoder = PolyDecoder(
            latent_dim=self.embed_dim,
            out_chans=2,
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
        """Encode inputs into continuous latent grids `[B,D,H_lat,W_lat]`."""
        return self.encoder(imgs)

    def encode_to_code_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode inputs into pre-quantization latent code features."""
        latent = self.encode(imgs)
        return self.pre_vq_proj(self.pre_vq_norm(latent))

    def decode_from_code_features(self, code_features: torch.Tensor) -> torch.Tensor:
        """Decode code features into physically constrained frequency images."""
        decoder_raw = self.decoder(self.post_vq_proj(code_features))
        mag_raw = decoder_raw[:, 0:1]
        phase_raw = decoder_raw[:, 1:2]
        mag = torch.nn.functional.softplus(mag_raw)
        cos = torch.cos(phase_raw)
        sin = torch.sin(phase_raw)
        return torch.cat([mag, cos, sin], dim=1)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode integer code indices `[B,H,W]` into reconstructed images."""
        quantized = self.quantizer.lookup_indices(indices.to(self.post_vq_proj.weight.device))
        return self.decode_from_code_features(quantized)

    def initialize_codebook(self, vectors: torch.Tensor, num_iters: int = 10) -> None:
        """Initialize the EMA codebook from one set of latent vectors."""
        self.quantizer.initialize_codebook(vectors=vectors, num_iters=num_iters)

    def forward(self, imgs: torch.Tensor, use_vq: bool = True) -> VqAeForwardOutput:
        """Run full-image VQAE forward pass.

        During AE warmup, `use_vq=False` bypasses the quantizer while keeping the
        same projection path (`pre_vq_norm -> pre_vq_proj -> post_vq_proj`).
        """
        code_features = self.encode_to_code_features(imgs)

        if use_vq:
            quantizer_output: QuantizerOutput = self.quantizer(code_features)
            decoded_features = quantizer_output.quantized
            vq_loss = quantizer_output.vq_loss
            perplexity = quantizer_output.perplexity
            active_codes = quantizer_output.active_codes
            indices = quantizer_output.indices
        else:
            decoded_features = code_features
            device = code_features.device
            vq_loss = torch.zeros((), device=device)
            perplexity = torch.zeros((), device=device)
            active_codes = torch.zeros((), device=device)
            indices = None

        recon_imgs = self.decode_from_code_features(decoded_features)
        return VqAeForwardOutput(
            recon_imgs=recon_imgs,
            vq_loss=vq_loss,
            perplexity=perplexity,
            active_codes=active_codes,
            indices=indices,
            using_vq=bool(use_vq),
        )
