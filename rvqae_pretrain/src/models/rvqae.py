"""Residual vector-quantized autoencoder model for polygon pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .decoder import PolyDecoder
from .encoder import PolyEncoder
from .quantizer import QuantizerOutput, ResidualEMAVectorQuantizer


class ChannelLayerNorm2d(nn.Module):
    """Apply LayerNorm across channel dimension for each spatial location."""

    def __init__(self, num_channels: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(num_channels), eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


@dataclass
class RvqAeForwardOutput:
    """Structured RVQAE forward result."""

    recon_imgs: torch.Tensor
    vq_loss: torch.Tensor
    perplexity: torch.Tensor
    active_codes: torch.Tensor
    indices: torch.Tensor | None
    using_vq: bool
    usage_counts: torch.Tensor | None = None
    embed_sum: torch.Tensor | None = None
    restart_candidates: torch.Tensor | None = None


class PolyRvqAutoencoder(nn.Module):
    """Polygon RVQ-AE with optional conv stems, ViT bottleneck, and RVQ."""

    def __init__(
        self,
        img_size: tuple[int, int] = (31, 31),
        in_chans: int = 3,
        enc_conv_channels: Sequence[int] = (),
        enc_conv_strides: Sequence[int] = (),
        patch_size: int = 4,
        embed_dim: int = 256,
        enc_vit_depth: int = 8,
        enc_vit_head: int = 8,
        enc_vit_mlp_ratio: float = 4.0,
        dec_vit_depth: int = 8,
        dec_vit_head: int = 8,
        dec_vit_mlp_ratio: float = 4.0,
        full_res_head_channels: Sequence[int] = (),
        codebook_size: int = 8192,
        code_dim: int = 128,
        rvq_num_quantizers: int = 1,
        rvq_loss_weights: Sequence[float] | None = None,
        vq_decay: float = 0.99,
        vq_eps: float = 1.0e-5,
        vq_dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.code_dim = int(code_dim)
        self.rvq_num_quantizers = int(rvq_num_quantizers)

        self.encoder = PolyEncoder(
            img_size=self.img_size,
            in_chans=self.in_chans,
            enc_conv_channels=enc_conv_channels,
            enc_conv_strides=enc_conv_strides,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            enc_vit_depth=enc_vit_depth,
            enc_vit_head=enc_vit_head,
            enc_vit_mlp_ratio=enc_vit_mlp_ratio,
        )

        self.pre_vq_norm = ChannelLayerNorm2d(self.embed_dim)
        self.pre_vq_proj = nn.Conv2d(self.embed_dim, self.code_dim, kernel_size=1, bias=True)
        self.quantizer = ResidualEMAVectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=self.code_dim,
            num_quantizers=self.rvq_num_quantizers,
            loss_weights=list(rvq_loss_weights) if rvq_loss_weights is not None else None,
            decay=vq_decay,
            eps=vq_eps,
            dead_code_threshold=vq_dead_code_threshold,
        )
        self.post_vq_proj = nn.Conv2d(self.code_dim, self.embed_dim, kernel_size=1, bias=True)

        enc_conv_channels = tuple(int(v) for v in enc_conv_channels)
        unpatch_channels = enc_conv_channels[-1] if enc_conv_channels else self.embed_dim
        self.decoder = PolyDecoder(
            latent_dim=self.embed_dim,
            out_chans=2,
            latent_grid_size=self.encoder.latent_grid_size,
            patch_size=self.encoder.patch_size,
            unpatch_channels=unpatch_channels,
            enc_conv_channels=enc_conv_channels,
            enc_conv_strides=tuple(int(v) for v in enc_conv_strides),
            dec_vit_depth=dec_vit_depth,
            dec_vit_head=dec_vit_head,
            dec_vit_mlp_ratio=dec_vit_mlp_ratio,
            full_res_head_channels=full_res_head_channels,
        )

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode inputs into continuous latent grids `[B,D,H_lat,W_lat]`."""
        return self.encoder(imgs)

    def encode_to_code_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode inputs into pre-quantization latent code features."""
        latent = self.encode(imgs)
        return self.pre_vq_proj(self.pre_vq_norm(latent))

    @torch.no_grad()
    def tokenize(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode inputs into discrete code-index grids."""
        code_features = self.encode_to_code_features(imgs)
        indices = self.quantizer.encode_indices(code_features)
        return indices[:, 0] if self.rvq_num_quantizers == 1 else indices

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
        """Decode integer RVQ indices into reconstructed images."""
        quantized = self.quantizer.lookup_indices(indices.to(self.post_vq_proj.weight.device))
        return self.decode_from_code_features(quantized)

    def initialize_codebook(self, vectors: torch.Tensor, num_iters: int = 10) -> None:
        """Initialize the residual EMA codebooks from one set of latent vectors."""
        self.quantizer.initialize_codebook(vectors=vectors, num_iters=num_iters)

    def forward(self, imgs: torch.Tensor, use_vq: bool = True, restart_pool_size: int = 0) -> RvqAeForwardOutput:
        """Run full-image RVQAE forward pass."""
        code_features = self.encode_to_code_features(imgs)

        if use_vq:
            quantizer_output: QuantizerOutput = self.quantizer(
                code_features,
                restart_pool_size=restart_pool_size,
            )
            decoded_features = quantizer_output.quantized
            vq_loss = quantizer_output.vq_loss
            perplexity = quantizer_output.perplexity
            active_codes = quantizer_output.active_codes
            indices = quantizer_output.indices
            usage_counts = quantizer_output.usage_counts
            embed_sum = quantizer_output.embed_sum
            restart_candidates = quantizer_output.restart_candidates
        else:
            decoded_features = code_features
            device = code_features.device
            vq_loss = torch.zeros((), device=device)
            perplexity = torch.zeros((self.rvq_num_quantizers,), device=device)
            active_codes = torch.zeros((self.rvq_num_quantizers,), device=device)
            indices = None
            usage_counts = None
            embed_sum = None
            restart_candidates = None

        recon_imgs = self.decode_from_code_features(decoded_features)
        return RvqAeForwardOutput(
            recon_imgs=recon_imgs,
            vq_loss=vq_loss,
            perplexity=perplexity,
            active_codes=active_codes,
            indices=indices,
            using_vq=bool(use_vq),
            usage_counts=usage_counts,
            embed_sum=embed_sum,
            restart_candidates=restart_candidates,
        )
