"""Encoder definitions for polygon AE pretraining.

This encoder follows the project-specific design:
1) A convolutional stem performs local feature extraction and spatial downsampling.
2) A ViT neck processes the latent token grid with absolute positional embeddings.
3) The final output is a dense latent grid `[B, D, H_lat, W_lat]` suitable for
   compression-oriented reconstruction, instead of a cls-token representation.
"""

from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Sequence

import torch
import torch.nn as nn

from .pos_embed import get_2d_sincos_pos_embed
from .vit_block import Block


def _group_count(num_channels: int) -> int:
    """Choose a valid GroupNorm group count for one channel width."""
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class ConvStemStage(nn.Module):
    """One convolutional downsampling stage."""

    def __init__(self, in_chans: int, out_chans: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_chans), out_chans)
        self.act1 = nn.GELU()
        self.refine = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_chans), out_chans)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.act1(self.norm1(x))
        x = self.refine(x)
        x = self.act2(self.norm2(x))
        return x


class ConvStem(nn.Module):
    """Hierarchical convolutional stem for local texture extraction."""

    def __init__(self, in_chans: int, stage_channels: Sequence[int], stage_strides: Sequence[int]) -> None:
        super().__init__()
        if len(stage_channels) == 0:
            raise ValueError("`stage_channels` must not be empty")
        if len(stage_channels) != len(stage_strides):
            raise ValueError("`stage_channels` and `stage_strides` must have the same length")

        stages = []
        prev_channels = int(in_chans)
        for out_channels, stride in zip(stage_channels, stage_strides):
            stages.append(ConvStemStage(prev_channels, int(out_channels), int(stride)))
            prev_channels = int(out_channels)

        self.stages = nn.ModuleList(stages)
        self.out_channels = int(stage_channels[-1])
        self.output_stride = int(reduce(mul, (int(s) for s in stage_strides), 1))
        self.stage_channels = tuple(int(c) for c in stage_channels)
        self.stage_strides = tuple(int(s) for s in stage_strides)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class PolyEncoder(nn.Module):
    """Conv+ViT encoder for frequency-domain polygon images.

    Args:
        img_size: Input image size `(H, W)`.
        in_chans: Number of channels.
        stem_channels: Convolutional stem channels.
        stem_strides: Convolutional stage strides.
        embed_dim: Latent grid channel dimension after ViT neck.
        depth: ViT neck depth.
        num_heads: ViT neck attention head count.
        mlp_ratio: ViT MLP expansion ratio.
        drop_rate: Token dropout probability.
        drop_path_rate: Reserved for future use. Currently kept for config/API
            compatibility and ignored by the local lightweight ViT blocks.
    """

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
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        del drop_path_rate

        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.in_chans = int(in_chans)
        self.stem = ConvStem(self.in_chans, stem_channels, stem_strides)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.drop_rate = float(drop_rate)

        latent_h = self.img_size[0] // self.stem.output_stride
        latent_w = self.img_size[1] // self.stem.output_stride
        if latent_h < 1 or latent_w < 1:
            raise ValueError(
                f"Invalid latent grid from img_size={self.img_size} and stem strides={self.stem.stage_strides}"
            )
        if latent_h * self.stem.output_stride != self.img_size[0] or latent_w * self.stem.output_stride != self.img_size[1]:
            raise ValueError(
                f"`img_size` must be divisible by total stem stride={self.stem.output_stride}, got {self.img_size}"
            )

        self.latent_grid_size = (latent_h, latent_w)
        self.num_patches = latent_h * latent_w
        self.latent_stride = self.stem.output_stride

        self.token_proj = nn.Conv2d(self.stem.out_channels, self.embed_dim, kernel_size=1, bias=True)
        self.token_drop = nn.Dropout(self.drop_rate)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize positional embeddings and module weights."""
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            grid_size=self.latent_grid_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear/layernorm/conv layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one image batch into a latent feature grid."""
        x = self.stem(x)
        x = self.token_proj(x)
        batch_size, channels, height, width = x.shape

        tokens = x.flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.num_patches:
            raise RuntimeError(
                f"Unexpected latent token count {tokens.shape[1]} != {self.num_patches}. "
                f"Got feature map {height}x{width} for img_size={self.img_size}"
            )

        tokens = self.token_drop(tokens + self.pos_embed)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        latent = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full inputs into a latent grid `[B, D, H_lat, W_lat]`."""
        return self.forward_features(x)

    def extra_repr(self) -> str:
        return (
            f"img_size={self.img_size}, latent_grid_size={self.latent_grid_size}, "
            f"latent_stride={self.latent_stride}, embed_dim={self.embed_dim}, depth={self.depth}, "
            f"num_heads={self.num_heads}"
        )
