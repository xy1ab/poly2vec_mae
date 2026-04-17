"""Encoder definitions for polygon RVQAE pretraining."""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Sequence

import torch
import torch.nn as nn

from .norm_utils import group_count
from .pos_embed import get_2d_sincos_pos_embed
from .vit_block import Block


class ConvStemStage(nn.Module):
    """One optional convolutional downsampling stage."""

    def __init__(self, in_chans: int, out_chans: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(group_count(out_chans), out_chans)
        self.act1 = nn.GELU()
        self.refine = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(group_count(out_chans), out_chans)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.act1(self.norm1(x))
        x = self.refine(x)
        x = self.act2(self.norm2(x))
        return x


class ConvStem(nn.Module):
    """Optional hierarchical convolutional stem."""

    def __init__(self, in_chans: int, stage_channels: Sequence[int], stage_strides: Sequence[int]) -> None:
        super().__init__()
        if len(stage_channels) != len(stage_strides):
            raise ValueError("`enc_conv_channels` and `enc_conv_strides` must have the same length")

        stages = []
        prev_channels = int(in_chans)
        for out_channels, stride in zip(stage_channels, stage_strides):
            stages.append(ConvStemStage(prev_channels, int(out_channels), int(stride)))
            prev_channels = int(out_channels)

        self.stages = nn.ModuleList(stages)
        self.out_channels = prev_channels
        self.output_stride = int(reduce(mul, (int(s) for s in stage_strides), 1))
        self.stage_channels = tuple(int(c) for c in stage_channels)
        self.stage_strides = tuple(int(s) for s in stage_strides)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class LinearPatchify(nn.Module):
    """Patchify one feature map with reshape plus linear projection."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"`patch_size` must be > 0, got {patch_size}")
        self.proj = nn.Linear(self.in_chans * self.patch_size * self.patch_size, self.embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        batch, channels, height, width = x.shape
        if channels != self.in_chans:
            raise ValueError(f"Patchify channel mismatch: expected {self.in_chans}, got {channels}")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"Feature map {height}x{width} is not divisible by patch_size={self.patch_size}")

        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        patches = x.reshape(batch, channels, grid_h, self.patch_size, grid_w, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, grid_h * grid_w, -1)
        return self.proj(patches), (grid_h, grid_w)


class PolyEncoder(nn.Module):
    """Optional conv stem + linear patchify + ViT encoder."""

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
    ) -> None:
        super().__init__()

        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.in_chans = int(in_chans)
        self.stem = ConvStem(self.in_chans, enc_conv_channels, enc_conv_strides)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(enc_vit_depth)
        self.num_heads = int(enc_vit_head)
        self.mlp_ratio = float(enc_vit_mlp_ratio)

        self.latent_stride = int(self.stem.output_stride * self.patch_size)
        latent_h = self.img_size[0] // self.latent_stride
        latent_w = self.img_size[1] // self.latent_stride
        if latent_h < 1 or latent_w < 1:
            raise ValueError(
                f"Invalid latent grid from img_size={self.img_size}, "
                f"conv strides={self.stem.stage_strides}, patch_size={self.patch_size}"
            )
        if latent_h * self.latent_stride != self.img_size[0] or latent_w * self.latent_stride != self.img_size[1]:
            raise ValueError(f"`img_size` must be divisible by latent_stride={self.latent_stride}, got {self.img_size}")

        self.latent_grid_size = (latent_h, latent_w)
        self.num_latent_tokens = latent_h * latent_w

        self.patchify = LinearPatchify(self.stem.out_channels, self.embed_dim, self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.embed_dim), requires_grad=False)
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

    @property
    def stem_output_channels(self) -> int:
        """Feature channels after the optional convolutional stem."""
        return int(self.stem.out_channels)

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
        tokens, (height, width) = self.patchify(x)
        if tokens.shape[1] != self.num_latent_tokens:
            raise RuntimeError(
                f"Unexpected latent token count {tokens.shape[1]} != {self.num_latent_tokens}. "
                f"Got grid {height}x{width} for img_size={self.img_size}"
            )

        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        latent = tokens.transpose(1, 2).reshape(x.shape[0], self.embed_dim, height, width)
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full inputs into a latent grid `[B,D,H_lat,W_lat]`."""
        return self.forward_features(x)

    def extra_repr(self) -> str:
        return (
            f"img_size={self.img_size}, latent_grid_size={self.latent_grid_size}, "
            f"latent_stride={self.latent_stride}, embed_dim={self.embed_dim}, depth={self.depth}, "
            f"num_heads={self.num_heads}"
        )
