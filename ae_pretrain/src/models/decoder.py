"""Decoder definitions for polygon AE pretraining.

The decoder uses a multi-stage coarse-to-fine design. Each stage is composed
of optional attention blocks followed by convolutional refinement blocks, then
an upsampling step that increases spatial resolution.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_block import Block


def _group_count(num_channels: int) -> int:
    """Choose a valid GroupNorm group count for one channel width."""
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class ConvRefineBlock(nn.Module):
    """Residual convolutional refinement block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


class GlobalAttention2D(nn.Module):
    """Apply global token attention over one feature map."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop_rate: float) -> None:
        super().__init__()
        self.block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.block(tokens)
        tokens = self.drop(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class WindowAttention2D(nn.Module):
    """Apply local window attention on one feature map."""

    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float, drop_rate: float) -> None:
        super().__init__()
        self.window_size = int(window_size)
        if self.window_size < 1:
            raise ValueError(f"`window_size` must be >= 1, got {self.window_size}")
        self.block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        ws = min(self.window_size, height, width)
        if ws == height and ws == width:
            tokens = x.flatten(2).transpose(1, 2)
            tokens = self.block(tokens)
            tokens = self.drop(tokens)
            return tokens.transpose(1, 2).reshape(batch, channels, height, width)

        pad_h = (ws - (height % ws)) % ws
        pad_w = (ws - (width % ws)) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        padded_h, padded_w = x.shape[-2:]
        num_windows_h = padded_h // ws
        num_windows_w = padded_w // ws

        windows = x.reshape(batch, channels, num_windows_h, ws, num_windows_w, ws)
        windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(batch * num_windows_h * num_windows_w, ws * ws, channels)

        windows = self.block(windows)
        windows = self.drop(windows)

        x = windows.reshape(batch, num_windows_h, num_windows_w, ws, ws, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_h, padded_w)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :height, :width]
        return x


class AttentionConvStage(nn.Module):
    """One decoder stage composed of attention + conv refinement."""

    def __init__(
        self,
        channels: int,
        attention_type: str,
        num_heads: int,
        attention_depth: int,
        conv_depth: int,
        window_size: int,
        mlp_ratio: float,
        drop_rate: float,
    ) -> None:
        super().__init__()
        attention_type = str(attention_type).lower()
        if attention_type not in {"none", "window", "global"}:
            raise ValueError(f"`attention_type` must be one of none/window/global, got {attention_type!r}")

        attn_blocks = []
        for _ in range(max(0, int(attention_depth))):
            if attention_type == "none":
                continue
            if attention_type == "window":
                attn_blocks.append(WindowAttention2D(channels, num_heads, window_size, mlp_ratio, drop_rate))
            else:
                attn_blocks.append(GlobalAttention2D(channels, num_heads, mlp_ratio, drop_rate))
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.conv_blocks = nn.ModuleList([ConvRefineBlock(channels) for _ in range(max(0, int(conv_depth)))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.attn_blocks:
            x = block(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class UpsampleProject(nn.Module):
    """Upsample one feature map and project channel width."""

    def __init__(self, in_chans: int, out_chans: int, scale: int, mode: str) -> None:
        super().__init__()
        self.scale = int(scale)
        self.mode = str(mode).lower()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_chans), out_chans)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale > 1:
            align_corners = False if self.mode in {"bilinear", "bicubic"} else None
            if align_corners is None:
                x = F.interpolate(x, scale_factor=self.scale, mode=self.mode)
            else:
                x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=align_corners)
        x = self.proj(x)
        x = self.act(self.norm(x))
        return x


class PolyDecoder(nn.Module):
    """Generic multi-stage decoder with configurable conv/attention composition."""

    def __init__(
        self,
        latent_dim: int,
        out_chans: int,
        stage_channels: Sequence[int],
        upsample_scales: Sequence[int],
        attention_type: str = "window",
        attention_heads: Sequence[int] | None = None,
        attention_depths: Sequence[int] | None = None,
        conv_depths: Sequence[int] | None = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if len(stage_channels) == 0:
            raise ValueError("`stage_channels` must not be empty")
        if len(stage_channels) != len(upsample_scales):
            raise ValueError("`stage_channels` and `upsample_scales` must have the same length")

        stage_channels = [int(ch) for ch in stage_channels]
        upsample_scales = [int(scale) for scale in upsample_scales]
        attention_heads = [int(v) for v in (attention_heads or [4] * len(stage_channels))]
        attention_depths = [int(v) for v in (attention_depths or [1] * len(stage_channels))]
        conv_depths = [int(v) for v in (conv_depths or [2] * len(stage_channels))]

        if not (len(stage_channels) == len(attention_heads) == len(attention_depths) == len(conv_depths)):
            raise ValueError("Decoder stage config lists must have matching lengths")

        self.latent_proj = nn.Conv2d(latent_dim, stage_channels[0], kernel_size=1, bias=True)
        self.stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        current_channels = stage_channels[0]
        for index, channels in enumerate(stage_channels):
            if index > 0:
                current_channels = stage_channels[index]

            self.stages.append(
                AttentionConvStage(
                    channels=current_channels,
                    attention_type=attention_type,
                    num_heads=max(1, attention_heads[index]),
                    attention_depth=attention_depths[index],
                    conv_depth=conv_depths[index],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                )
            )
            next_channels = stage_channels[min(index + 1, len(stage_channels) - 1)]
            self.upsamplers.append(
                UpsampleProject(
                    in_chans=current_channels,
                    out_chans=next_channels,
                    scale=upsample_scales[index],
                    mode=upsample_mode,
                )
            )
            current_channels = next_channels

        self.out_channels = current_channels
        self.output_head = nn.Conv2d(self.out_channels, int(out_chans), kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.latent_proj(x)
        for stage, upsample in zip(self.stages, self.upsamplers):
            x = stage(x)
            x = upsample(x)
        return self.output_head(x)
