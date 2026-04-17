"""Decoder definitions for polygon RVQAE pretraining."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm_utils import group_count
from .pos_embed import get_2d_sincos_pos_embed
from .vit_block import Block


class ConvRefineBlock(nn.Module):
    """Residual convolutional refinement block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(channels), channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(group_count(channels), channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


class LinearUnpatchify(nn.Module):
    """Unpatchify latent tokens with a linear projection."""

    def __init__(self, embed_dim: int, out_chans: int, patch_size: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.out_chans = int(out_chans)
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"`patch_size` must be > 0, got {patch_size}")
        self.proj = nn.Linear(self.embed_dim, self.out_chans * self.patch_size * self.patch_size)

    def forward(self, tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        batch, token_count, channels = tokens.shape
        height, width = int(grid_size[0]), int(grid_size[1])
        if token_count != height * width:
            raise ValueError(f"Token count {token_count} does not match grid {height}x{width}")
        if channels != self.embed_dim:
            raise ValueError(f"Unpatch channel mismatch: expected {self.embed_dim}, got {channels}")

        patches = self.proj(tokens)
        patches = patches.reshape(batch, height, width, self.out_chans, self.patch_size, self.patch_size)
        feature = patches.permute(0, 3, 1, 4, 2, 5)
        return feature.reshape(batch, self.out_chans, height * self.patch_size, width * self.patch_size)


class UpsampleResidualStage(nn.Module):
    """One symmetric upsampling projection with residual refinement."""

    def __init__(self, in_chans: int, out_chans: int, scale: int) -> None:
        super().__init__()
        self.scale = int(scale)
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.GroupNorm(group_count(out_chans), out_chans)
        self.act = nn.GELU()
        self.refine = ConvRefineBlock(out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        x = self.act(self.norm(self.proj(x)))
        return self.refine(x)


class SymmetricUpsampleStem(nn.Module):
    """Upsample with stages inferred from the encoder conv stem."""

    def __init__(self, in_chans: int, enc_conv_channels: Sequence[int], enc_conv_strides: Sequence[int]) -> None:
        super().__init__()
        if len(enc_conv_channels) != len(enc_conv_strides):
            raise ValueError("`enc_conv_channels` and `enc_conv_strides` must have the same length")

        rev_channels = [int(ch) for ch in enc_conv_channels[::-1]]
        rev_strides = [int(stride) for stride in enc_conv_strides[::-1]]
        stages = []
        current_channels = int(in_chans)
        for index, stride in enumerate(rev_strides):
            next_channels = rev_channels[index + 1] if index + 1 < len(rev_channels) else rev_channels[-1]
            stages.append(UpsampleResidualStage(current_channels, next_channels, stride))
            current_channels = next_channels

        self.stages = nn.ModuleList(stages)
        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class PixelLinearHead(nn.Module):
    """Per-pixel linear output head used by the strict pure-ViT path."""

    def __init__(self, in_chans: int, out_chans: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(in_chans), int(out_chans))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.proj(x)
        return x.permute(0, 3, 1, 2).contiguous()


class FullResResidualHead(nn.Module):
    """Full-resolution residual conv head with configurable channel list."""

    def __init__(self, in_chans: int, channels: Sequence[int], out_chans: int) -> None:
        super().__init__()
        channels = [int(ch) for ch in channels]
        if not channels:
            self.net = PixelLinearHead(in_chans, out_chans)
            return

        layers: list[nn.Module] = []
        current_channels = int(in_chans)
        for out_channels in channels:
            if current_channels != out_channels:
                layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=True))
            layers.append(ConvRefineBlock(out_channels))
            current_channels = out_channels
        layers.append(nn.Conv2d(current_channels, int(out_chans), kernel_size=3, stride=1, padding=1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolyDecoder(nn.Module):
    """ViT decoder with linear unpatchify and symmetric optional conv upsampling."""

    def __init__(
        self,
        latent_dim: int,
        out_chans: int,
        latent_grid_size: tuple[int, int],
        patch_size: int,
        unpatch_channels: int,
        enc_conv_channels: Sequence[int],
        enc_conv_strides: Sequence[int],
        dec_vit_depth: int = 8,
        dec_vit_head: int = 8,
        dec_vit_mlp_ratio: float = 4.0,
        full_res_head_channels: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.latent_grid_size = (int(latent_grid_size[0]), int(latent_grid_size[1]))
        self.num_latent_tokens = self.latent_grid_size[0] * self.latent_grid_size[1]
        self.depth = int(dec_vit_depth)
        self.num_heads = int(dec_vit_head)
        self.mlp_ratio = float(dec_vit_mlp_ratio)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.latent_dim), requires_grad=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.latent_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.latent_dim)
        self.unpatch = LinearUnpatchify(self.latent_dim, int(unpatch_channels), int(patch_size))
        self.upsample_stem = SymmetricUpsampleStem(int(unpatch_channels), enc_conv_channels, enc_conv_strides)
        self.out_channels = int(self.upsample_stem.out_channels)
        self.output_head = FullResResidualHead(self.out_channels, full_res_head_channels, out_chans)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize positional embeddings and module weights."""
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.latent_dim,
            grid_size=self.latent_grid_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)
        self._init_output_projection()

    def _init_weights(self, module: nn.Module) -> None:
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

    def _init_output_projection(self) -> None:
        """Keep raw magnitude/phase predictions near a stable initial range."""
        projections = [
            module
            for module in self.output_head.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        if not projections:
            return

        output_proj = projections[-1]
        nn.init.normal_(output_proj.weight, mean=0.0, std=1.0e-3)
        if output_proj.bias is None:
            return

        nn.init.constant_(output_proj.bias, 0.0)
        if output_proj.bias.numel() >= 1:
            target_mag = 0.05
            with torch.no_grad():
                output_proj.bias[0].fill_(math.log(math.expm1(target_mag)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.latent_dim:
            raise ValueError(f"Decoder channel mismatch: expected {self.latent_dim}, got {channels}")
        if (height, width) != self.latent_grid_size:
            raise ValueError(f"Decoder grid mismatch: expected {self.latent_grid_size}, got {(height, width)}")

        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        x = self.unpatch(tokens, self.latent_grid_size)
        x = self.upsample_stem(x)
        return self.output_head(x)
