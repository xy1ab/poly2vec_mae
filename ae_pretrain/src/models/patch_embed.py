"""Patch embedding layer definitions.

This module provides a lightweight image-to-token projection used by the ViT
encoder in AE pretraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Project an image tensor into patch tokens.

    Args:
        img_size: Input image size as `(H, W)`.
        patch_size: Patch edge length.
        in_chans: Number of input channels.
        embed_dim: Token embedding dimension.
    """

    def __init__(self, img_size: tuple[int, int], patch_size: int, in_chans: int, embed_dim: int):
        """Initialize patch embedding projection."""
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed image tensor into patch tokens.

        Args:
            x: Input tensor with shape `[B, C, H, W]`.

        Returns:
            Token tensor with shape `[B, L, D]`.
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
