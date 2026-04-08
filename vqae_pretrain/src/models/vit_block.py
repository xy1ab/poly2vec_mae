"""Minimal ViT transformer blocks used by polygon VQAE encoder/decoder.

This local implementation avoids importing `timm`, which can transitively pull
in `torchvision` features that are unrelated to this project and may be
incompatible with the active runtime environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Multi-head self-attention with timm-compatible parameter names."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"`dim` must be divisible by `num_heads`, got dim={dim}, num_heads={num_heads}")

        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, token_count, dim = x.shape
        qkv = self.qkv(x).reshape(batch, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch, token_count, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """Feed-forward network with timm-compatible layer names."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(hidden_features or in_features)
        out = int(out_features or in_features)

        self.fc1 = nn.Linear(in_features, hidden)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, out)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    """Transformer encoder block compatible with stored checkpoint keys."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=act_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
