"""Sin-cos positional embedding utilities.

These helpers are used by both encoder and decoder modules to build deterministic
2D positional embeddings without trainable position parameters.
"""

from __future__ import annotations

import numpy as np


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int], cls_token: bool = False) -> np.ndarray:
    """Build 2D sin-cos positional embedding for a HxW latent token grid.

    Args:
        embed_dim: Output embedding dimension.
        grid_size: Patch grid size `(grid_h, grid_w)`.
        cls_token: Whether to prepend one cls-token embedding row.

    Returns:
        Positional embedding array with shape `[H*W, D]` or `[1+H*W, D]`.
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim], dtype=np.float32), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Build 2D sin-cos embedding from a prebuilt 2-channel position grid.

    Args:
        embed_dim: Output embedding dimension (must be even).
        grid: Position grid with shape `[2,1,H,W]`.

    Returns:
        Embedding array with shape `[H*W, D]`.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Build 1D sin-cos embedding from a flattened position vector.

    Args:
        embed_dim: Output embedding dimension (must be even).
        pos: Position array.

    Returns:
        Embedding array with shape `[M, D]`.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= float(embed_dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)
