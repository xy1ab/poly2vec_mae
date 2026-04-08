"""Shared normalization helpers for VQAE model modules."""

from __future__ import annotations


def group_count(num_channels: int) -> int:
    """Choose a valid GroupNorm group count for one channel width."""
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1
