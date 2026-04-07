"""Batch collation helpers for variable-length triangle sets.

This module handles padding logic required by CFT batch processing where each
sample may contain a different number of triangles.
"""

from __future__ import annotations

import torch


def ae_collate_fn(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length triangle tensors to the max length in the batch.

    Args:
        batch: List of tensors each shaped `[T_i, 3, 2]`.

    Returns:
        Tuple `(padded_batch, lengths)` where:
        - `padded_batch` has shape `[B, max_T, 3, 2]`
        - `lengths` has shape `[B]`
    """
    lengths = torch.tensor([item.shape[0] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded_batch = torch.zeros((len(batch), max_len, 3, 2), dtype=torch.float32)
    for idx, item in enumerate(batch):
        padded_batch[idx, : lengths[idx], :, :] = item

    return padded_batch, lengths


# Legacy alias kept for copied code paths during migration.
mae_collate_fn = ae_collate_fn
