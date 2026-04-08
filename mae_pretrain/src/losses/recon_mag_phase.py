"""Magnitude/phase reconstruction losses for MAE pretraining.

This module implements the dual magnitude loss (base + high-frequency penalty)
and phase loss used in polygon MAE training.
"""

from __future__ import annotations

import torch


def compute_mag_phase_losses(
    pred: torch.Tensor,
    target_patches: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    freq_span_patches: torch.Tensor,
    weight_mag_hf: float,
    loss_mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute magnitude and phase losses on patch tokens.

    Args:
        pred: MAE predicted patches `[B,L,3*p*p]`.
        target_patches: Ground-truth patches `[B,L,3*p*p]`.
        mask: MAE mask `[B,L]` where 1 means masked.
        patch_size: Patch edge length.
        freq_span_patches: Frequency-area weight map `[1,L,p*p]`.
        weight_mag_hf: Weight for high-frequency magnitude penalty.
        loss_mode: Loss aggregation mode. `"mask"` uses MAE masked-only loss,
            while `"full"` computes reconstruction loss over all patch tokens.

    Returns:
        Tuple `(loss_mag, loss_phase)`.
    """
    if loss_mode not in {"mask", "full"}:
        raise ValueError(f"`loss_mode` must be 'mask' or 'full', got {loss_mode!r}")

    p2 = patch_size**2

    target_mag = target_patches[:, :, :p2]
    target_cos = target_patches[:, :, p2 : 2 * p2]
    target_sin = target_patches[:, :, 2 * p2 :]

    pred_mag = pred[:, :, :p2]
    pred_cos = pred[:, :, p2 : 2 * p2]
    pred_sin = pred[:, :, 2 * p2 :]

    patch_weights = mask if loss_mode == "mask" else torch.ones_like(mask)
    weight_denom = patch_weights.sum() + 1e-8

    mag_l1 = torch.abs(pred_mag - target_mag)
    loss_mag_base = (mag_l1.mean(dim=-1) * patch_weights).sum() / weight_denom

    weighted_mag = mag_l1 * freq_span_patches
    loss_mag_penalty = (weighted_mag.mean(dim=-1) * patch_weights).sum() / weight_denom
    loss_mag = loss_mag_base + weight_mag_hf * loss_mag_penalty

    phase_l1 = torch.abs(pred_cos - target_cos) + torch.abs(pred_sin - target_sin)
    loss_phase = (phase_l1.mean(dim=-1) * patch_weights).sum() / weight_denom

    return loss_mag, loss_phase
