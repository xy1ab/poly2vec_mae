"""Magnitude/phase reconstruction losses for AE pretraining.

This module implements the dual magnitude loss (base + high-frequency penalty)
and phase loss used in polygon AE training.
"""

from __future__ import annotations

import torch


def compute_mag_phase_losses(
    pred: torch.Tensor,
    target_patches: torch.Tensor,
    patch_size: int,
    freq_span_patches: torch.Tensor,
    weight_mag_hf: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute AE magnitude and phase losses on all patch tokens.

    Args:
        pred: AE predicted patches `[B,L,3*p*p]`.
        target_patches: Ground-truth patches `[B,L,3*p*p]`.
        patch_size: Patch edge length.
        freq_span_patches: Frequency-area weight map `[1,L,p*p]`.
        weight_mag_hf: Weight for high-frequency magnitude penalty.

    Returns:
        Tuple `(loss_mag, loss_phase)`.
    """
    p2 = patch_size**2

    target_mag = target_patches[:, :, :p2]
    target_cos = target_patches[:, :, p2 : 2 * p2]
    target_sin = target_patches[:, :, 2 * p2 :]

    pred_mag = pred[:, :, :p2]
    pred_cos = pred[:, :, p2 : 2 * p2]
    pred_sin = pred[:, :, 2 * p2 :]

    patch_weights = torch.ones(pred.shape[:2], device=pred.device, dtype=pred.dtype)
    weight_denom = patch_weights.sum() + 1e-8

    mag_l1 = torch.abs(pred_mag - target_mag)
    loss_mag_base = (mag_l1.mean(dim=-1) * patch_weights).sum() / weight_denom

    weighted_mag = mag_l1 * freq_span_patches
    loss_mag_penalty = (weighted_mag.mean(dim=-1) * patch_weights).sum() / weight_denom
    loss_mag = loss_mag_base + weight_mag_hf * loss_mag_penalty

    phase_l1 = torch.abs(pred_cos - target_cos) + torch.abs(pred_sin - target_sin)
    loss_phase = (phase_l1.mean(dim=-1) * patch_weights).sum() / weight_denom

    return loss_mag, loss_phase
