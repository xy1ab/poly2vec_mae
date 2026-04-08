"""Magnitude/phase reconstruction losses for VQAE pretraining.

This module implements the dual magnitude loss (base + high-frequency penalty)
and phase loss used in polygon VQAE training.
"""

from __future__ import annotations

import torch


def compute_mag_phase_losses(
    pred_imgs: torch.Tensor,
    target_imgs: torch.Tensor,
    freq_span_map: torch.Tensor,
    valid_mask: torch.Tensor,
    weight_mag_hf: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute VQAE magnitude and phase losses on full frequency images.

    Args:
        pred_imgs: VQAE reconstructed frequency images `[B,3,H,W]`.
        target_imgs: Ground-truth frequency images `[B,3,H,W]`.
        freq_span_map: Frequency-area weight map `[1,1,H,W]`.
        valid_mask: Binary valid-region mask `[1,1,H,W]`.
        weight_mag_hf: Weight for high-frequency magnitude penalty.

    Returns:
        Tuple `(loss_mag, loss_phase)`.
    """
    target_mag = target_imgs[:, 0:1]
    target_cos = target_imgs[:, 1:2]
    target_sin = target_imgs[:, 2:3]

    pred_mag = pred_imgs[:, 0:1]
    pred_cos = pred_imgs[:, 1:2]
    pred_sin = pred_imgs[:, 2:3]

    valid_mask = valid_mask.to(dtype=pred_imgs.dtype, device=pred_imgs.device)
    valid_mask_expanded = valid_mask.expand_as(pred_mag)
    valid_denom = valid_mask_expanded.sum().clamp_min(1.0)

    mag_l1 = torch.abs(pred_mag - target_mag)
    loss_mag_base = (mag_l1 * valid_mask_expanded).sum() / valid_denom

    weighted_mag = mag_l1 * freq_span_map * valid_mask_expanded
    loss_mag_penalty = weighted_mag.sum() / valid_denom
    loss_mag = loss_mag_base + weight_mag_hf * loss_mag_penalty

    phase_l1 = (torch.abs(pred_cos - target_cos) + torch.abs(pred_sin - target_sin)) * valid_mask_expanded
    loss_phase = phase_l1.sum() / valid_denom

    return loss_mag, loss_phase
