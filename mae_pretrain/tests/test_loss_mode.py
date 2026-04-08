"""Loss-mode tests for MAE pretraining reconstruction losses."""

from __future__ import annotations

import torch
import pytest

from ..src.losses.recon_mag_phase import compute_mag_phase_losses


def _build_loss_tensors():
    """Create a tiny deterministic loss fixture."""
    pred = torch.tensor([[[2.0, 3.0, 4.0]]], dtype=torch.float32)
    target = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32)
    mask = torch.zeros((1, 1), dtype=torch.float32)
    freq_span = torch.ones((1, 1, 1), dtype=torch.float32)
    return pred, target, mask, freq_span


def test_mask_loss_mode_keeps_masked_only_behavior() -> None:
    """Masked loss should remain zero when no patch is masked."""
    pred, target, mask, freq_span = _build_loss_tensors()

    loss_mag, loss_phase = compute_mag_phase_losses(
        pred=pred,
        target_patches=target,
        mask=mask,
        patch_size=1,
        freq_span_patches=freq_span,
        weight_mag_hf=1.0,
        loss_mode="mask",
    )

    assert torch.isclose(loss_mag, torch.tensor(0.0))
    assert torch.isclose(loss_phase, torch.tensor(0.0))


def test_full_loss_mode_uses_all_patches_even_when_mask_is_zero() -> None:
    """Full-image loss should stay active even when MAE mask is all zeros."""
    pred, target, mask, freq_span = _build_loss_tensors()

    loss_mag, loss_phase = compute_mag_phase_losses(
        pred=pred,
        target_patches=target,
        mask=mask,
        patch_size=1,
        freq_span_patches=freq_span,
        weight_mag_hf=1.0,
        loss_mode="full",
    )

    assert torch.isclose(loss_mag, torch.tensor(2.0))
    assert torch.isclose(loss_phase, torch.tensor(5.0))


def test_invalid_loss_mode_raises_value_error() -> None:
    """Unknown loss modes should fail fast."""
    pred, target, mask, freq_span = _build_loss_tensors()

    with pytest.raises(ValueError, match="loss_mode"):
        compute_mag_phase_losses(
            pred=pred,
            target_patches=target,
            mask=mask,
            patch_size=1,
            freq_span_patches=freq_span,
            weight_mag_hf=1.0,
            loss_mode="unknown",
        )
