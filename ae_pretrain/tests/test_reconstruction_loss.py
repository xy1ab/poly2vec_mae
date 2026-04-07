"""Reconstruction-loss tests for AE frequency-domain pretraining."""

from __future__ import annotations

import torch

from ..src.losses.recon_mag_phase import compute_mag_phase_losses


def _build_loss_tensors():
    """Create a tiny deterministic loss fixture."""
    pred = torch.tensor([[[2.0, 3.0, 4.0]]], dtype=torch.float32)
    target = torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32)
    freq_span = torch.ones((1, 1, 1), dtype=torch.float32)
    return pred, target, freq_span


def test_full_patch_loss_matches_expected_values() -> None:
    """Magnitude and phase losses should match the hand-computed fixture."""
    pred, target, freq_span = _build_loss_tensors()

    loss_mag, loss_phase = compute_mag_phase_losses(
        pred=pred,
        target_patches=target,
        patch_size=1,
        freq_span_patches=freq_span,
        weight_mag_hf=1.0,
    )

    assert torch.isclose(loss_mag, torch.tensor(2.0))
    assert torch.isclose(loss_phase, torch.tensor(5.0))


def test_high_frequency_penalty_can_be_disabled() -> None:
    """Setting the high-frequency weight to zero should keep only the base L1 term."""
    pred, target, freq_span = _build_loss_tensors()

    loss_mag, loss_phase = compute_mag_phase_losses(
        pred=pred,
        target_patches=target,
        patch_size=1,
        freq_span_patches=freq_span,
        weight_mag_hf=0.0,
    )

    assert torch.isclose(loss_mag, torch.tensor(1.0))
    assert torch.isclose(loss_phase, torch.tensor(5.0))
