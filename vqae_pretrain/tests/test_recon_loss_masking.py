"""Tests for valid-region masking in VQAE reconstruction losses."""

from __future__ import annotations

import unittest

import torch

from ..src.losses.recon_mag_phase import compute_mag_phase_losses


class ReconLossMaskingTest(unittest.TestCase):
    """Verify padding region does not contribute to reconstruction loss."""

    def test_padding_region_is_ignored_by_mag_and_phase_losses(self) -> None:
        """Large errors in padded area should not affect masked losses."""
        target = torch.zeros(1, 3, 2, 3, dtype=torch.float32)
        pred = target.clone()
        freq_span_map = torch.ones(1, 1, 2, 3, dtype=torch.float32)
        valid_mask = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]], dtype=torch.float32)

        pred[:, :, :, 2] = 100.0

        loss_mag, loss_phase = compute_mag_phase_losses(
            pred_imgs=pred,
            target_imgs=target,
            freq_span_map=freq_span_map,
            valid_mask=valid_mask,
            weight_mag_hf=1.0,
        )

        self.assertEqual(loss_mag.item(), 0.0)
        self.assertEqual(loss_phase.item(), 0.0)

    def test_masked_mean_divides_by_batch_times_valid_area(self) -> None:
        """Masked losses should average over every sample's valid region."""
        target = torch.zeros(2, 3, 1, 2, dtype=torch.float32)
        pred = target.clone()
        freq_span_map = torch.ones(1, 1, 1, 2, dtype=torch.float32)
        valid_mask = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)

        pred[0, 0, 0, 0] = 2.0
        pred[1, 0, 0, 0] = 4.0

        loss_mag, loss_phase = compute_mag_phase_losses(
            pred_imgs=pred,
            target_imgs=target,
            freq_span_map=freq_span_map,
            valid_mask=valid_mask,
            weight_mag_hf=0.0,
        )

        self.assertAlmostEqual(loss_mag.item(), 3.0, places=6)
        self.assertEqual(loss_phase.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
