"""Runtime validation tests for VQ warmup scheduling and trainer arguments."""

from __future__ import annotations

from types import SimpleNamespace

import unittest

from ..src.engine.trainer import _effective_vq_beta, _should_use_vq, _validate_training_args


def _build_training_args(**overrides):
    """Create a minimal valid VQAE training-args namespace."""
    payload = {
        "lr": 1e-3,
        "min_lr": 0.0,
        "batch_size": 8,
        "val_ratio": 0.1,
        "augment_times": 1,
        "warmup_epochs": 5,
        "vq_warmup_epochs": 10,
        "vq_beta_warmup_epochs": 5,
        "log_interval": 10,
        "freq_type": "geometric",
        "w_min": 0.1,
        "w_max": 200.0,
        "depth": 2,
        "num_heads": 4,
        "stem_channels": "32,64,128",
        "stem_strides": "2,2,2",
        "decoder_stage_channels": "128,96,64",
        "decoder_attention_heads": "4,4,4",
        "decoder_attention_depths": "1,1,0",
        "decoder_conv_depths": "2,2,2",
        "codebook_size": 128,
        "code_dim": 64,
        "vq_beta": 0.25,
        "vq_decay": 0.99,
        "vq_eps": 1.0e-5,
        "vq_dead_code_threshold": 1.0,
        "vq_init_max_vectors": 1024,
        "vq_kmeans_iters": 3,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


class RuntimeValidationTest(unittest.TestCase):
    """Warmup schedule and argument validation tests."""

    def test_vq_is_disabled_during_warmup(self) -> None:
        """VQ should stay off before the configured VQ warmup boundary."""
        self.assertFalse(_should_use_vq(epoch_index=0, vq_warmup_epochs=10))
        self.assertFalse(_should_use_vq(epoch_index=9, vq_warmup_epochs=10))
        self.assertTrue(_should_use_vq(epoch_index=10, vq_warmup_epochs=10))

    def test_vq_beta_ramps_linearly_after_vq_warmup(self) -> None:
        """Effective beta should stay zero, then increase linearly after VQ warmup."""
        args = _build_training_args(vq_beta=0.25, vq_warmup_epochs=10, vq_beta_warmup_epochs=5)

        self.assertEqual(_effective_vq_beta(epoch_index=0, args=args), 0.0)
        self.assertEqual(_effective_vq_beta(epoch_index=10, args=args), 0.0)
        self.assertAlmostEqual(_effective_vq_beta(epoch_index=12, args=args), 0.1, places=6)
        self.assertAlmostEqual(_effective_vq_beta(epoch_index=15, args=args), 0.25, places=6)
        self.assertAlmostEqual(_effective_vq_beta(epoch_index=30, args=args), 0.25, places=6)

    def test_training_args_reject_invalid_vq_decay(self) -> None:
        """Trainer validation should reject VQ decay values outside `(0,1)`."""
        for vq_decay in (0.0, 1.0, 1.1):
            with self.assertRaisesRegex(ValueError, "vq_decay"):
                _validate_training_args(_build_training_args(vq_decay=vq_decay))


if __name__ == "__main__":
    unittest.main()
