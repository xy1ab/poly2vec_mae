"""Runtime validation tests for AE training, pipeline, and legacy evaluator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ..src.datasets.shard_io import save_triangle_shard
from ..src.datasets.sharded_pt_dataset import EagerShardedPolyDataset
from ..src.engine.evaluator import eval_main
from ..src.engine.pipeline import PolyAeReconstructionPipeline
from ..src.engine.trainer import _validate_training_args


def _build_training_args(**overrides):
    """Create a minimal valid AE training-args namespace for validator tests."""
    payload = {
        "lr": 1e-3,
        "min_lr": 0.0,
        "batch_size": 8,
        "patch_size": 4,
        "val_ratio": 0.1,
        "augment_times": 1,
        "warmup_epochs": 0,
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
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


@pytest.mark.parametrize("val_ratio", [0.0, -0.1, 1.0, 1.1])
def test_training_args_reject_invalid_val_ratio(val_ratio: float) -> None:
    """Training validation should fail fast on invalid validation split ratios."""
    with pytest.raises(ValueError, match="val_ratio"):
        _validate_training_args(_build_training_args(val_ratio=val_ratio))


@pytest.mark.parametrize("augment_times", [0, -1, 1.5])
def test_training_args_reject_invalid_augment_times(augment_times: float) -> None:
    """Training validation should reject non-positive augmentation repeats."""
    with pytest.raises(ValueError, match="augment_times"):
        _validate_training_args(_build_training_args(augment_times=augment_times))


@pytest.mark.parametrize("augment_times", [0, -1, 1.5])
def test_dataset_rejects_invalid_augment_times(augment_times: float) -> None:
    """Indexed datasets should no longer clamp invalid augmentation repeats."""
    with pytest.raises(ValueError, match="augment_times"):
        EagerShardedPolyDataset(
            all_samples=[np.zeros((1, 3, 2), dtype=np.float32)],
            sample_indices=[0],
            augment_times=augment_times,
        )


def test_reconstruction_pipeline_does_not_accept_mask_ratio() -> None:
    """AE reconstruction pipeline should not expose any legacy masking argument."""
    pipeline = object.__new__(PolyAeReconstructionPipeline)

    with pytest.raises(TypeError, match="mask_ratio"):
        pipeline.reconstruct_real_imag([], mask_ratio=1.0)


def test_legacy_evaluator_rejects_negative_index(tmp_path) -> None:
    """Legacy evaluator should reject negative sample indices explicitly."""
    data_path = save_triangle_shard(
        tmp_path / "samples.pt",
        [np.zeros((1, 3, 2), dtype=np.float32)],
    )
    args = SimpleNamespace(
        index=-1,
        data_path=str(data_path),
        save_dir=str(tmp_path),
        spatial_size=64,
        pos_freqs=31,
        w_min=0.1,
        w_max=200.0,
        freq_type="geometric",
        patch_size=4,
    )

    with pytest.raises(IndexError, match="out of range"):
        eval_main(args)
