"""Contract tests for the public MAE forward interface."""

from __future__ import annotations

import pytest
import torch

from ..src.models.mae import MaskedAutoencoderViTPoly


def _build_small_model() -> MaskedAutoencoderViTPoly:
    """Create a tiny MAE model suitable for interface tests."""
    return MaskedAutoencoderViTPoly(
        img_size=(4, 4),
        patch_size=2,
        in_chans=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
    )


def test_forward_returns_pred_and_mask_only() -> None:
    """Public forward should expose a strict `(pred, mask)` tuple."""
    model = _build_small_model()
    imgs = torch.randn(2, 3, 4, 4)

    output = model(imgs, mask_ratio=0.5)

    assert isinstance(output, tuple)
    assert len(output) == 2

    pred, mask = output
    assert pred.shape == (2, 4, 12)
    assert mask.shape == (2, 4)


@pytest.mark.parametrize("mask_ratio", [-0.1, 1.0, 1.2])
def test_forward_rejects_invalid_mask_ratio(mask_ratio: float) -> None:
    """Forward should fail fast on invalid mask ratios."""
    model = _build_small_model()
    imgs = torch.randn(1, 3, 4, 4)

    with pytest.raises(ValueError, match="mask_ratio"):
        model(imgs, mask_ratio=mask_ratio)


def test_forward_encoder_rejects_invalid_mask_ratio() -> None:
    """Encoder path should reject invalid mask ratios independently."""
    model = _build_small_model()
    imgs = torch.randn(1, 3, 4, 4)

    with pytest.raises(ValueError, match="mask_ratio"):
        model.forward_encoder(imgs, mask_ratio=1.0)


def test_forward_supports_training_and_inference_style_unpacking() -> None:
    """Repeated unpacking should expose the same `(pred, mask)` contract."""
    model = _build_small_model()
    imgs = torch.randn(2, 3, 4, 4)

    torch.manual_seed(0)
    pred_train, mask_train = model(imgs, mask_ratio=0.5)
    torch.manual_seed(0)
    pred_infer, mask_infer = model(imgs, mask_ratio=0.5)

    assert torch.allclose(pred_train, pred_infer)
    assert torch.equal(mask_train, mask_infer)
