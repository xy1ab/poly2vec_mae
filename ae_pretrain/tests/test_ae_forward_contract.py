"""Contract tests for the public AE forward interface."""

from __future__ import annotations

import torch

from ..src.models.mae import PolyAutoencoder


def _build_small_model() -> PolyAutoencoder:
    """Create a tiny AE model suitable for interface tests."""
    return PolyAutoencoder(
        img_size=(8, 8),
        patch_size=2,
        in_chans=3,
        stem_channels=(8,),
        stem_strides=(2,),
        embed_dim=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        decoder_stage_channels=(16,),
        decoder_attention_type="none",
        decoder_attention_heads=(1,),
        decoder_attention_depths=(0,),
        decoder_conv_depths=(1,),
        decoder_upsample_mode="nearest",
    )


def test_forward_returns_patch_predictions_only() -> None:
    """Public forward should expose reconstructed patch tokens only."""
    model = _build_small_model()
    imgs = torch.randn(2, 3, 8, 8)

    pred = model(imgs)

    assert torch.is_tensor(pred)
    assert pred.shape == (2, 16, 12)


def test_forward_image_returns_full_resolution_reconstruction() -> None:
    """Image-forward path should return one full reconstructed image per input."""
    model = _build_small_model()
    imgs = torch.randn(2, 3, 8, 8)

    recon = model.forward_image(imgs)

    assert recon.shape == imgs.shape


def test_encoder_outputs_dense_latent_grid() -> None:
    """Encoder should emit a dense latent grid instead of a cls-token sequence."""
    model = _build_small_model()
    imgs = torch.randn(2, 3, 8, 8)

    latent = model.encode(imgs)

    assert latent.shape == (2, 16, 4, 4)
    assert model.encoder.latent_stride == 2


def test_forward_is_deterministic_without_training_noise() -> None:
    """Repeated forward passes should agree when dropout is disabled."""
    model = _build_small_model()
    imgs = torch.randn(2, 3, 8, 8)

    pred_a = model(imgs)
    pred_b = model(imgs)

    assert torch.allclose(pred_a, pred_b)
