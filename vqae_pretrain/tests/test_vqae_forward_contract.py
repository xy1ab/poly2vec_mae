"""Contract tests for the public VQAE forward interface."""

from __future__ import annotations

import torch
import unittest

from ..src.models.vqae import PolyVqAutoencoder


def _build_small_model() -> PolyVqAutoencoder:
    """Create a tiny VQAE model suitable for interface tests."""
    return PolyVqAutoencoder(
        img_size=(8, 8),
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
        codebook_size=32,
        code_dim=8,
    )


class VqAeForwardContractTest(unittest.TestCase):
    """Public forward and decode contracts for the VQAE model."""

    def test_forward_without_vq_emits_full_resolution_reconstruction(self) -> None:
        """Warmup path should emit full-image reconstructions and no discrete indices."""
        model = _build_small_model()
        imgs = torch.randn(2, 3, 8, 8)

        outputs = model(imgs, use_vq=False)

        self.assertEqual(outputs.recon_imgs.shape, imgs.shape)
        self.assertTrue(torch.all(outputs.recon_imgs[:, 0] >= 0.0))
        phase_norm = outputs.recon_imgs[:, 1].square() + outputs.recon_imgs[:, 2].square()
        self.assertTrue(torch.allclose(phase_norm, torch.ones_like(phase_norm), atol=1e-5, rtol=1e-5))
        self.assertIsNone(outputs.indices)
        self.assertFalse(outputs.using_vq)

    def test_forward_with_vq_emits_index_grid(self) -> None:
        """VQ path should return one integer index grid per sample."""
        model = _build_small_model()
        imgs = torch.randn(2, 3, 8, 8)

        outputs = model(imgs, use_vq=True)

        self.assertIsNotNone(outputs.indices)
        self.assertEqual(outputs.indices.shape, (2, 4, 4))
        self.assertEqual(outputs.recon_imgs.shape, imgs.shape)
        self.assertTrue(torch.all(outputs.recon_imgs[:, 0] >= 0.0))
        phase_norm = outputs.recon_imgs[:, 1].square() + outputs.recon_imgs[:, 2].square()
        self.assertTrue(torch.allclose(phase_norm, torch.ones_like(phase_norm), atol=1e-5, rtol=1e-5))
        self.assertTrue(outputs.using_vq)

    def test_decode_indices_restores_full_resolution_images(self) -> None:
        """Decoding indices should return one reconstructed image per sample."""
        model = _build_small_model()
        imgs = torch.randn(2, 3, 8, 8)
        outputs = model(imgs, use_vq=True)

        recon = model.decode_indices(outputs.indices)

        self.assertEqual(recon.shape, imgs.shape)


if __name__ == "__main__":
    unittest.main()
