"""Contract tests for the public RVQAE forward interface."""

from __future__ import annotations

import torch
import unittest

from ..src.models.rvqae import PolyRvqAutoencoder


def _build_small_model() -> PolyRvqAutoencoder:
    """Create a tiny RVQAE model suitable for interface tests."""
    return PolyRvqAutoencoder(
        img_size=(8, 8),
        in_chans=3,
        enc_conv_channels=(8,),
        enc_conv_strides=(2,),
        patch_size=2,
        embed_dim=16,
        enc_vit_depth=1,
        enc_vit_head=4,
        enc_vit_mlp_ratio=2.0,
        dec_vit_depth=1,
        dec_vit_head=4,
        dec_vit_mlp_ratio=2.0,
        full_res_head_channels=(),
        codebook_size=32,
        code_dim=8,
        rvq_num_quantizers=1,
        rvq_loss_weights=(1.0,),
    )


class RvqAeForwardContractTest(unittest.TestCase):
    """Public forward and decode contracts for the RVQAE model."""

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
        self.assertEqual(outputs.indices.shape, (2, 1, 2, 2))
        self.assertEqual(outputs.perplexity.shape, (1,))
        self.assertEqual(outputs.active_codes.shape, (1,))
        self.assertEqual(outputs.recon_imgs.shape, imgs.shape)
        self.assertTrue(torch.all(outputs.recon_imgs[:, 0] >= 0.0))
        phase_norm = outputs.recon_imgs[:, 1].square() + outputs.recon_imgs[:, 2].square()
        self.assertTrue(torch.allclose(phase_norm, torch.ones_like(phase_norm), atol=1e-5, rtol=1e-5))
        self.assertTrue(outputs.using_vq)

    def test_tokenize_matches_forward_indices_without_running_decoder_logic(self) -> None:
        """Tokenizer API should emit the same discrete grid as the VQ forward path."""
        model = _build_small_model()
        imgs = torch.randn(2, 3, 8, 8)

        token_grid = model.tokenize(imgs)
        outputs = model(imgs, use_vq=True)

        self.assertEqual(token_grid.shape, (2, 2, 2))
        self.assertTrue(torch.equal(token_grid, outputs.indices[:, 0]))

    def test_decode_indices_restores_full_resolution_images(self) -> None:
        """Decoding indices should return one reconstructed image per sample."""
        model = _build_small_model()
        imgs = torch.randn(2, 3, 8, 8)
        outputs = model(imgs, use_vq=True)

        recon = model.decode_indices(outputs.indices)

        self.assertEqual(recon.shape, imgs.shape)

    def test_forward_supports_full_res_head(self) -> None:
        """Full-resolution residual head should preserve output image shapes."""
        model = PolyRvqAutoencoder(
            img_size=(8, 8),
            in_chans=3,
            enc_conv_channels=(8,),
            enc_conv_strides=(2,),
            patch_size=2,
            embed_dim=16,
            enc_vit_depth=1,
            enc_vit_head=4,
            enc_vit_mlp_ratio=2.0,
            dec_vit_depth=1,
            dec_vit_head=4,
            dec_vit_mlp_ratio=2.0,
            full_res_head_channels=(12, 12),
            codebook_size=32,
            code_dim=8,
            rvq_num_quantizers=1,
            rvq_loss_weights=(1.0,),
        )
        imgs = torch.randn(2, 3, 8, 8)

        outputs = model(imgs, use_vq=True)

        self.assertEqual(outputs.recon_imgs.shape, imgs.shape)

    def test_forward_supports_two_level_rvq(self) -> None:
        """RVQ path should return one index grid per quantizer level."""
        model = PolyRvqAutoencoder(
            img_size=(8, 8),
            in_chans=3,
            enc_conv_channels=(8,),
            enc_conv_strides=(2,),
            patch_size=2,
            embed_dim=16,
            enc_vit_depth=1,
            enc_vit_head=4,
            enc_vit_mlp_ratio=2.0,
            dec_vit_depth=1,
            dec_vit_head=4,
            dec_vit_mlp_ratio=2.0,
            full_res_head_channels=(),
            codebook_size=32,
            code_dim=8,
            rvq_num_quantizers=2,
            rvq_loss_weights=(1.0, 0.5),
        )
        imgs = torch.randn(2, 3, 8, 8)

        outputs = model(imgs, use_vq=True)

        self.assertEqual(outputs.recon_imgs.shape, imgs.shape)
        self.assertEqual(outputs.indices.shape, (2, 2, 2, 2))
        self.assertEqual(outputs.perplexity.shape, (2,))
        self.assertEqual(model.tokenize(imgs).shape, (2, 2, 2, 2))

    def test_forward_supports_pure_vit_path(self) -> None:
        """Empty conv/head channel lists should produce a conv-free ViT body."""
        model = PolyRvqAutoencoder(
            img_size=(8, 8),
            in_chans=3,
            enc_conv_channels=(),
            enc_conv_strides=(),
            patch_size=4,
            embed_dim=16,
            enc_vit_depth=1,
            enc_vit_head=4,
            enc_vit_mlp_ratio=2.0,
            dec_vit_depth=1,
            dec_vit_head=4,
            dec_vit_mlp_ratio=2.0,
            full_res_head_channels=(),
            codebook_size=32,
            code_dim=8,
            rvq_num_quantizers=1,
            rvq_loss_weights=(1.0,),
        )
        imgs = torch.randn(2, 3, 8, 8)

        outputs = model(imgs, use_vq=True)

        self.assertEqual(outputs.recon_imgs.shape, imgs.shape)
        self.assertEqual(model.tokenize(imgs).shape, (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
