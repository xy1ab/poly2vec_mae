"""Contract tests for the EMA vector quantizer."""

from __future__ import annotations

import torch
import unittest

from ..src.models.quantizer import EMAVectorQuantizer


class QuantizerContractTest(unittest.TestCase):
    """Shape and initialization contracts for the EMA quantizer."""

    def test_quantizer_forward_returns_expected_shapes(self) -> None:
        """Quantizer should preserve latent-grid shape and emit 2D index grids."""
        quantizer = EMAVectorQuantizer(
            num_embeddings=16,
            embedding_dim=8,
            decay=0.99,
            eps=1.0e-5,
            dead_code_threshold=1.0,
            query_chunk_size=8,
        )
        latents = torch.randn(2, 8, 4, 3)

        outputs = quantizer(latents, restart_pool_size=6)

        self.assertEqual(outputs.quantized.shape, latents.shape)
        self.assertEqual(outputs.indices.shape, (2, 4, 3))
        self.assertEqual(outputs.vq_loss.ndim, 0)
        self.assertEqual(outputs.usage_counts.shape, (16,))
        self.assertEqual(outputs.embed_sum.shape, (16, 8))
        self.assertEqual(outputs.restart_candidates.shape[1], 8)
        self.assertLessEqual(outputs.restart_candidates.shape[0], 6)

    def test_lookup_indices_restores_quantized_grid_shape(self) -> None:
        """Lookup path should map `[B,H,W]` indices back to `[B,C,H,W]` tensors."""
        quantizer = EMAVectorQuantizer(
            num_embeddings=8,
            embedding_dim=4,
            decay=0.99,
            eps=1.0e-5,
            dead_code_threshold=1.0,
            query_chunk_size=8,
        )
        latents = torch.randn(1, 4, 2, 2)
        outputs = quantizer(latents)

        restored = quantizer.lookup_indices(outputs.indices)

        self.assertEqual(restored.shape, latents.shape)

    def test_codebook_initialization_from_vectors_marks_quantizer_ready(self) -> None:
        """Explicit codebook initialization should flip the initialized flag."""
        quantizer = EMAVectorQuantizer(
            num_embeddings=8,
            embedding_dim=4,
            decay=0.99,
            eps=1.0e-5,
            dead_code_threshold=1.0,
            query_chunk_size=8,
        )
        vectors = torch.randn(32, 4)

        quantizer.initialize_codebook(vectors, num_iters=2)

        self.assertTrue(quantizer.is_initialized)
        self.assertEqual(quantizer.codebook.shape, (8, 4))

    def test_restart_candidate_pool_is_bounded_by_requested_size(self) -> None:
        """Restart candidate sampling should cap the per-rank pool size."""
        quantizer = EMAVectorQuantizer(
            num_embeddings=8,
            embedding_dim=4,
            decay=0.99,
            eps=1.0e-5,
            dead_code_threshold=1.0,
            query_chunk_size=8,
        )
        quantizer.train()
        latents = torch.randn(3, 4, 2, 2)

        outputs = quantizer(latents, restart_pool_size=5)

        self.assertIsNotNone(outputs.restart_candidates)
        self.assertLessEqual(outputs.restart_candidates.shape[0], 5)
        self.assertEqual(outputs.restart_candidates.shape[1], 4)

    def test_restart_dead_codes_can_reuse_a_small_replacement_pool(self) -> None:
        """Dead-code restart should refill every dead code even with few replacements."""
        quantizer = EMAVectorQuantizer(
            num_embeddings=8,
            embedding_dim=4,
            decay=0.99,
            eps=1.0e-5,
            dead_code_threshold=1.0,
            query_chunk_size=8,
        )
        quantizer.initialize_codebook(torch.randn(16, 4), num_iters=2)
        quantizer.cluster_size.zero_()
        replacement_pool = torch.randn(2, 4)

        restarted = quantizer.restart_dead_codes(replacement_pool)

        self.assertTrue(restarted)
        self.assertTrue(torch.allclose(quantizer.cluster_size, torch.ones_like(quantizer.cluster_size)))


if __name__ == "__main__":
    unittest.main()
