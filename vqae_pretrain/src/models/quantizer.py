"""EMA vector-quantization modules for polygon VQAE pretraining.

This file intentionally keeps the implementation self-contained and explicit.
The quantizer is used after the encoder latent grid projection:

1. Flatten one `[B, C, H, W]` latent grid into `[N, C]`.
2. Find the nearest codebook vector for each latent vector with fp32 distance.
3. Use a straight-through estimator for the quantized output.
4. Update the codebook with EMA statistics.

The codebook initialization follows an engineering-oriented mini-batch K-means:
1. Randomly sample vectors from the continuous warmup latent pool.
2. Run a small number of chunked assignment/update iterations on GPU.
3. Use the resulting cluster centers as the initial codebook.

This is intentionally lighter than a full exact K-means over the entire
training set, while still providing a much better initialization than pure
random codebook vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    """Structured quantizer forward result."""

    quantized: torch.Tensor
    indices: torch.Tensor
    vq_loss: torch.Tensor
    perplexity: torch.Tensor
    active_codes: torch.Tensor


class EMAVectorQuantizer(nn.Module):
    """EMA-updated vector quantizer for latent grids.

    Args:
        num_embeddings: Codebook size.
        embedding_dim: Dimension of each code vector.
        decay: EMA decay factor.
        eps: Numerical-stability term used in EMA normalization.
        dead_code_threshold: EMA-count threshold below which a code is
            considered dead and will be restarted from current batch features.
        query_chunk_size: Query chunk size for chunked nearest-neighbor search.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        eps: float = 1.0e-5,
        dead_code_threshold: float = 1.0,
        query_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if int(num_embeddings) <= 1:
            raise ValueError(f"`num_embeddings` must be > 1, got {num_embeddings}")
        if int(embedding_dim) <= 0:
            raise ValueError(f"`embedding_dim` must be > 0, got {embedding_dim}")

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.decay = float(decay)
        self.eps = float(eps)
        self.dead_code_threshold = float(dead_code_threshold)
        self.query_chunk_size = max(1, int(query_chunk_size))

        self.register_buffer("codebook", torch.empty(self.num_embeddings, self.embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("embed_avg", torch.zeros(self.num_embeddings, self.embedding_dim))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the codebook to a small random state."""
        bound = 1.0 / math.sqrt(self.embedding_dim)
        with torch.no_grad():
            self.codebook.uniform_(-bound, bound)
            self.cluster_size.zero_()
            self.embed_avg.zero_()
            self.initialized.fill_(False)

    @property
    def is_initialized(self) -> bool:
        """Whether the codebook has been initialized from latent data."""
        return bool(self.initialized.item())

    @staticmethod
    def _dist_enabled() -> bool:
        """Whether distributed collectives are currently available."""
        return dist.is_available() and dist.is_initialized()

    @classmethod
    def _is_main_rank(cls) -> bool:
        """Whether current process is rank 0 under distributed execution."""
        return not cls._dist_enabled() or dist.get_rank() == 0

    @classmethod
    def _all_reduce_in_place(cls, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce one tensor in place when distributed execution is enabled."""
        if cls._dist_enabled():
            dist.all_reduce(tensor)
        return tensor

    @classmethod
    def _broadcast_in_place(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast one tensor from rank 0 when distributed execution is enabled."""
        if cls._dist_enabled():
            dist.broadcast(tensor, src=0)
        return tensor

    def _broadcast_state_from_rank0(self) -> None:
        """Synchronize EMA buffers after one rank-only dead-code restart."""
        self._broadcast_in_place(self.codebook)
        self._broadcast_in_place(self.cluster_size)
        self._broadcast_in_place(self.embed_avg)
        self._broadcast_in_place(self.initialized)

    def _flatten(self, z: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Flatten `[B,C,H,W]` latent grids into `[N,C]` vectors."""
        if z.ndim != 4:
            raise ValueError(f"Expected latent shape [B,C,H,W], got {tuple(z.shape)}")
        batch, channels, height, width = z.shape
        if channels != self.embedding_dim:
            raise ValueError(
                f"Quantizer input channel mismatch: expected {self.embedding_dim}, got {channels}"
            )
        vectors = z.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
        return vectors, (batch, channels, height, width)

    def _compute_distance_chunk(self, vectors: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Compute one fp32 squared-L2 distance matrix chunk."""
        vectors_fp32 = vectors.float()
        codebook_fp32 = codebook.float()
        return (
            vectors_fp32.pow(2).sum(dim=1, keepdim=True)
            + codebook_fp32.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * vectors_fp32 @ codebook_fp32.t()
        )

    def _nearest_indices(self, vectors: torch.Tensor) -> torch.Tensor:
        """Find nearest code indices with chunked fp32 distance computation."""
        index_chunks: list[torch.Tensor] = []
        for start in range(0, vectors.shape[0], self.query_chunk_size):
            chunk = vectors[start : start + self.query_chunk_size]
            distances = self._compute_distance_chunk(chunk, self.codebook)
            index_chunks.append(torch.argmin(distances, dim=1))
        return torch.cat(index_chunks, dim=0)

    @torch.no_grad()
    def _restart_dead_codes(self, vectors: torch.Tensor) -> bool:
        """Restart dead codes from current batch latent vectors.

        Returns:
            Whether any dead-code restart was actually performed.
        """
        if self.dead_code_threshold <= 0:
            return False

        dead_mask = self.cluster_size < self.dead_code_threshold
        dead_count = int(dead_mask.sum().item())
        if dead_count == 0 or vectors.numel() == 0:
            return False

        sample_count = min(dead_count, vectors.shape[0])
        perm = torch.randperm(vectors.shape[0], device=vectors.device)[:sample_count]
        replacements = vectors[perm].float()
        if sample_count < dead_count:
            extra_perm = torch.randint(0, sample_count, (dead_count - sample_count,), device=vectors.device)
            replacements = torch.cat([replacements, replacements[extra_perm]], dim=0)

        dead_indices = torch.nonzero(dead_mask, as_tuple=False).flatten()
        self.codebook[dead_indices] = replacements
        self.embed_avg[dead_indices] = replacements
        self.cluster_size[dead_indices] = 1.0
        return True

    @torch.no_grad()
    def initialize_codebook(self, vectors: torch.Tensor, num_iters: int = 10) -> None:
        """Initialize the codebook with a chunked engineering K-means.

        The algorithm uses:
        1. Random latent-vector sampling for initial cluster centers.
        2. Chunked assignment on GPU.
        3. A small number of centroid refinement iterations.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                "K-means initialization vectors must have shape [N, embedding_dim], "
                f"got {tuple(vectors.shape)}"
            )
        if vectors.shape[0] == 0:
            raise ValueError("Cannot initialize codebook from an empty latent-vector set.")

        vectors = vectors.detach()
        if vectors.shape[0] < self.num_embeddings:
            extra_indices = torch.randint(
                0,
                vectors.shape[0],
                size=(self.num_embeddings - vectors.shape[0],),
                device=vectors.device,
            )
            vectors = torch.cat([vectors, vectors[extra_indices]], dim=0)

        init_perm = torch.randperm(vectors.shape[0], device=vectors.device)[: self.num_embeddings]
        centroids = vectors[init_perm].float().clone()

        for _ in range(max(1, int(num_iters))):
            counts = torch.zeros(self.num_embeddings, device=vectors.device, dtype=torch.float32)
            sums = torch.zeros(
                self.num_embeddings,
                self.embedding_dim,
                device=vectors.device,
                dtype=torch.float32,
            )

            for start in range(0, vectors.shape[0], self.query_chunk_size):
                chunk = vectors[start : start + self.query_chunk_size]
                distances = self._compute_distance_chunk(chunk, centroids)
                indices = torch.argmin(distances, dim=1)
                counts.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
                sums.index_add_(0, indices, chunk.float())

            dead_mask = counts <= 0
            if bool(dead_mask.any().item()):
                dead_count = int(dead_mask.sum().item())
                refill_perm = torch.randperm(vectors.shape[0], device=vectors.device)[:dead_count]
                sums[dead_mask] = vectors[refill_perm].float()
                counts[dead_mask] = 1.0

            centroids = sums / counts.unsqueeze(1).clamp_min(1.0)

        self.codebook.copy_(centroids)
        self.embed_avg.copy_(centroids)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def lookup_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map code indices `[B,H,W]` back to quantized code vectors `[B,C,H,W]`."""
        if indices.ndim != 3:
            raise ValueError(f"Expected code index shape [B,H,W], got {tuple(indices.shape)}")
        batch, height, width = indices.shape
        flat_indices = indices.reshape(-1).long()
        quantized = self.codebook.index_select(0, flat_indices)
        return quantized.reshape(batch, height, width, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        """Quantize one latent grid and update the EMA codebook during training."""
        vectors, (batch, channels, height, width) = self._flatten(z)
        if not self.is_initialized:
            self.initialize_codebook(vectors, num_iters=1)

        indices = self._nearest_indices(vectors)
        quantized = self.codebook.index_select(0, indices).reshape(batch, height, width, channels)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        vq_loss = F.mse_loss(z.float(), quantized.detach().float())
        quantized_st = z + (quantized - z).detach()

        with torch.no_grad():
            usage_counts = torch.bincount(indices, minlength=self.num_embeddings).float()
            self._all_reduce_in_place(usage_counts)
            usage_probs = usage_counts / usage_counts.sum().clamp_min(1.0)
            perplexity = torch.exp(-(usage_probs * torch.log(usage_probs.clamp_min(1.0e-10))).sum())
            active_codes = (usage_counts > 0).sum().float()

            if self.training:
                embed_sum = torch.zeros_like(self.embed_avg)
                embed_sum.index_add_(0, indices, vectors.float())
                self._all_reduce_in_place(embed_sum)

                self.cluster_size.mul_(self.decay).add_(usage_counts, alpha=1.0 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

                total_count = self.cluster_size.sum()
                normalized_cluster_size = (
                    (self.cluster_size + self.eps)
                    / (total_count + self.num_embeddings * self.eps)
                    * total_count.clamp_min(1.0)
                )
                self.codebook.copy_(self.embed_avg / normalized_cluster_size.unsqueeze(1))
                restarted_dead_codes = False
                if self._is_main_rank():
                    restarted_dead_codes = self._restart_dead_codes(vectors)
                if restarted_dead_codes:
                    self._broadcast_state_from_rank0()

        return QuantizerOutput(
            quantized=quantized_st,
            indices=indices.reshape(batch, height, width),
            vq_loss=vq_loss,
            perplexity=perplexity,
            active_codes=active_codes,
        )
