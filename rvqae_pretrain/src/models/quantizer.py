"""EMA vector quantizers for polygon RVQAE pretraining."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
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
    usage_counts: torch.Tensor | None = None
    embed_sum: torch.Tensor | None = None
    restart_candidates: torch.Tensor | None = None


class EMAVectorQuantizer(nn.Module):
    """EMA-updated vector quantizer for one latent grid."""

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

    @property
    def is_initialized(self) -> bool:
        """Whether this codebook has been initialized from latent data."""
        return bool(self.initialized.item())

    def reset_parameters(self) -> None:
        """Reset the codebook to a small random state."""
        bound = 1.0 / math.sqrt(self.embedding_dim)
        with torch.no_grad():
            self.codebook.uniform_(-bound, bound)
            self.cluster_size.zero_()
            self.embed_avg.zero_()
            self.initialized.fill_(False)

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

    @staticmethod
    def compute_usage_metrics(usage_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute perplexity and active-code count from one usage histogram."""
        usage_counts = usage_counts.float()
        usage_probs = usage_counts / usage_counts.sum().clamp_min(1.0)
        perplexity = torch.exp(-(usage_probs * torch.log(usage_probs.clamp_min(1.0e-10))).sum())
        active_codes = (usage_counts > 0).sum().float()
        return perplexity, active_codes

    @torch.no_grad()
    def _build_embed_sum(self, vectors: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Build one local embedding-sum tensor indexed by chosen code ids."""
        embed_sum = torch.zeros_like(self.embed_avg)
        embed_sum.index_add_(0, indices, vectors.float())
        return embed_sum

    @torch.no_grad()
    def sample_restart_candidates(self, vectors: torch.Tensor, max_vectors: int) -> torch.Tensor:
        """Sample one bounded local candidate pool for dead-code restart."""
        max_vectors = max(0, int(max_vectors))
        if max_vectors == 0 or vectors.numel() == 0:
            return vectors.new_zeros((0, self.embedding_dim), dtype=torch.float32)

        sample_count = min(max_vectors, int(vectors.shape[0]))
        perm = torch.randperm(vectors.shape[0], device=vectors.device)[:sample_count]
        return vectors[perm].float()

    @torch.no_grad()
    def restart_dead_codes(self, replacement_vectors: torch.Tensor) -> bool:
        """Restart dead codes from one already-aggregated replacement pool."""
        if self.dead_code_threshold <= 0:
            return False

        dead_mask = self.cluster_size < self.dead_code_threshold
        dead_count = int(dead_mask.sum().item())
        if dead_count == 0 or replacement_vectors.numel() == 0:
            return False

        sample_count = min(dead_count, int(replacement_vectors.shape[0]))
        if sample_count <= 0:
            return False

        perm = torch.randperm(replacement_vectors.shape[0], device=replacement_vectors.device)[:sample_count]
        replacements = replacement_vectors[perm].float()
        if sample_count < dead_count:
            extra_perm = torch.randint(0, sample_count, (dead_count - sample_count,), device=replacement_vectors.device)
            replacements = torch.cat([replacements, replacements[extra_perm]], dim=0)

        dead_indices = torch.nonzero(dead_mask, as_tuple=False).flatten()
        self.codebook[dead_indices] = replacements
        self.embed_avg[dead_indices] = replacements
        self.cluster_size[dead_indices] = 1.0
        return True

    @torch.no_grad()
    def apply_ema_update(self, usage_counts: torch.Tensor, embed_sum: torch.Tensor) -> None:
        """Apply one EMA codebook update from already-aggregated global stats."""
        if usage_counts.shape != self.cluster_size.shape:
            raise ValueError(
                "Global usage-count shape mismatch: "
                f"expected {tuple(self.cluster_size.shape)}, got {tuple(usage_counts.shape)}"
            )
        if embed_sum.shape != self.embed_avg.shape:
            raise ValueError(
                "Global embed-sum shape mismatch: "
                f"expected {tuple(self.embed_avg.shape)}, got {tuple(embed_sum.shape)}"
            )

        usage_counts = usage_counts.float()
        embed_sum = embed_sum.float()

        self.cluster_size.mul_(self.decay).add_(usage_counts, alpha=1.0 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

        total_count = self.cluster_size.sum()
        normalized_cluster_size = (
            (self.cluster_size + self.eps)
            / (total_count + self.num_embeddings * self.eps)
            * total_count.clamp_min(1.0)
        )
        self.codebook.copy_(self.embed_avg / normalized_cluster_size.unsqueeze(1))
        self.initialized.fill_(True)

    @torch.no_grad()
    def initialize_codebook(self, vectors: torch.Tensor, num_iters: int = 10) -> None:
        """Initialize the codebook with a chunked engineering K-means."""
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
    def encode_indices(self, z: torch.Tensor) -> torch.Tensor:
        """Encode one latent grid `[B,C,H,W]` into discrete code indices `[B,H,W]`."""
        vectors, (batch, _channels, height, width) = self._flatten(z)
        if not self.is_initialized:
            self.initialize_codebook(vectors, num_iters=1)
        indices = self._nearest_indices(vectors)
        return indices.reshape(batch, height, width)

    @torch.no_grad()
    def lookup_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map code indices `[B,H,W]` back to quantized code vectors `[B,C,H,W]`."""
        if indices.ndim != 3:
            raise ValueError(f"Expected code index shape [B,H,W], got {tuple(indices.shape)}")
        batch, height, width = indices.shape
        flat_indices = indices.reshape(-1).long()
        quantized = self.codebook.index_select(0, flat_indices)
        return quantized.reshape(batch, height, width, self.embedding_dim).permute(0, 3, 1, 2).contiguous()

    def forward(self, z: torch.Tensor, restart_pool_size: int = 0) -> QuantizerOutput:
        """Quantize one latent grid and emit local stats for trainer-side sync."""
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
            perplexity, active_codes = self.compute_usage_metrics(usage_counts)

            embed_sum = None
            restart_candidates = None
            if self.training:
                embed_sum = self._build_embed_sum(vectors, indices)
                if self.dead_code_threshold > 0 and int(restart_pool_size) > 0:
                    restart_candidates = self.sample_restart_candidates(vectors, max_vectors=restart_pool_size)

        return QuantizerOutput(
            quantized=quantized_st,
            indices=indices.reshape(batch, height, width),
            vq_loss=vq_loss,
            perplexity=perplexity,
            active_codes=active_codes,
            usage_counts=usage_counts,
            embed_sum=embed_sum,
            restart_candidates=restart_candidates,
        )


class ResidualEMAVectorQuantizer(nn.Module):
    """Residual stack of independent EMA vector quantizers."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_quantizers: int = 1,
        loss_weights: list[float] | tuple[float, ...] | None = None,
        decay: float = 0.99,
        eps: float = 1.0e-5,
        dead_code_threshold: float = 1.0,
        query_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.num_quantizers = int(num_quantizers)
        if self.num_quantizers <= 0:
            raise ValueError(f"`num_quantizers` must be > 0, got {num_quantizers}")

        weights = [1.0] * self.num_quantizers if loss_weights is None else [float(v) for v in loss_weights]
        if len(weights) != self.num_quantizers:
            raise ValueError(
                "`rvq_loss_weights` length must equal `rvq_num_quantizers`: "
                f"got {len(weights)} vs {self.num_quantizers}"
            )
        if any(value <= 0 for value in weights):
            raise ValueError("All `rvq_loss_weights` values must be > 0")

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.dead_code_threshold = float(dead_code_threshold)
        self.register_buffer("loss_weights", torch.tensor(weights, dtype=torch.float32))
        self.quantizers = nn.ModuleList(
            [
                EMAVectorQuantizer(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    decay=decay,
                    eps=eps,
                    dead_code_threshold=dead_code_threshold,
                    query_chunk_size=query_chunk_size,
                )
                for _ in range(self.num_quantizers)
            ]
        )

    @property
    def is_initialized(self) -> bool:
        """Whether all residual codebooks have been initialized."""
        return all(quantizer.is_initialized for quantizer in self.quantizers)

    def iter_quantizers(self):
        """Yield residual quantizers in order."""
        return iter(self.quantizers)

    def _flatten(self, z: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Flatten `[B,C,H,W]` latent grids into `[N,C]` vectors."""
        return self.quantizers[0]._flatten(z)

    def compute_usage_metrics(self, usage_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-level perplexity and active-code counts."""
        if usage_counts.ndim == 1:
            return EMAVectorQuantizer.compute_usage_metrics(usage_counts)
        perplexities = []
        active_codes = []
        for level_counts in usage_counts:
            level_perplexity, level_active = EMAVectorQuantizer.compute_usage_metrics(level_counts)
            perplexities.append(level_perplexity)
            active_codes.append(level_active)
        return torch.stack(perplexities), torch.stack(active_codes)

    @torch.no_grad()
    def initialize_codebook(self, vectors: torch.Tensor, num_iters: int = 10) -> None:
        """Sequentially initialize residual codebooks from residual vectors."""
        residual = vectors.detach().float()
        for quantizer in self.quantizers:
            quantizer.initialize_codebook(residual, num_iters=num_iters)
            indices = quantizer._nearest_indices(residual)
            quantized = quantizer.codebook.index_select(0, indices).float()
            residual = residual - quantized

    @torch.no_grad()
    def encode_indices(self, z: torch.Tensor) -> torch.Tensor:
        """Encode one latent grid into RVQ indices `[B,Q,H,W]`."""
        vectors, (batch, _channels, height, width) = self._flatten(z)
        if not self.is_initialized:
            self.initialize_codebook(vectors, num_iters=1)

        residual = vectors.float()
        indices_by_level = []
        for quantizer in self.quantizers:
            indices = quantizer._nearest_indices(residual)
            quantized = quantizer.codebook.index_select(0, indices).float()
            residual = residual - quantized
            indices_by_level.append(indices.reshape(batch, height, width))
        return torch.stack(indices_by_level, dim=1)

    @torch.no_grad()
    def lookup_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Map RVQ indices `[B,Q,H,W]` back to summed code features `[B,C,H,W]`."""
        if indices.ndim == 3:
            if self.num_quantizers != 1:
                raise ValueError("Single index grid is only valid when `rvq_num_quantizers=1`")
            indices = indices.unsqueeze(1)
        if indices.ndim != 4:
            raise ValueError(f"Expected RVQ index shape [B,Q,H,W], got {tuple(indices.shape)}")
        batch, levels, height, width = indices.shape
        if int(levels) != self.num_quantizers:
            raise ValueError(f"Expected {self.num_quantizers} RVQ levels, got {levels}")

        quantized_sum = None
        for level, quantizer in enumerate(self.quantizers):
            flat_indices = indices[:, level].reshape(-1).long()
            quantized = quantizer.codebook.index_select(0, flat_indices)
            quantized = quantized.reshape(batch, height, width, self.embedding_dim)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            quantized_sum = quantized if quantized_sum is None else quantized_sum + quantized
        return quantized_sum

    def forward(self, z: torch.Tensor, restart_pool_size: int = 0) -> QuantizerOutput:
        """Quantize one latent grid with residual codebooks."""
        vectors, (_batch, _channels, _height, _width) = self._flatten(z)
        if not self.is_initialized:
            self.initialize_codebook(vectors, num_iters=1)

        residual = z
        quantized_sum = torch.zeros_like(z)
        indices_by_level = []
        losses = []
        perplexities = []
        active_codes = []
        usage_counts = []
        embed_sums = []
        restart_candidates = []

        for quantizer in self.quantizers:
            outputs = quantizer(residual, restart_pool_size=restart_pool_size)
            quantized_raw = outputs.quantized.detach()
            quantized_sum = quantized_sum + quantized_raw
            residual = residual - quantized_raw

            indices_by_level.append(outputs.indices)
            losses.append(outputs.vq_loss.float())
            perplexities.append(outputs.perplexity.float())
            active_codes.append(outputs.active_codes.float())
            usage_counts.append(outputs.usage_counts.float())
            if outputs.embed_sum is not None:
                embed_sums.append(outputs.embed_sum.float())
            if outputs.restart_candidates is not None:
                restart_candidates.append(outputs.restart_candidates.float())

        weights = self.loss_weights.to(device=z.device, dtype=torch.float32)
        loss_stack = torch.stack(losses)
        vq_loss = (loss_stack * weights).sum() / weights.sum().clamp_min(1.0e-8)
        quantized_st = z + (quantized_sum - z).detach()

        embed_sum_tensor = torch.stack(embed_sums) if len(embed_sums) == self.num_quantizers else None
        restart_tensor = torch.stack(restart_candidates) if len(restart_candidates) == self.num_quantizers else None
        return QuantizerOutput(
            quantized=quantized_st,
            indices=torch.stack(indices_by_level, dim=1),
            vq_loss=vq_loss,
            perplexity=torch.stack(perplexities),
            active_codes=torch.stack(active_codes),
            usage_counts=torch.stack(usage_counts),
            embed_sum=embed_sum_tensor,
            restart_candidates=restart_tensor,
        )
