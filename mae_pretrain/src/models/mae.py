"""Masked Autoencoder wrapper built on top of PolyEncoder.

This module defines the MAE training shell (masking + decoder) while keeping
encoder reusable for downstream tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .encoder import PolyEncoder
from .pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViTPoly(nn.Module):
    """MAE model for polygon frequency-domain images.

    Args:
        img_size: Input image size `(H, W)`.
        patch_size: Patch edge length.
        in_chans: Input channel count.
        embed_dim: Encoder embedding dimension.
        depth: Encoder block count.
        num_heads: Encoder attention head count.
        decoder_embed_dim: Decoder embedding dimension.
        decoder_depth: Decoder block count.
        decoder_num_heads: Decoder attention head count.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (31, 31),
        patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 256,
        depth: int = 12,
        num_heads: int = 8,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
    ):
        """Initialize encoder, decoder, and MAE mask tokens."""
        super().__init__()

        self.encoder = PolyEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.encoder.patch_embed.num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    4.0,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize decoder position embeddings and decoder parameters."""
        decoder_pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.decoder_pos_embed.shape[-1],
            grid_size=self.encoder.patch_embed.grid_size,
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear/layernorm layers.

        Args:
            module: Layer module visited by `nn.Module.apply`.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @staticmethod
    def _validate_mask_ratio(mask_ratio: float) -> float:
        """Validate MAE mask ratio and return it as float."""
        mask_ratio = float(mask_ratio)
        if not (0.0 <= mask_ratio < 1.0):
            raise ValueError(f"`mask_ratio` must be in [0, 1), got {mask_ratio}")
        return mask_ratio

    def random_masking_ids(self, batch_size: int, token_count: int, device: torch.device, mask_ratio: float):
        """Generate random MAE masking indices.

        Args:
            batch_size: Batch size.
            token_count: Number of patch tokens per sample.
            device: Runtime device.
            mask_ratio: Fraction of tokens to mask.

        Returns:
            Tuple `(ids_keep, mask, ids_restore)`.
        """
        len_keep = int(token_count * (1.0 - mask_ratio))
        noise = torch.rand(batch_size, token_count, device=device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones([batch_size, token_count], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        """Encode visible tokens after MAE random masking.

        Args:
            x: Input tensor `[B, C, H, W]`.
            mask_ratio: Fraction of tokens to mask.

        Returns:
            Tuple `(latent, mask, ids_restore)`.
        """
        mask_ratio = self._validate_mask_ratio(mask_ratio)
        batch_size = x.shape[0]
        token_count = self.encoder.patch_embed.num_patches
        ids_keep, mask, ids_restore = self.random_masking_ids(batch_size, token_count, x.device, mask_ratio)
        latent = self.encoder.forward_features(x, ids_keep=ids_keep)
        return latent, mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Decode full token sequence from masked latent tokens.

        Args:
            latent: Latent token sequence from encoder.
            ids_restore: Restore indices generated by masking.

        Returns:
            Predicted patch values with shape `[B, L, p*p*C]`.
        """
        x = self.decoder_embed(latent)

        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        """Run full MAE forward process.

        Args:
            imgs: Input tensor `[B, C, H, W]`.
            mask_ratio: Fraction of patch tokens to mask.

        Returns:
            Tuple `(pred, mask)`.
        """
        mask_ratio = self._validate_mask_ratio(mask_ratio)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask
