"""Training engine for polygon VQAE pretraining.

This module implements:
1) Data loading and train/val split.
2) AE warmup -> VQ training loop with fp32/bf16/fp16 precision support.
3) DDP-aware metric reduction.
4) Checkpoint/export generation.
5) Reconstruction visualization for qualitative monitoring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..datasets.collate import triangle_collate_fn
from ..datasets.pt_manifest import PtShardManifest
from ..datasets.registry import get_geometry_codec
from ..datasets.shard_io import resolve_triangle_shard_paths
from ..datasets.sharded_pt_dataset import (
    EagerShardedPolyDataset,
    LazyShardedPolyDataset,
    load_all_samples_from_manifest,
)
from ..losses.recon_mag_phase import compute_mag_phase_losses
from ..models.factory import build_vqae_model_from_config
from ..utils.checkpoint import (
    load_latest_training_state,
    save_checkpoint,
    save_latest_training_state_pair,
    save_training_state,
)
from ..utils.config import dump_yaml_config
from ..utils.dist import (
    DistContext,
    all_reduce_mean,
    cleanup_distributed,
    distributed_barrier,
    init_distributed,
    is_main_process,
)
from ..utils.filesystem import ensure_dir, make_timestamped_dir
from ..utils.logger import attach_tee_stdout
from ..utils.precision import autocast_context, build_grad_scaler, normalize_precision
from ..utils.safe_load import register_numpy_safe_globals
from ..utils.seed import capture_rng_state, restore_rng_state, set_global_seed


_RESUME_LOCKED_KEYS = {
    "geom_type",
    "data_dir",
    "data_path",
    "stem_channels",
    "stem_strides",
    "embed_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "drop_rate",
    "decoder_stage_channels",
    "decoder_attention_type",
    "decoder_attention_heads",
    "decoder_attention_depths",
    "decoder_conv_depths",
    "decoder_window_size",
    "decoder_upsample_mode",
    "decoder_mlp_ratio",
    "decoder_drop_rate",
    "codebook_size",
    "code_dim",
    "vq_beta",
    "vq_decay",
    "vq_eps",
    "vq_dead_code_threshold",
    "vq_warmup_epochs",
    "vq_beta_warmup_epochs",
    "vq_init_max_vectors",
    "vq_kmeans_iters",
    "pos_freqs",
    "w_min",
    "w_max",
    "freq_type",
}


def mag_phase_to_real_imag(mag_log: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert log-magnitude and phase channels into real/imaginary components.

    Args:
        mag_log: Log-scaled magnitude tensor.
        phase: Phase tensor.

    Returns:
        Tuple `(real_part, imag_part)` in float tensors.
    """
    raw_mag = torch.expm1(mag_log)
    real_part = raw_mag * torch.cos(phase)
    imag_part = raw_mag * torch.sin(phase)
    return real_part, imag_part


def compute_freq_span_map(converter, device: torch.device) -> torch.Tensor:
    """Compute full-image frequency-span weighting map.

    Args:
        converter: Polygon Fourier converter instance.
        device: Runtime device.

    Returns:
        Frequency-span weights shaped `[1,1,H,W]`.
    """
    h, w = converter.U.shape
    valid_h = h - converter.pad_h
    valid_w = w - converter.pad_w

    wx = converter.U[:valid_h, 0].clone()
    wy = converter.V[0, :valid_w].clone()

    du = torch.zeros_like(wx)
    for i in range(valid_h):
        if i == 0:
            du[i] = wx[1] - wx[0] if valid_h > 1 else 1.0
        elif i == valid_h - 1:
            du[i] = wx[i] - wx[i - 1]
        else:
            du[i] = (wx[i + 1] - wx[i - 1]) / 2.0

    dv = torch.zeros_like(wy)
    for j in range(valid_w):
        if j == 0:
            dv[j] = wy[1] - wy[0] if valid_w > 1 else 1.0
        elif j == valid_w - 1:
            dv[j] = wy[j] - wy[j - 1]
        else:
            dv[j] = (wy[j + 1] - wy[j - 1]) / 2.0

    d_u, d_v = torch.meshgrid(du, dv, indexing="ij")
    freq_span_map = torch.zeros((h, w), device=device)
    freq_span_map[:valid_h, :valid_w] = d_u * d_v

    freq_span_map = torch.sqrt(freq_span_map.clamp_min(0.0))
    valid_mean = freq_span_map[:valid_h, :valid_w].mean()
    freq_span_map = freq_span_map / (valid_mean + 1e-8)

    return freq_span_map.unsqueeze(0).unsqueeze(0)


def compute_valid_mask(converter, device: torch.device) -> torch.Tensor:
    """Build one binary mask covering only valid non-padding frequency region."""
    h, w = converter.U.shape
    valid_h = h - converter.pad_h
    valid_w = w - converter.pad_w
    valid_mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    valid_mask[:valid_h, :valid_w] = 1.0
    return valid_mask.unsqueeze(0).unsqueeze(0)


def rasterize_tris_to_grid(tris: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Rasterize triangle set into binary occupancy grid.

    Args:
        tris: Triangle tensor `[T,3,2]` on CPU.
        height: Output raster height.
        width: Output raster width.

    Returns:
        Binary raster image `[H,W]` as float32 ndarray.
    """
    x = np.linspace(-1, 1, width)
    y = np.linspace(1, -1, height)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T

    mask = np.zeros(height * width, dtype=bool)
    for tri in tris.numpy():
        poly_path = MplPath(tri)
        mask = mask | poly_path.contains_points(points)

    return mask.reshape(height, width).astype(np.float32)


def plot_reconstruction(
    img_orig: torch.Tensor,
    img_masked: torch.Tensor,
    img_recon: torch.Tensor,
    spatial_gt: np.ndarray,
    spatial_icft_orig: torch.Tensor,
    spatial_icft_recon: torch.Tensor,
    epoch: int,
    save_dir: str | Path,
) -> None:
    """Save multi-panel reconstruction visualization for one validation sample.

    Args:
        img_orig: Original 3-channel frequency image `[3,H,W]`.
        img_masked: Masked input image `[3,H,W]`.
        img_recon: Reconstructed image `[3,H,W]`.
        spatial_gt: Spatial ground truth raster `[H,W]`.
        spatial_icft_orig: ICFT reconstruction from original spectrum.
        spatial_icft_recon: ICFT reconstruction from VQAE output.
        epoch: Current epoch index (1-based).
        save_dir: Directory for output images.
    """
    fig = plt.figure(figsize=(30, 12))
    # Use nested grids so we can tighten only the left frequency panels while
    # keeping right spatial panels enlarged and readable.
    outer = fig.add_gridspec(1, 2, width_ratios=[3.0, 5.1], wspace=0.10)
    gs_left = outer[0, 0].subgridspec(3, 3, wspace=0.02, hspace=0.24)
    gs_right = outer[0, 1].subgridspec(1, 3, wspace=0.20)

    vmin_mag, vmax_mag = img_orig[0].min(), img_orig[0].max()
    vmin_trig, vmax_trig = -1.0, 1.0

    vmin_sp = 0.0
    vmax_sp = max(1.0, spatial_icft_orig.max().item(), spatial_icft_recon.max().item())

    axes_left = [
        [fig.add_subplot(gs_left[0, 0]), fig.add_subplot(gs_left[0, 1]), fig.add_subplot(gs_left[0, 2])],
        [fig.add_subplot(gs_left[1, 0]), fig.add_subplot(gs_left[1, 1]), fig.add_subplot(gs_left[1, 2])],
        [fig.add_subplot(gs_left[2, 0]), fig.add_subplot(gs_left[2, 1]), fig.add_subplot(gs_left[2, 2])],
    ]
    axes_right = [fig.add_subplot(gs_right[0, 0]), fig.add_subplot(gs_right[0, 1]), fig.add_subplot(gs_right[0, 2])]

    titles_left = [
        ["Original Mag", "Input Mag", "Reconstructed Mag"],
        ["Original Cos(Phase)", "Input Cos(Phase)", "Reconstructed Cos(Phase)"],
        ["Original Sin(Phase)", "Input Sin(Phase)", "Reconstructed Sin(Phase)"],
    ]
    titles_right = ["Strict Spatial GT (0/1)", "ICFT (Orig Freqs)", "ICFT (AE Recon Freqs)"]

    data_left = [
        [(img_orig[0], vmin_mag, vmax_mag), (img_masked[0], vmin_mag, vmax_mag), (img_recon[0], vmin_mag, vmax_mag)],
        [(img_orig[1], vmin_trig, vmax_trig), (img_masked[1], vmin_trig, vmax_trig), (img_recon[1], vmin_trig, vmax_trig)],
        [(img_orig[2], vmin_trig, vmax_trig), (img_masked[2], vmin_trig, vmax_trig), (img_recon[2], vmin_trig, vmax_trig)],
    ]
    data_right = [(spatial_gt, vmin_sp, vmax_sp), (spatial_icft_orig, vmin_sp, vmax_sp), (spatial_icft_recon, vmin_sp, vmax_sp)]

    for row in range(3):
        for col in range(3):
            ax = axes_left[row][col]
            data, vmin, vmax = data_left[row][col]
            data_np = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
            data_np = np.ma.masked_invalid(data_np)
            im = ax.imshow(
                data_np,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation="nearest",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(titles_left[row][col], fontsize=14)
            ax.axis("off")
            # Keep colorbars compact so left frequency columns stay visually tight.
            plt.colorbar(im, ax=ax, fraction=0.040, pad=0.015)

    for col in range(3):
        ax = axes_right[col]
        data, vmin, vmax = data_right[col]
        data_np = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
        data_np = np.ma.masked_invalid(data_np)
        im = ax.imshow(
            data_np,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles_right[col], fontsize=14)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # `tight_layout` may be unstable with this mixed GridSpec + colorbar layout.
    # Use explicit margins for deterministic panel placement across matplotlib versions.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"recon_epoch_{epoch}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _build_model_kwargs(args, img_size: tuple[int, int]) -> dict:
    """Build one shared VQAE model-kwargs dict from runtime args."""
    return {
        "img_size": img_size,
        "in_chans": 3,
        "stem_channels": args.stem_channels,
        "stem_strides": args.stem_strides,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "drop_rate": args.drop_rate,
        "decoder_stage_channels": args.decoder_stage_channels,
        "decoder_attention_type": args.decoder_attention_type,
        "decoder_attention_heads": args.decoder_attention_heads,
        "decoder_attention_depths": args.decoder_attention_depths,
        "decoder_conv_depths": args.decoder_conv_depths,
        "decoder_window_size": args.decoder_window_size,
        "decoder_upsample_mode": args.decoder_upsample_mode,
        "decoder_mlp_ratio": args.decoder_mlp_ratio,
        "decoder_drop_rate": args.decoder_drop_rate,
        "codebook_size": args.codebook_size,
        "code_dim": args.code_dim,
        "vq_decay": args.vq_decay,
        "vq_eps": args.vq_eps,
        "vq_dead_code_threshold": args.vq_dead_code_threshold,
    }


def _build_model_config(args, img_size: tuple[int, int]) -> dict:
    """Build persisted model config dict for downstream reload.

    Args:
        args: Parsed training arguments.
        img_size: Runtime image size inferred from Fourier grid.

    Returns:
        Serializable model config dict.
    """
    return {
        "geom_type": args.geom_type,
        "train_type": args.train_type,
        **_build_model_kwargs(args, img_size=img_size),
        "pos_freqs": args.pos_freqs,
        "w_min": args.w_min,
        "w_max": args.w_max,
        "freq_type": args.freq_type,
    }


def _build_model(args, img_size: tuple[int, int], device: torch.device, dist_ctx: DistContext):
    """Construct VQAE model and optional DDP wrapper.

    Args:
        args: Parsed training arguments.
        img_size: Runtime image size.
        device: Runtime device.
        dist_ctx: Distributed context.

    Returns:
        Model instance (possibly DDP-wrapped).
    """
    model = build_vqae_model_from_config(_build_model_kwargs(args, img_size=img_size), device=device, precision="fp32")

    if dist_ctx.enabled and dist_ctx.world_size > 1:
        if device.type == "cuda":
            model = DDP(model, device_ids=[dist_ctx.local_rank], broadcast_buffers=False)
        else:
            model = DDP(model, broadcast_buffers=False)
    return model


def _build_scheduler(args, optimizer):
    """Construct the epoch scheduler from runtime arguments."""
    if args.warmup_epochs > 0:
        min_factor = float(args.min_lr) / float(args.lr)
        cosine_epochs = max(1, int(args.epochs) - int(args.warmup_epochs))

        class _WarmupThenCosineLambda:
            def __init__(self, warmup_epochs: int, cosine_epochs: int, min_factor: float) -> None:
                self.warmup_epochs = int(warmup_epochs)
                self.cosine_epochs = int(cosine_epochs)
                self.min_factor = float(min_factor)

            def __call__(self, epoch: int) -> float:
                current_epoch = max(0, int(epoch))
                if current_epoch <= self.warmup_epochs:
                    warmup_progress = current_epoch / float(self.warmup_epochs)
                    return self.min_factor + (1.0 - self.min_factor) * warmup_progress

                cosine_progress = min(current_epoch - self.warmup_epochs, self.cosine_epochs)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_progress / self.cosine_epochs))
                return self.min_factor + (1.0 - self.min_factor) * cosine_factor

        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_WarmupThenCosineLambda(
                warmup_epochs=args.warmup_epochs,
                cosine_epochs=cosine_epochs,
                min_factor=min_factor,
            ),
        )

    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )


def _validate_training_args(args) -> None:
    """Validate numeric hyperparameters before building the training pipeline."""
    def _parse_csv_ints(name: str, value: str) -> list[int]:
        try:
            items = [int(item.strip()) for item in str(value).split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError(f"`{name}` must be a comma-separated integer list, got {value!r}") from exc
        if not items:
            raise ValueError(f"`{name}` must not be empty")
        return items

    try:
        augment_float = float(args.augment_times)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`augment_times` must be an integer >= 1, got {args.augment_times!r}") from exc

    if args.lr <= 0:
        raise ValueError(f"`lr` must be > 0, got {args.lr}")
    if args.min_lr < 0:
        raise ValueError(f"`min_lr` must be >= 0, got {args.min_lr}")
    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.depth <= 0:
        raise ValueError(f"`depth` must be > 0, got {args.depth}")
    if args.num_heads <= 0:
        raise ValueError(f"`num_heads` must be > 0, got {args.num_heads}")
    if not (0.0 < float(args.val_ratio) < 1.0):
        raise ValueError(f"`val_ratio` must be in (0, 1), got {args.val_ratio}")
    if not augment_float.is_integer():
        raise ValueError(f"`augment_times` must be an integer >= 1, got {args.augment_times!r}")
    if int(augment_float) < 1:
        raise ValueError(f"`augment_times` must be >= 1, got {args.augment_times}")
    if args.warmup_epochs < 0:
        raise ValueError(f"`warmup_epochs` must be >= 0, got {args.warmup_epochs}")
    if args.log_interval <= 0:
        raise ValueError(f"`log_interval` must be > 0, got {args.log_interval}")
    if args.codebook_size <= 1:
        raise ValueError(f"`codebook_size` must be > 1, got {args.codebook_size}")
    if args.code_dim <= 0:
        raise ValueError(f"`code_dim` must be > 0, got {args.code_dim}")
    if args.vq_beta < 0:
        raise ValueError(f"`vq_beta` must be >= 0, got {args.vq_beta}")
    if not (0.0 < float(args.vq_decay) < 1.0):
        raise ValueError(f"`vq_decay` must be in (0, 1), got {args.vq_decay}")
    if args.vq_eps <= 0:
        raise ValueError(f"`vq_eps` must be > 0, got {args.vq_eps}")
    if args.vq_warmup_epochs < 0:
        raise ValueError(f"`vq_warmup_epochs` must be >= 0, got {args.vq_warmup_epochs}")
    if args.vq_beta_warmup_epochs < 0:
        raise ValueError(f"`vq_beta_warmup_epochs` must be >= 0, got {args.vq_beta_warmup_epochs}")
    if args.vq_init_max_vectors <= 0:
        raise ValueError(f"`vq_init_max_vectors` must be > 0, got {args.vq_init_max_vectors}")
    if args.vq_kmeans_iters <= 0:
        raise ValueError(f"`vq_kmeans_iters` must be > 0, got {args.vq_kmeans_iters}")
    if args.freq_type == "geometric" and args.w_min <= 0:
        raise ValueError(f"`w_min` must be > 0 for geometric freq grids, got {args.w_min}")
    if args.w_max < args.w_min:
        raise ValueError(f"`w_max` must be >= `w_min`, got w_min={args.w_min}, w_max={args.w_max}")

    stem_channels = _parse_csv_ints("stem_channels", args.stem_channels)
    stem_strides = _parse_csv_ints("stem_strides", args.stem_strides)
    if len(stem_channels) != len(stem_strides):
        raise ValueError("`stem_channels` and `stem_strides` must have the same length")

    decoder_stage_channels = _parse_csv_ints("decoder_stage_channels", args.decoder_stage_channels)
    decoder_attention_heads = _parse_csv_ints("decoder_attention_heads", args.decoder_attention_heads)
    decoder_attention_depths = _parse_csv_ints("decoder_attention_depths", args.decoder_attention_depths)
    decoder_conv_depths = _parse_csv_ints("decoder_conv_depths", args.decoder_conv_depths)
    stage_count = len(stem_strides)
    if len(decoder_stage_channels) != stage_count:
        raise ValueError(
            f"`decoder_stage_channels` length must equal encoder stem stage count {stage_count}, "
            f"got {len(decoder_stage_channels)}"
        )
    if not (len(decoder_attention_heads) == len(decoder_attention_depths) == len(decoder_conv_depths) == stage_count):
        raise ValueError("Decoder stage config lengths must all match the encoder stem stage count")


def _advance_scheduler(scheduler, completed_epochs: int) -> None:
    """Advance one freshly-built epoch scheduler to the resumed epoch index."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(max(0, int(completed_epochs))):
            scheduler.step()


def _normalize_config_value(key: str, value):
    """Normalize config values before resume-compatibility comparison."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    if key.endswith("_dir") or key.endswith("_path"):
        return str(Path(str(value)).expanduser().resolve())
    return value


def _validate_resume_config(args, saved_run_config: dict) -> None:
    """Validate that resume-time overrides keep checkpoint compatibility."""
    for key in _RESUME_LOCKED_KEYS:
        current_value = _normalize_config_value(key, getattr(args, key, None))
        saved_value = _normalize_config_value(key, saved_run_config.get(key))
        if current_value != saved_value:
            raise ValueError(
                f"Resume config mismatch for `{key}`: current={current_value}, saved={saved_value}"
            )


def _validate_resume_manifest(resume_state: dict, pt_files: list[Path], manifest: PtShardManifest) -> None:
    """Validate saved data manifest metadata against current runtime files."""
    saved_pt_files = [str(Path(path).expanduser().resolve()) for path in resume_state.get("pt_files", [])]
    current_pt_files = [str(path.expanduser().resolve()) for path in pt_files]
    if saved_pt_files and saved_pt_files != current_pt_files:
        raise ValueError("Resume data files do not match the checkpoint manifest.")

    saved_total_samples = resume_state.get("manifest_total_samples")
    if saved_total_samples is not None and int(saved_total_samples) != int(manifest.total_samples):
        raise ValueError(
            "Resume dataset sample count does not match the checkpoint manifest: "
            f"current={manifest.total_samples}, saved={saved_total_samples}"
        )


def _move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    """Move optimizer state tensors onto the current runtime device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _apply_optimizer_runtime_overrides(optimizer, args) -> None:
    """Apply resume-time optimizer hyperparameter overrides."""
    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["weight_decay"] = args.weight_decay
        group["initial_lr"] = args.lr


def _is_eval_epoch(epoch_index: int, total_epochs: int, eval_every: int) -> bool:
    """Check whether current epoch should run validation/checkpoint logic."""
    epoch_number = epoch_index + 1
    return epoch_number % eval_every == 0 or epoch_number == total_epochs


def _write_json(path: str | Path, payload: dict) -> Path:
    """Write one JSON file with UTF-8 encoding and pretty formatting."""
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=4, ensure_ascii=False)
    return out_path


def _sync_run_metadata(best_dir: Path, ckpt_dir: Path, run_config: dict, model_config: dict) -> None:
    """Persist training config and model config into both `best/` and `ckpt/`."""
    dump_yaml_config(run_config, best_dir / "config.yaml")
    dump_yaml_config(run_config, ckpt_dir / "config.yaml")
    _write_json(best_dir / "poly_vqae_config.json", model_config)
    _write_json(ckpt_dir / "poly_vqae_config.json", model_config)


def _should_use_vq(epoch_index: int, vq_warmup_epochs: int) -> bool:
    """Whether current epoch should enable quantization."""
    return int(epoch_index) >= int(vq_warmup_epochs)


def _effective_vq_beta(epoch_index: int, args) -> float:
    """Compute current epoch VQ beta with post-warmup linear ramp."""
    if not _should_use_vq(epoch_index, args.vq_warmup_epochs):
        return 0.0
    if args.vq_beta_warmup_epochs <= 0:
        return float(args.vq_beta)

    beta_epoch = max(0, int(epoch_index) - int(args.vq_warmup_epochs))
    ramp_progress = min(float(beta_epoch) / float(args.vq_beta_warmup_epochs), 1.0)
    return float(args.vq_beta) * ramp_progress


@torch.no_grad()
def _collect_vq_init_vectors(
    model_to_save,
    train_loader,
    codec,
    device: torch.device,
    max_vectors: int,
) -> torch.Tensor:
    """Collect one bounded latent-vector pool for codebook initialization."""
    vector_chunks: list[torch.Tensor] = []
    remaining = int(max_vectors)

    model_to_save.eval()
    for batch_tris, lengths in train_loader:
        mag, phase = codec.cft_batch(batch_tris, lengths)
        imgs = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)
        code_features = model_to_save.encode_to_code_features(imgs).detach().float()
        vectors = code_features.permute(0, 2, 3, 1).reshape(-1, code_features.shape[1])

        if vectors.shape[0] > remaining:
            perm = torch.randperm(vectors.shape[0], device=vectors.device)[:remaining]
            vectors = vectors[perm]

        vector_chunks.append(vectors)
        remaining -= int(vectors.shape[0])
        if remaining <= 0:
            break

    if not vector_chunks:
        raise RuntimeError("Failed to collect any latent vectors for VQ codebook initialization.")
    return torch.cat(vector_chunks, dim=0)


def _broadcast_quantizer_state(model_to_save, dist_ctx: DistContext) -> None:
    """Broadcast initialized quantizer buffers from rank 0 to all ranks."""
    if not (dist_ctx.enabled and dist_ctx.world_size > 1):
        return
    for buffer in (
        model_to_save.quantizer.codebook,
        model_to_save.quantizer.cluster_size,
        model_to_save.quantizer.embed_avg,
        model_to_save.quantizer.initialized,
    ):
        dist.broadcast(buffer, src=0)


def _resolve_training_pt_files(args, warn_fn=None) -> list[Path]:
    """Resolve training shard files from directory input or compatibility path.

    Args:
        args: Parsed training arguments namespace.
        warn_fn: Optional warning sink for manifest fallback cases.

    Returns:
        Ordered absolute `.pt` shard paths.
    """
    data_dir = getattr(args, "data_dir", None)
    if data_dir:
        data_dir = Path(str(data_dir)).expanduser().resolve()
        return resolve_triangle_shard_paths(data_dir, warn_fn=warn_fn)

    data_path = getattr(args, "data_path", None)
    if data_path:
        resolved = Path(str(data_path)).expanduser().resolve()
        if resolved.is_file():
            return [resolved]
        if resolved.is_dir():
            return resolve_triangle_shard_paths(resolved, warn_fn=warn_fn)

    raise ValueError("Training data source is missing. Please provide --data_dir.")


def _split_dataset_indices(total_size: int, val_ratio: float, split_seed: int) -> tuple[list[int], list[int]]:
    """Split global sample indices into train/validation subsets.

    Args:
        total_size: Total number of samples in the manifest.
        val_ratio: Validation ratio.
        split_seed: Random seed for split reproducibility.

    Returns:
        Tuple `(train_indices, val_indices)`.
    """
    if total_size < 2:
        raise ValueError("Training requires at least two samples to build train/val splits.")

    val_size = min(total_size - 1, max(1, int(total_size * val_ratio)))
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return train_indices, val_indices


def _build_datasets(
    args,
    manifest: PtShardManifest,
    dist_ctx: DistContext,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
):
    """Build train/validation datasets for eager or lazy loading.

    Args:
        args: Parsed training arguments.
        manifest: Manifest of all training shard files.
        dist_ctx: Distributed context.

    Returns:
        Tuple `(train_dataset, val_dataset, cache_shards)`.
    """
    if train_indices is None or val_indices is None:
        train_indices, val_indices = _split_dataset_indices(
            total_size=manifest.total_samples,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
        )
    else:
        train_indices = [int(index) for index in train_indices]
        val_indices = [int(index) for index in val_indices]

    if args.load_mode == "eager":
        all_samples = load_all_samples_from_manifest(manifest)
        train_dataset = EagerShardedPolyDataset(
            all_samples=all_samples,
            sample_indices=train_indices,
            augment_times=args.augment_times,
        )
        val_dataset = EagerShardedPolyDataset(
            all_samples=all_samples,
            sample_indices=val_indices,
            augment_times=1,
        )
        return train_dataset, val_dataset, None, train_indices, val_indices

    max_cached_shards = manifest.recommend_cache_shards(
        world_size=dist_ctx.world_size,
        num_workers=args.num_workers,
    )
    train_dataset = LazyShardedPolyDataset(
        manifest=manifest,
        sample_indices=train_indices,
        augment_times=args.augment_times,
        max_cached_shards=max_cached_shards,
    )
    val_dataset = LazyShardedPolyDataset(
        manifest=manifest,
        sample_indices=val_indices,
        augment_times=1,
        max_cached_shards=max_cached_shards,
    )
    return train_dataset, val_dataset, max_cached_shards, train_indices, val_indices


def _build_loaders(args, train_dataset, val_dataset, dist_ctx: DistContext):
    """Build train/validation dataloaders from prepared datasets.

    Args:
        args: Parsed training args.
        train_dataset: Training dataset object.
        val_dataset: Validation dataset object.
        dist_ctx: Distributed context.

    Returns:
        Tuple `(train_dataset, val_dataset, train_loader, val_loader, train_sampler)`.
    """
    persistent_workers = bool(args.num_workers > 0 and args.load_mode == "lazy")

    use_distributed_sampler = dist_ctx.enabled and dist_ctx.world_size > 1
    common_loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": triangle_collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": persistent_workers,
    }

    if use_distributed_sampler:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            **common_loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            **common_loader_kwargs,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **common_loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **common_loader_kwargs,
        )

    return train_dataset, val_dataset, train_loader, val_loader, train_sampler


def _prepare_fixed_visual_sample(args, codec, val_dataset, device: torch.device):
    """Prepare one fixed validation sample for per-epoch reconstruction plots.

    Args:
        args: Parsed training args.
        codec: Polygon geometry codec.
        val_dataset: Validation dataset.
        device: Runtime device.

    Returns:
        Tuple `(fixed_batch_tris, fixed_lengths, spatial_gt, spatial_icft_orig)`.
    """
    fixed_tris_orig = val_dataset.get_base_sample(0)
    fixed_tris = torch.tensor(fixed_tris_orig, dtype=torch.float32)

    fixed_batch_tris = fixed_tris.unsqueeze(0).to(device)
    fixed_lengths = torch.tensor([fixed_tris.shape[0]], device=device)

    with torch.no_grad():
        mag_fix, phase_fix = codec.cft_batch(fixed_batch_tris, fixed_lengths)
        real_fix, imag_fix = mag_phase_to_real_imag(mag_fix, phase_fix)
        spatial_icft_orig = codec.icft_2d(real_fix.squeeze(1), imag_fix.squeeze(1))[0].detach().cpu()
        h_sp, w_sp = spatial_icft_orig.shape[-2], spatial_icft_orig.shape[-1]
        spatial_gt = rasterize_tris_to_grid(fixed_tris.cpu(), h_sp, w_sp)

    return fixed_batch_tris, fixed_lengths, spatial_gt, spatial_icft_orig


def _accumulate_epoch_metrics(metric_sums: dict[str, float], step_outputs: dict[str, torch.Tensor]) -> None:
    """Accumulate scalar metrics from one model step into running sums."""
    metric_sums["total"] += float(step_outputs["loss_total"].item())
    metric_sums["mag"] += float(step_outputs["loss_mag"].item())
    metric_sums["phase"] += float(step_outputs["loss_phase"].item())
    metric_sums["vq"] += float(step_outputs["weighted_vq"].item())
    metric_sums["perplexity"] += float(step_outputs["outputs"].perplexity.item())
    metric_sums["active"] += float(step_outputs["outputs"].active_codes.item())


def _reduce_epoch_metrics(
    metric_sums: dict[str, float],
    steps: int,
    *,
    device: torch.device,
    dist_ctx: DistContext,
) -> dict[str, torch.Tensor]:
    """Average and all-reduce one epoch metric bundle."""
    step_count = max(1, int(steps))
    return {
        name: all_reduce_mean(torch.tensor(value / step_count, device=device), dist_ctx)
        for name, value in metric_sums.items()
    }


def _format_epoch_metrics(prefix: str, metrics: dict[str, torch.Tensor]) -> str:
    """Format one train/val epoch metric line."""
    return (
        f"  [{prefix}] Total: {metrics['total'].item():.4f} | "
        f"Mag: {metrics['mag'].item():.4f} | Phase: {metrics['phase'].item():.4f} | "
        f"VQ: {metrics['vq'].item():.4f} | Perplexity: {metrics['perplexity'].item():.2f} | "
        f"ActiveCodes: {int(metrics['active'].item())}"
    )


def _save_fixed_visualization(
    *,
    model,
    codec,
    fixed_batch_tris: torch.Tensor,
    fixed_lengths: torch.Tensor,
    spatial_gt,
    spatial_icft_orig: torch.Tensor,
    device: torch.device,
    precision: str,
    use_vq: bool,
    epoch: int,
    save_dir: Path,
) -> None:
    """Render and save one fixed validation visualization snapshot."""
    with torch.no_grad():
        mag_fix, phase_fix = codec.cft_batch(fixed_batch_tris, fixed_lengths)
        imgs_fix = torch.cat([mag_fix, torch.cos(phase_fix), torch.sin(phase_fix)], dim=1)

        with autocast_context(device, precision):
            outputs_fix = model(imgs_fix, use_vq=use_vq)

        img_orig = imgs_fix[0].cpu()
        img_masked = img_orig.clone()
        img_recon = outputs_fix.recon_imgs[0].float().cpu()

        mag_recon = img_recon[0].unsqueeze(0).to(device)
        cos_recon = img_recon[1].unsqueeze(0).to(device)
        sin_recon = img_recon[2].unsqueeze(0).to(device)

        phase_recon = torch.atan2(sin_recon, cos_recon)
        real_recon, imag_recon = mag_phase_to_real_imag(mag_recon, phase_recon)
        spatial_icft_recon = codec.icft_2d(real_recon, imag_recon)[0].squeeze().cpu()

        plot_reconstruction(
            img_orig=img_orig,
            img_masked=img_masked,
            img_recon=img_recon,
            spatial_gt=spatial_gt,
            spatial_icft_orig=spatial_icft_orig.squeeze(),
            spatial_icft_recon=spatial_icft_recon,
            epoch=epoch + 1,
            save_dir=save_dir,
        )


def _build_nonfinite_tensor_map(
    *,
    stage: str,
    imgs: torch.Tensor,
    recon_imgs: torch.Tensor,
    loss_mag: torch.Tensor,
    loss_phase: torch.Tensor,
    vq_loss: torch.Tensor,
    loss_total: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build one standardized tensor bundle for non-finite diagnostics."""
    suffix = "_val" if stage == "val_loss" else ""
    return {
        f"imgs{suffix}": imgs,
        f"recon_imgs{suffix}": recon_imgs,
        f"loss_mag{suffix}": loss_mag,
        f"loss_phase{suffix}": loss_phase,
        f"loss_vq{suffix}": vq_loss,
        f"loss_total{suffix}": loss_total,
    }


def _run_model_step(
    *,
    model,
    codec,
    batch_tris,
    lengths,
    device: torch.device,
    precision: str,
    use_vq: bool,
    freq_span_map: torch.Tensor,
    valid_mask: torch.Tensor,
    args,
    current_vq_beta: float,
    epoch: int,
    step: int,
    stage: str,
) -> dict[str, torch.Tensor]:
    """Run one shared forward+loss step for both train and validation."""
    with torch.no_grad():
        mag, phase = codec.cft_batch(batch_tris, lengths)
        imgs = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)

    with autocast_context(device, precision):
        outputs = model(imgs, use_vq=use_vq)

    recon_imgs = outputs.recon_imgs.float()
    vq_loss = outputs.vq_loss.float()

    loss_mag, loss_phase = compute_mag_phase_losses(
        pred_imgs=recon_imgs,
        target_imgs=imgs.float(),
        freq_span_map=freq_span_map,
        valid_mask=valid_mask,
        weight_mag_hf=args.weight_mag_hf,
    )
    recon_loss = args.weight_mag * loss_mag + args.weight_phase * loss_phase
    weighted_vq = current_vq_beta * vq_loss
    loss_total = recon_loss + weighted_vq

    if not bool(torch.isfinite(loss_total).item()):
        _raise_nonfinite_training_error(
            stage=stage,
            epoch=epoch,
            step=step,
            tensors=_build_nonfinite_tensor_map(
                stage=stage,
                imgs=imgs,
                recon_imgs=recon_imgs,
                loss_mag=loss_mag,
                loss_phase=loss_phase,
                vq_loss=vq_loss,
                loss_total=loss_total,
            ),
        )

    return {
        "imgs": imgs,
        "outputs": outputs,
        "recon_imgs": recon_imgs,
        "loss_mag": loss_mag,
        "loss_phase": loss_phase,
        "vq_loss": vq_loss,
        "weighted_vq": weighted_vq,
        "loss_total": loss_total,
    }

def count_parameters(model):
    model = model.module if hasattr(model, 'module') else model
    # 计算总参数量 (单位: 百万 M)
    total_params = sum(p.numel() for p in model.parameters())
    
    # 计算 Encoder 部分参数量
    # 假设你的模型中 encoder 对象的变量名是 'encoder'
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    
    print(f"总参数量: {total_params / 1e6:.2f} M")
    print(f"Encoder 参数量: {encoder_params / 1e6:.2f} M")
    print(f"Encoder 占比: {(encoder_params / total_params) * 100:.2f}%")


def _describe_tensor_finiteness(name: str, tensor: torch.Tensor) -> str:
    """Build a concise finiteness summary for one tensor."""
    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    total_count = int(detached.numel())
    finite_count = int(finite_mask.sum().item())
    nonfinite_count = total_count - finite_count
    if finite_count > 0:
        finite_values = detached[finite_mask]
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
        mean_value = float(finite_values.mean().item())
        return (
            f"{name}: shape={tuple(detached.shape)}, "
            f"nonfinite={nonfinite_count}/{total_count}, "
            f"min={min_value:.4e}, max={max_value:.4e}, mean={mean_value:.4e}"
        )
    return f"{name}: shape={tuple(detached.shape)}, nonfinite={nonfinite_count}/{total_count}, no finite values"


def _raise_nonfinite_training_error(stage: str, epoch: int, step: int, tensors: dict[str, torch.Tensor]) -> None:
    """Raise a detailed runtime error once non-finite tensors are observed."""
    summaries = [
        _describe_tensor_finiteness(name, tensor)
        for name, tensor in tensors.items()
    ]
    raise RuntimeError(
        f"Non-finite tensor detected during {stage} at epoch={epoch + 1}, step={step}: "
        + " | ".join(summaries)
    )

def train_main(args) -> None:
    """Main training entrypoint."""
    register_numpy_safe_globals()
    args.load_mode = str(args.load_mode).lower()
    args.resume_dir = str(Path(args.resume_dir).expanduser().resolve()) if args.resume_dir else None
    args.eval_every = max(1, int(args.eval_every))
    _validate_training_args(args)

    set_global_seed(args.seed, deterministic=args.deterministic)

    dist_ctx = init_distributed()
    device = dist_ctx.device

    args.precision = normalize_precision(args.precision)
    args.checkpoint_dtype = normalize_precision(args.checkpoint_dtype)
    scaler = build_grad_scaler(device=device, precision=args.precision)

    max_train_steps = max(0, int(getattr(args, "max_train_steps", 0)))
    max_val_steps = max(0, int(getattr(args, "max_val_steps", 0)))

    resume_state = None
    resume_path = None
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume_dir:
        resume_state, resume_path = load_latest_training_state(args.resume_dir)
        saved_run_config = dict(resume_state.get("run_config", {}))
        _validate_resume_config(args, saved_run_config)
        start_epoch = int(resume_state.get("completed_epoch", 0))
        best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
        if args.epochs <= start_epoch:
            raise ValueError(
                f"Resume target epochs must exceed completed epochs: completed={start_epoch}, target={args.epochs}"
            )

    if is_main_process(dist_ctx):
        if args.resume_dir:
            run_dir = Path(args.resume_dir)
            run_timestamp = run_dir.name
        else:
            run_dir, run_timestamp = make_timestamped_dir(args.save_dir)

        best_dir = ensure_dir(run_dir / "best")
        ckpt_dir = ensure_dir(run_dir / "ckpt")
        viz_dir = ensure_dir(ckpt_dir / "viz")
        attach_tee_stdout(best_dir / "train_log.txt")

        print("\n[INFO] ========================================================")
        print(f"[INFO] Run directory   : {run_dir}")
        print(f"[INFO] Precision      : train={args.precision}, ckpt={args.checkpoint_dtype}")
        print(
            "[INFO] Distributed    : "
            f"enabled={dist_ctx.enabled}, world_size={dist_ctx.world_size}, "
            f"rank={dist_ctx.rank}, local_rank={dist_ctx.local_rank}, device={device}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
        print(f"[INFO] Eval frequency  : every {args.eval_every} epoch(s)")
        print(
            "[INFO] Warmups        : "
            f"lr={args.warmup_epochs}, vq={args.vq_warmup_epochs}, beta={args.vq_beta_warmup_epochs}"
        )
        if args.resume_dir:
            print(f"[INFO] Resume mode    : dir={args.resume_dir}, checkpoint={resume_path}")
            print(f"[INFO] Resume state   : completed_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")
        if max_train_steps > 0 or max_val_steps > 0:
            print(
                "[INFO] Debug limits   : "
                f"max_train_steps={max_train_steps or 'full'}, "
                f"max_val_steps={max_val_steps or 'full'}"
            )
        print("[INFO] ========================================================\n")
    else:
        run_dir = Path(args.resume_dir) if args.resume_dir else Path(args.save_dir)
        run_timestamp = run_dir.name if args.resume_dir else "distributed_worker"
        best_dir = run_dir / "best"
        ckpt_dir = run_dir / "ckpt"
        viz_dir = ckpt_dir / "viz"

    codec = get_geometry_codec(args.geom_type, vars(args), device=str(device))
    converter = codec.converter
    img_size = (converter.U.shape[0], converter.U.shape[1])
    model_config = _build_model_config(args, img_size=img_size)
    run_config = dict(vars(args))

    if is_main_process(dist_ctx):
        print(f"[INFO] CFT tri chunk   : {converter.triangle_chunk_size}")
        print(f"[INFO] ICFT spatial chunk: {converter.icft_spatial_chunk_size}")

    freq_span_map = compute_freq_span_map(converter, device=device)
    valid_mask = compute_valid_mask(converter, device=device)

    model = _build_model(args, img_size=img_size, device=device, dist_ctx=dist_ctx)
    model_to_save = model.module if isinstance(model, DDP) else model
    model_config["latent_stride"] = int(model_to_save.encoder.latent_stride)
    model_config["latent_grid_size"] = tuple(int(v) for v in model_to_save.encoder.latent_grid_size)
    model_config["num_latent_tokens"] = int(model_to_save.encoder.num_latent_tokens)
    run_config["latent_stride"] = model_config["latent_stride"]
    run_config["latent_grid_size"] = model_config["latent_grid_size"]
    run_config["num_latent_tokens"] = model_config["num_latent_tokens"]
    if is_main_process(dist_ctx):
        print(f"[INFO] Latent stride  : {model_config['latent_stride']}")
        print(f"[INFO] Latent grid    : {model_config['latent_grid_size']}")
        print(f"[INFO] Latent tokens  : {model_config['num_latent_tokens']}")
        _sync_run_metadata(best_dir=best_dir, ckpt_dir=ckpt_dir, run_config=run_config, model_config=model_config)

    if resume_state is not None:
        model_to_save.load_state_dict(resume_state["model_state"], strict=True)

    if is_main_process(dist_ctx):
        count_parameters(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        _move_optimizer_state_to_device(optimizer, device=device)
        _apply_optimizer_runtime_overrides(optimizer, args)

    scheduler = _build_scheduler(args, optimizer)
    if start_epoch > 0:
        _advance_scheduler(scheduler, completed_epochs=start_epoch)

    if resume_state is not None:
        saved_precision = normalize_precision(resume_state.get("train_precision", args.precision))
        scaler_state = resume_state.get("scaler_state")
        if scaler_state and scaler.is_enabled() and saved_precision == args.precision:
            scaler.load_state_dict(scaler_state)

    pt_files = _resolve_training_pt_files(
        args,
        warn_fn=(lambda message: print(message)) if is_main_process(dist_ctx) else None,
    )
    manifest = PtShardManifest.from_pt_files(pt_files)

    if resume_state is not None:
        _validate_resume_manifest(resume_state, pt_files=pt_files, manifest=manifest)
        saved_train_indices = resume_state.get("train_indices")
        saved_val_indices = resume_state.get("val_indices")
        if saved_train_indices is None or saved_val_indices is None:
            raise ValueError("Resume checkpoint is missing saved dataset split indices.")
    else:
        saved_train_indices = None
        saved_val_indices = None

    train_dataset, val_dataset, cache_shards, train_indices, val_indices = _build_datasets(
        args,
        manifest,
        dist_ctx,
        train_indices=saved_train_indices,
        val_indices=saved_val_indices,
    )

    if is_main_process(dist_ctx):
        total_size_mb = manifest.total_size_bytes / (1024.0 * 1024.0)
        print(
            "[INFO] Data pipeline   : "
            f"mode={args.load_mode}, shards={manifest.num_shards}, "
            f"samples={manifest.total_samples}, size={total_size_mb:.2f} MB"
        )
        print(f"[INFO] Data directory  : {pt_files[0].parent}")
        if cache_shards is not None:
            print(f"[INFO] Lazy cache     : max_cached_shards={cache_shards} per dataset instance")

    train_dataset, val_dataset, train_loader, val_loader, train_sampler = _build_loaders(
        args,
        train_dataset,
        val_dataset,
        dist_ctx,
    )

    if is_main_process(dist_ctx):
        fixed_batch_tris, fixed_lengths, spatial_gt, spatial_icft_orig = _prepare_fixed_visual_sample(
            args,
            codec,
            val_dataset,
            device,
        )
    else:
        fixed_batch_tris = fixed_lengths = spatial_gt = spatial_icft_orig = None

    if resume_state is not None:
        restore_rng_state(resume_state.get("rng_state"))

    try:
        for epoch in range(start_epoch, args.epochs):
            if dist_ctx.enabled and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            use_vq = _should_use_vq(epoch, args.vq_warmup_epochs)
            current_vq_beta = _effective_vq_beta(epoch, args)

            if use_vq and not model_to_save.quantizer.is_initialized:
                if is_main_process(dist_ctx):
                    print(
                        f"[INFO] Initializing VQ codebook at epoch {epoch + 1} "
                        f"from up to {args.vq_init_max_vectors} latent vectors..."
                    )
                    init_vectors = _collect_vq_init_vectors(
                        model_to_save=model_to_save,
                        train_loader=train_loader,
                        codec=codec,
                        device=device,
                        max_vectors=args.vq_init_max_vectors,
                    )
                    model_to_save.initialize_codebook(init_vectors, num_iters=args.vq_kmeans_iters)
                _broadcast_quantizer_state(model_to_save, dist_ctx)
                distributed_barrier(dist_ctx)

            model.train()
            train_metric_sums = {
                "total": 0.0,
                "mag": 0.0,
                "phase": 0.0,
                "vq": 0.0,
                "perplexity": 0.0,
                "active": 0.0,
            }
            train_steps = 0
            start_time = time.time()

            if is_main_process(dist_ctx):
                stage_name = "VQAE" if use_vq else "AE warmup"
                print(f"\n--- Epoch [{epoch + 1}/{args.epochs}] Started | stage={stage_name}, vq_beta={current_vq_beta:.4f} ---")

            for step, (batch_tris, lengths) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)

                step_outputs = _run_model_step(
                    model=model,
                    codec=codec,
                    batch_tris=batch_tris,
                    lengths=lengths,
                    device=device,
                    precision=args.precision,
                    use_vq=use_vq,
                    freq_span_map=freq_span_map,
                    valid_mask=valid_mask,
                    args=args,
                    current_vq_beta=current_vq_beta,
                    epoch=epoch,
                    step=step,
                    stage="train_loss",
                )
                loss = step_outputs["loss_total"]

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                _accumulate_epoch_metrics(train_metric_sums, step_outputs)
                train_steps += 1

                if is_main_process(dist_ctx) and step % args.log_interval == 0:
                    print(
                        f"  -> Step [{step}/{len(train_loader)}], "
                        f"Train Loss: {loss.item():.4f}, VQ: {step_outputs['weighted_vq'].item():.4f}, "
                        f"Perplexity: {step_outputs['outputs'].perplexity.item():.2f}, "
                        f"ActiveCodes: {int(step_outputs['outputs'].active_codes.item())}"
                    )

                if max_train_steps > 0 and train_steps >= max_train_steps:
                    break

            train_metrics = _reduce_epoch_metrics(
                train_metric_sums,
                train_steps,
                device=device,
                dist_ctx=dist_ctx,
            )

            should_eval = _is_eval_epoch(epoch, total_epochs=args.epochs, eval_every=args.eval_every)
            val_metrics = None

            if should_eval:
                model.eval()
                val_metric_sums = {
                    "total": 0.0,
                    "mag": 0.0,
                    "phase": 0.0,
                    "vq": 0.0,
                    "perplexity": 0.0,
                    "active": 0.0,
                }
                val_steps = 0

                with torch.no_grad():
                    for _, (val_batch_tris, val_lengths) in enumerate(val_loader):
                        step_outputs = _run_model_step(
                            model=model,
                            codec=codec,
                            batch_tris=val_batch_tris,
                            lengths=val_lengths,
                            device=device,
                            precision=args.precision,
                            use_vq=use_vq,
                            freq_span_map=freq_span_map,
                            valid_mask=valid_mask,
                            args=args,
                            current_vq_beta=current_vq_beta,
                            epoch=epoch,
                            step=val_steps,
                            stage="val_loss",
                        )
                        _accumulate_epoch_metrics(val_metric_sums, step_outputs)
                        val_steps += 1

                        if max_val_steps > 0 and val_steps >= max_val_steps:
                            break

                val_metrics = _reduce_epoch_metrics(
                    val_metric_sums,
                    val_steps,
                    device=device,
                    dist_ctx=dist_ctx,
                )

            current_lr = optimizer.param_groups[0]["lr"]

            if is_main_process(dist_ctx):
                print(f"Epoch {epoch + 1} Completed in {time.time() - start_time:.2f}s | LR: {current_lr:.2e}")
                print(_format_epoch_metrics("Train", train_metrics))
                if should_eval and val_metrics is not None:
                    print(_format_epoch_metrics("Val", val_metrics))

            if should_eval and is_main_process(dist_ctx) and val_metrics is not None:
                _save_fixed_visualization(
                    model=model,
                    codec=codec,
                    fixed_batch_tris=fixed_batch_tris,
                    fixed_lengths=fixed_lengths,
                    spatial_gt=spatial_gt,
                    spatial_icft_orig=spatial_icft_orig,
                    device=device,
                    precision=args.precision,
                    use_vq=use_vq,
                    epoch=epoch,
                    save_dir=viz_dir,
                )

                current_val_loss = float(val_metrics["total"].item())
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    save_checkpoint(best_dir / "vqae_best.pth", model_to_save.state_dict(), precision=args.checkpoint_dtype)
                    save_checkpoint(best_dir / "encoder.pth", model_to_save.encoder.state_dict(), precision=args.checkpoint_dtype)
                    save_training_state(
                        best_dir / "decoder.pth",
                        {
                            "post_vq_proj": model_to_save.post_vq_proj.state_dict(),
                            "decoder": model_to_save.decoder.state_dict(),
                        },
                    )
                    save_training_state(best_dir / "quantizer.pth", model_to_save.quantizer.state_dict())
                    print(f"  [Best] Updated best checkpoint at epoch {epoch + 1} | val={best_val_loss:.4f}")

                train_state = {
                    "checkpoint_version": 1,
                    "completed_epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "model_state": model_to_save.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "rng_state": capture_rng_state(),
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "run_config": dict(vars(args)),
                    "model_config": model_config,
                    "run_timestamp": run_timestamp,
                    "run_dir": str(run_dir),
                    "train_precision": args.precision,
                    "pt_files": [str(path) for path in pt_files],
                    "manifest_total_samples": manifest.total_samples,
                }
                save_latest_training_state_pair(ckpt_dir=ckpt_dir, state=train_state)
                print(f"  [Ckpt] Updated latest resume checkpoint at epoch {epoch + 1}")

            distributed_barrier(dist_ctx)
            scheduler.step()
    except KeyboardInterrupt:
        if is_main_process(dist_ctx):
            print("[WARN] Training interrupted by user, starting cleanup...")
        raise
    finally:
        cleanup_distributed(dist_ctx)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for training engine."""
    parser = argparse.ArgumentParser(description="VQAE pretraining trainer")

    parser.add_argument("--train_type", type=str, default="vqae")
    parser.add_argument("--geom_type", type=str, default="polygon")

    parser.add_argument("--data_dir", type=str, default="./data/processed/hangzhou")
    parser.add_argument("--data_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--load_mode", type=str, default="eager", choices=("eager", "lazy"))
    parser.add_argument("--save_dir", type=str, default="./outputs/ckpt")
    parser.add_argument("--resume_dir", type=str, default=None)

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--weight_mag", type=float, default=1.0)
    parser.add_argument("--weight_mag_hf", type=float, default=1.0)
    parser.add_argument("--weight_phase", type=float, default=1.0)

    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--vq_warmup_epochs", type=int, default=10)
    parser.add_argument("--vq_beta_warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--pos_freqs", type=int, default=63)
    parser.add_argument("--w_min", type=float, default=0.1)
    parser.add_argument("--w_max", type=float, default=200.0)
    parser.add_argument("--freq_type", type=str, default="geometric")
    parser.add_argument("--cft_triangle_chunk_size", type=int, default=2048)
    parser.add_argument("--icft_spatial_chunk_size", type=int, default=256)

    parser.add_argument("--stem_channels", type=str, default="64,128,256")
    parser.add_argument("--stem_strides", type=str, default="2,2,2")
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--drop_rate", type=float, default=0.0)

    parser.add_argument("--decoder_stage_channels", type=str, default="256,192,128")
    parser.add_argument("--decoder_attention_type", type=str, default="window", choices=("none", "window", "global"))
    parser.add_argument("--decoder_attention_heads", type=str, default="8,4,4")
    parser.add_argument("--decoder_attention_depths", type=str, default="1,1,0")
    parser.add_argument("--decoder_conv_depths", type=str, default="2,2,2")
    parser.add_argument("--decoder_window_size", type=int, default=8)
    parser.add_argument("--decoder_upsample_mode", type=str, default="bilinear", choices=("nearest", "bilinear", "bicubic"))
    parser.add_argument("--decoder_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--decoder_drop_rate", type=float, default=0.0)

    parser.add_argument("--codebook_size", type=int, default=8192)
    parser.add_argument("--code_dim", type=int, default=128)
    parser.add_argument("--vq_beta", type=float, default=0.25)
    parser.add_argument("--vq_decay", type=float, default=0.99)
    parser.add_argument("--vq_eps", type=float, default=1.0e-5)
    parser.add_argument("--vq_dead_code_threshold", type=float, default=1.0)
    parser.add_argument("--vq_init_max_vectors", type=int, default=100000)
    parser.add_argument("--vq_kmeans_iters", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--augment_times", type=int, default=10)

    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--checkpoint_dtype", type=str, default="bf16")

    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_val_steps", type=int, default=0)

    return parser


def run_cli(argv=None) -> None:
    """CLI wrapper for trainer.

    Args:
        argv: Optional argv list for programmatic invocation.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Keep compatibility with existing usage where users pass visible devices.
    if "LOCAL_RANK" not in os.environ:
        # Only set CUDA_VISIBLE_DEVICES for non-torchrun single-process launches.
        # torchrun already controls process-local device visibility.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    train_main(args)
