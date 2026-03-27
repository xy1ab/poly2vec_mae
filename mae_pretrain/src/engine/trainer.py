"""Training engine for polygon MAE pretraining.

This module implements:
1) Data loading and train/val split.
2) MAE training loop with fp32/bf16/fp16 precision support.
3) DDP-aware metric reduction.
4) Checkpoint/export generation.
5) Reconstruction visualization for qualitative monitoring.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.collate import mae_collate_fn
from datasets.polygon_dataset import PolyMAEDataset
from datasets.registry import get_geometry_codec
from losses.recon_mag_phase import compute_mag_phase_losses
from models.mae import MaskedAutoencoderViTPoly
from utils.checkpoint import export_model_bundle, save_checkpoint
from utils.dist import DistContext, all_reduce_mean, cleanup_distributed, init_distributed, is_main_process
from utils.filesystem import make_timestamped_dir
from utils.logger import attach_tee_stdout
from utils.precision import autocast_context, build_grad_scaler, normalize_precision
from utils.safe_load import register_numpy_safe_globals
from utils.seed import set_global_seed


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert image tensor into flattened patch tokens.

    Args:
        imgs: Input tensor `[B,C,H,W]`.
        patch_size: Patch edge length.

    Returns:
        Patch tensor `[B,L,C*p*p]`.
    """
    batch, channels, height, width = imgs.shape
    h_patch, w_patch = height // patch_size, width // patch_size
    x = imgs.reshape(shape=(batch, channels, h_patch, patch_size, w_patch, patch_size))
    x = torch.einsum("nchpwq->nhwcpq", x)
    x = x.reshape(shape=(batch, h_patch * w_patch, channels * patch_size**2))
    return x


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


def compute_freq_span_patches(converter, patch_size: int, device: torch.device) -> torch.Tensor:
    """Compute patch-level frequency-span weighting map.

    Args:
        converter: Polygon Fourier converter instance.
        patch_size: Patch edge length.
        device: Runtime device.

    Returns:
        Frequency-span weights shaped `[1,L,p*p]`.
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

    freq_span_map = torch.sqrt(freq_span_map)
    freq_span_map = freq_span_map / (freq_span_map.mean() + 1e-8)

    return patchify(freq_span_map.unsqueeze(0).unsqueeze(0), patch_size)


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
        spatial_icft_recon: ICFT reconstruction from MAE output.
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
        ["Original Mag", "Masked Mag", "Reconstructed Mag"],
        ["Original Cos(Phase)", "Masked Cos(Phase)", "Reconstructed Cos(Phase)"],
        ["Original Sin(Phase)", "Masked Sin(Phase)", "Reconstructed Sin(Phase)"],
    ]
    titles_right = ["Strict Spatial GT (0/1)", "ICFT (Orig Freqs)", "ICFT (MAE Recon Freqs)"]

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
        "img_size": img_size,
        "patch_size": args.patch_size,
        "in_chans": 3,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "dec_embed_dim": args.dec_embed_dim,
        "dec_depth": args.dec_depth,
        "dec_num_heads": args.dec_num_heads,
        "pos_freqs": args.pos_freqs,
        "w_min": args.w_min,
        "w_max": args.w_max,
        "freq_type": args.freq_type,
    }


def _build_model(args, img_size: tuple[int, int], device: torch.device, dist_ctx: DistContext):
    """Construct MAE model and optional DDP wrapper.

    Args:
        args: Parsed training arguments.
        img_size: Runtime image size.
        device: Runtime device.
        dist_ctx: Distributed context.

    Returns:
        Model instance (possibly DDP-wrapped).
    """
    model = MaskedAutoencoderViTPoly(
        img_size=img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.dec_embed_dim,
        decoder_depth=args.dec_depth,
        decoder_num_heads=args.dec_num_heads,
    ).to(device)

    if dist_ctx.enabled:
        if device.type == "cuda":
            model = DDP(model, device_ids=[dist_ctx.local_rank])
        else:
            model = DDP(model)
    return model


def _split_dataset(full_data_list: list, val_ratio: float, split_seed: int) -> tuple[list, list]:
    """Split list dataset into train/validation subsets.

    Args:
        full_data_list: Full sample list.
        val_ratio: Validation ratio.
        split_seed: Random seed for split reproducibility.

    Returns:
        Tuple `(train_data_list, val_data_list)`.
    """
    total_size = len(full_data_list)
    val_size = max(1, int(total_size * val_ratio))
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data_list = [full_data_list[i] for i in train_indices]
    val_data_list = [full_data_list[i] for i in val_indices]
    return train_data_list, val_data_list


def _build_loaders(args, train_data_list: list, val_data_list: list, dist_ctx: DistContext):
    """Build train/validation dataloaders.

    Args:
        args: Parsed training args.
        train_data_list: Training samples.
        val_data_list: Validation samples.
        dist_ctx: Distributed context.

    Returns:
        Tuple `(train_dataset, val_dataset, train_loader, val_loader, train_sampler)`.
    """
    train_dataset = PolyMAEDataset(train_data_list, augment_times=args.augment_times)
    val_dataset = PolyMAEDataset(val_data_list, augment_times=1)

    if dist_ctx.enabled:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=mae_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            collate_fn=mae_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=mae_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=mae_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
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
    fixed_tris_orig = val_dataset.data_list[0]
    random.seed(args.split_seed)
    np.random.seed(args.split_seed)

    # Use the raw validation sample for visualization.
    # This keeps train/val/viz behavior aligned when augment_times=1 and avoids
    # hidden augmentation in reconstruction PNGs.
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

def train_main(args) -> None:
    """Main training entrypoint.

    Args:
        args: Parsed training arguments namespace.
    """
    register_numpy_safe_globals()
    set_global_seed(args.seed, deterministic=args.deterministic)

    dist_ctx = init_distributed()
    device = dist_ctx.device

    args.precision = normalize_precision(args.precision)
    args.checkpoint_dtype = normalize_precision(args.checkpoint_dtype)

    scaler = build_grad_scaler(device=device, precision=args.precision)
    args.viz_every = max(1, int(args.viz_every))

    if is_main_process(dist_ctx):
        run_dir, run_timestamp = make_timestamped_dir(args.save_dir)
        attach_tee_stdout(run_dir / "train_log.txt")
        print("\n[INFO] ========================================================")
        print(f"[INFO] Save directory : {run_dir}")
        print(f"[INFO] Precision      : train={args.precision}, ckpt={args.checkpoint_dtype}")
        print(
            "[INFO] Distributed    : "
            f"enabled={dist_ctx.enabled}, world_size={dist_ctx.world_size}, "
            f"rank={dist_ctx.rank}, local_rank={dist_ctx.local_rank}, device={device}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
        print("[INFO] ========================================================\n")
    else:
        run_dir = Path(args.save_dir)
        run_timestamp = "distributed_worker"

    codec = get_geometry_codec(args.geom_type, vars(args), device=str(device))
    converter = codec.converter
    img_size = (converter.U.shape[0], converter.U.shape[1])

    model_config = _build_model_config(args, img_size=img_size)

    if is_main_process(dist_ctx):
        with (run_dir / "poly_mae_config.json").open("w", encoding="utf-8") as fp:
            json.dump(model_config, fp, indent=4, ensure_ascii=False)

    freq_span_patches = compute_freq_span_patches(converter, args.patch_size, device=device)

    model = _build_model(args, img_size=img_size, device=device, dist_ctx=dist_ctx)
    count_parameters(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.min_lr / args.lr,
            total_iters=args.warmup_epochs,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=args.min_lr,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    full_data_list = torch.load(args.data_path, weights_only=False)
    train_data_list, val_data_list = _split_dataset(full_data_list, val_ratio=args.val_ratio, split_seed=args.split_seed)

    train_dataset, val_dataset, train_loader, val_loader, train_sampler = _build_loaders(
        args,
        train_data_list,
        val_data_list,
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

    try:
        for epoch in range(args.epochs):
            if dist_ctx.enabled and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            train_total, train_mag, train_phase = 0.0, 0.0, 0.0
            start_time = time.time()

            if is_main_process(dist_ctx):
                print(f"\n--- Epoch [{epoch + 1}/{args.epochs}] Started ---")

            for step, (batch_tris, lengths) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    mag, phase = codec.cft_batch(batch_tris, lengths)
                    imgs = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)

                with autocast_context(device, args.precision):
                    _, _, _, pred, mask = model(imgs, mask_ratio=args.mask_ratio)

                pred = pred.float()
                mask = mask.float()
                target_patches = patchify(imgs, args.patch_size).float()

                loss_mag, loss_phase = compute_mag_phase_losses(
                    pred=pred,
                    target_patches=target_patches,
                    mask=mask,
                    patch_size=args.patch_size,
                    freq_span_patches=freq_span_patches,
                    weight_mag_hf=args.weight_mag_hf,
                )
                loss = args.weight_mag * loss_mag + args.weight_phase * loss_phase

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                train_total += loss.item()
                train_mag += loss_mag.item()
                train_phase += loss_phase.item()

                if is_main_process(dist_ctx) and step % args.log_interval == 0:
                    print(f"  -> Step [{step}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

            avg_train_loss = torch.tensor(train_total / max(1, len(train_loader)), device=device)
            avg_train_mag = torch.tensor(train_mag / max(1, len(train_loader)), device=device)
            avg_train_phase = torch.tensor(train_phase / max(1, len(train_loader)), device=device)

            model.eval()
            val_total, val_mag, val_phase = 0.0, 0.0, 0.0

            with torch.no_grad():
                for val_batch_tris, val_lengths in val_loader:
                    mag_v, phase_v = codec.cft_batch(val_batch_tris, val_lengths)
                    imgs_v = torch.cat([mag_v, torch.cos(phase_v), torch.sin(phase_v)], dim=1)

                    with autocast_context(device, args.precision):
                        _, _, _, pred_v, mask_v = model(imgs_v, mask_ratio=args.mask_ratio)

                    pred_v = pred_v.float()
                    mask_v = mask_v.float()
                    target_patches_v = patchify(imgs_v, args.patch_size).float()

                    loss_mag_v, loss_phase_v = compute_mag_phase_losses(
                        pred=pred_v,
                        target_patches=target_patches_v,
                        mask=mask_v,
                        patch_size=args.patch_size,
                        freq_span_patches=freq_span_patches,
                        weight_mag_hf=args.weight_mag_hf,
                    )

                    loss_total_v = args.weight_mag * loss_mag_v + args.weight_phase * loss_phase_v
                    val_total += loss_total_v.item()
                    val_mag += loss_mag_v.item()
                    val_phase += loss_phase_v.item()

            avg_val_loss = torch.tensor(val_total / max(1, len(val_loader)), device=device)
            avg_val_mag = torch.tensor(val_mag / max(1, len(val_loader)), device=device)
            avg_val_phase = torch.tensor(val_phase / max(1, len(val_loader)), device=device)

            avg_train_loss = all_reduce_mean(avg_train_loss, dist_ctx)
            avg_train_mag = all_reduce_mean(avg_train_mag, dist_ctx)
            avg_train_phase = all_reduce_mean(avg_train_phase, dist_ctx)
            avg_val_loss = all_reduce_mean(avg_val_loss, dist_ctx)
            avg_val_mag = all_reduce_mean(avg_val_mag, dist_ctx)
            avg_val_phase = all_reduce_mean(avg_val_phase, dist_ctx)

            current_lr = optimizer.param_groups[0]["lr"]

            if is_main_process(dist_ctx):
                print(f"Epoch {epoch + 1} Completed in {time.time() - start_time:.2f}s | LR: {current_lr:.2e}")
                print(f"  [Train] Total: {avg_train_loss.item():.4f} | Mag: {avg_train_mag.item():.4f} | Phase: {avg_train_phase.item():.4f}")
                print(f"  [Val]   Total: {avg_val_loss.item():.4f} | Mag: {avg_val_mag.item():.4f} | Phase: {avg_val_phase.item():.4f}")

                if (epoch + 1) % args.viz_every == 0:
                    with torch.no_grad():
                        mag_fix, phase_fix = codec.cft_batch(fixed_batch_tris, fixed_lengths)
                        imgs_fix = torch.cat([mag_fix, torch.cos(phase_fix), torch.sin(phase_fix)], dim=1)

                        with autocast_context(device, args.precision):
                            _, _, _, pred_fix, mask_fix = model(imgs_fix, mask_ratio=args.mask_ratio)

                        pred_fix = pred_fix.float()
                        mask_fix = mask_fix.float()

                        p = args.patch_size
                        h, w = imgs_fix.shape[2], imgs_fix.shape[3]
                        h_p, w_p = h // p, w // p

                        img_orig = imgs_fix[0].cpu()

                        mask_map = mask_fix[0].cpu().reshape(h_p, w_p, 1, 1).expand(-1, -1, p, p)
                        mask_map = mask_map.permute(0, 2, 1, 3).reshape(h, w)

                        img_masked = img_orig.clone()
                        img_masked[:, mask_map == 1] = torch.nan

                        pred_img = pred_fix[0].cpu().reshape(h_p, w_p, 3, p, p)
                        pred_img = torch.einsum("hwcpq->chpwq", pred_img).reshape(3, h, w)

                        img_recon = img_orig.clone()
                        img_recon[:, mask_map == 1] = pred_img[:, mask_map == 1]

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
                            save_dir=run_dir,
                        )

                if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                    model_to_save = model.module if isinstance(model, DDP) else model
                    save_checkpoint(
                        run_dir / f"mae_ckpt_{epoch + 1}.pth",
                        model_to_save.state_dict(),
                        precision=args.checkpoint_dtype,
                    )
                    save_checkpoint(
                        run_dir / f"poly_encoder_epoch_{epoch + 1}.pth",
                        model_to_save.encoder.state_dict(),
                        precision=args.checkpoint_dtype,
                    )
                    print(f"  [Save] Saved checkpoints at epoch {epoch + 1}")

            scheduler.step()

        if is_main_process(dist_ctx):
            model_to_save = model.module if isinstance(model, DDP) else model
            export_dir = export_model_bundle(
                export_root=args.export_dir,
                run_name=f"{args.train_type}_{run_timestamp}",
                run_config=vars(args),
                full_state_dict=model_to_save.state_dict(),
                encoder_state_dict=model_to_save.encoder.state_dict(),
                checkpoint_precision=args.checkpoint_dtype,
                train_log_path=run_dir / "train_log.txt",
            )
            print(f"[Export] Bundle exported to: {export_dir}")
    except KeyboardInterrupt:
        if is_main_process(dist_ctx):
            print("[WARN] Training interrupted by user, starting cleanup...")
        raise
    finally:
        cleanup_distributed(dist_ctx)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for training engine.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="MAE pretraining trainer")

    parser.add_argument("--train_type", type=str, default="mae")
    parser.add_argument("--geom_type", type=str, default="polygon")

    parser.add_argument("--data_path", type=str, default="./data/processed/polygon_triangles_normalized.pt")
    parser.add_argument("--save_dir", type=str, default="./outputs/ckpt")
    parser.add_argument("--export_dir", type=str, default="./outputs/exports")

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
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--pos_freqs", type=int, default=63)
    parser.add_argument("--w_min", type=float, default=0.1)
    parser.add_argument("--w_max", type=float, default=200.0)
    parser.add_argument("--freq_type", type=str, default="geometric")

    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)

    parser.add_argument("--dec_embed_dim", type=int, default=128)
    parser.add_argument("--dec_depth", type=int, default=4)
    parser.add_argument("--dec_num_heads", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--augment_times", type=int, default=10)

    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--checkpoint_dtype", type=str, default="bf16")

    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--viz_every", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=50)

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
