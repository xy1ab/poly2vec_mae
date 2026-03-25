import os
import sys
import subprocess
import argparse
import time
import random
import json
import shutil
from datetime import datetime
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import yaml

# ---------------------------------------------------------
# 解决 PyTorch 2.6+ 默认 weights_only=True 导致加载 numpy 数据集报错的问题
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        torch.serialization.add_safe_globals([np.ndarray])
        if hasattr(np, '_core'):
            import numpy._core.multiarray
            torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct, numpy._core.multiarray.scalar])
        elif hasattr(np, 'core'):
            import numpy.core.multiarray
            torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct, numpy.core.multiarray.scalar])
        torch.serialization.add_safe_globals([np.dtype])
        if hasattr(np, 'dtypes'):
            for dtype_class in ['Float32DType', 'Float64DType', 'Int32DType', 'Int64DType']:
                if hasattr(np.dtypes, dtype_class):
                    torch.serialization.add_safe_globals([getattr(np.dtypes, dtype_class)])
        torch.serialization.add_safe_globals([type(np.dtype(np.float32)), type(np.dtype(np.float64)), np.float32, np.float64])
    except Exception:
        pass
# ---------------------------------------------------------

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.data.dataloader_mae import PolyMAEDataset, mae_collate_fn
from utils.fourier.engine import PolyFourierConverter
from utils.io.precision import (
    autocast_context,
    normalize_precision,
    precision_to_torch_dtype,
)
from mae_core.model import MaskedAutoencoderViTPoly

def patchify(imgs, p):
    B, C, H, W = imgs.shape
    h, w = H // p, W // p
    x = imgs.reshape(shape=(B, C, h, p, w, p))
    x = torch.einsum('nchpwq->nhwcpq', x)
    x = x.reshape(shape=(B, h * w, C * p**2))
    return x

def rasterize_tris_to_grid(tris, H, W):
    x = np.linspace(-1, 1, W)
    y = np.linspace(1, -1, H) # Y 轴向下
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.flatten(), Y.flatten())).T
    
    mask = np.zeros(H * W, dtype=bool)
    for tri in tris.numpy():
        p = Path(tri)
        mask = mask | p.contains_points(points)
        
    return mask.reshape(H, W).astype(np.float32)

def plot_reconstruction(img_orig, img_masked, img_recon, spatial_gt, spatial_icft_orig, spatial_icft_recon, epoch, save_dir):
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(3, 6)
    
    vmin_mag, vmax_mag = img_orig[0].min(), img_orig[0].max()
    vmin_trig, vmax_trig = -1.0, 1.0 
    
    vmin_sp = 0.0
    vmax_sp = max(1.0, spatial_icft_orig.max().item(), spatial_icft_recon.max().item())
    
    axes_left = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])], # Mag
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])], # Cos
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]  # Sin
    ]
    axes_right = [
        fig.add_subplot(gs[:, 3]), 
        fig.add_subplot(gs[:, 4]), 
        fig.add_subplot(gs[:, 5])
    ]
    
    titles_left = [
        ['Original Mag', 'Masked Mag', 'Reconstructed Mag'],
        ['Original Cos(Phase)', 'Masked Cos(Phase)', 'Reconstructed Cos(Phase)'],
        ['Original Sin(Phase)', 'Masked Sin(Phase)', 'Reconstructed Sin(Phase)']
    ]
    titles_right = ['Strict Spatial GT (0/1)', 'ICFT (Orig Freqs)', 'ICFT (MAE Recon Freqs)']
    
    data_left = [
        [(img_orig[0], vmin_mag, vmax_mag), (img_masked[0], vmin_mag, vmax_mag), (img_recon[0], vmin_mag, vmax_mag)],
        [(img_orig[1], vmin_trig, vmax_trig), (img_masked[1], vmin_trig, vmax_trig), (img_recon[1], vmin_trig, vmax_trig)],
        [(img_orig[2], vmin_trig, vmax_trig), (img_masked[2], vmin_trig, vmax_trig), (img_recon[2], vmin_trig, vmax_trig)]
    ]
    data_right = [
        (spatial_gt, vmin_sp, vmax_sp), 
        (spatial_icft_orig, vmin_sp, vmax_sp), 
        (spatial_icft_recon, vmin_sp, vmax_sp)
    ]
    
    for row in range(3):
        for col in range(3):
            ax = axes_left[row][col]
            data, vmin, vmax = data_left[row][col]
            im = ax.imshow(data.numpy() if isinstance(data, torch.Tensor) else data, 
                           cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(titles_left[row][col], fontsize=14)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    for col in range(3):
        ax = axes_right[col]
        data, vmin, vmax = data_right[col]
        im = ax.imshow(data.numpy() if isinstance(data, torch.Tensor) else data, 
                       cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(titles_right[col], fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'recon_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _cast_state_dict_dtype(state_dict, dtype):
    converted = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=dtype)
            converted[key] = tensor
        else:
            converted[key] = value
    return converted


def export_final_artifacts(args, model_to_save, experiment_dir, run_timestamp):
    export_base = os.path.join(args.export_dir, f"{args.train_type}_{run_timestamp}")
    os.makedirs(export_base, exist_ok=True)

    # 1) 保存运行参数（可复现实验）
    config_path = os.path.join(export_base, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, allow_unicode=True, sort_keys=False)

    # 2) 保存最终完整模型和纯编码器（可选降精度以减小体积）
    save_dtype = precision_to_torch_dtype(args.checkpoint_dtype)
    full_state_dict = _cast_state_dict_dtype(model_to_save.state_dict(), save_dtype)
    encoder_state_dict = _cast_state_dict_dtype(model_to_save.encoder.state_dict(), save_dtype)
    torch.save(full_state_dict, os.path.join(export_base, "encoder_decoder.pth"))
    torch.save(encoder_state_dict, os.path.join(export_base, "encoder.pth"))

    # 3) 复制训练日志
    src_log = os.path.join(experiment_dir, "train_log.txt")
    if os.path.exists(src_log):
        shutil.copy2(src_log, os.path.join(export_base, "train_log.txt"))

    print(f"[Export] 最终交付产物已导出到: {export_base}")

def main(args):
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if local_rank == 0:
            print(f"启动 DDP 多卡模式，World Size: {torch.distributed.get_world_size()}")
    else:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"启动单卡模式，使用设备: {device}")

    train_precision = normalize_precision(args.precision)
    checkpoint_precision = normalize_precision(args.checkpoint_dtype)
    if device.type != "cuda" and train_precision in ("fp16", "bf16"):
        if local_rank == 0:
            print(f"[Precision] 当前设备为 {device}，自动将训练精度从 {train_precision} 回退为 fp32。")
        train_precision = "fp32"
    if device.type == "cuda" and train_precision == "bf16" and not torch.cuda.is_bf16_supported():
        if local_rank == 0:
            print("[Precision] 当前 CUDA 设备不支持 bf16，自动回退到 fp16。")
        train_precision = "fp16"

    args.precision = train_precision
    args.checkpoint_dtype = checkpoint_precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and train_precision == "fp16"))

    # =========================================================================
    # 动态创建包含时间戳的专属保存文件夹，并配置自动双写日志记录 Logger
    # =========================================================================
    run_timestamp = None
    if local_rank == 0:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_dir = os.path.join(args.save_dir, run_timestamp)
        os.makedirs(experiment_dir, exist_ok=True)
        
        class Logger(object):
            def __init__(self, filename="Default.log"):
                self.terminal = sys.stdout
                self.log = open(filename, "a", encoding="utf-8")

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()

            def flush(self):
                self.terminal.flush()
                self.log.flush()

        sys.stdout = Logger(os.path.join(experiment_dir, 'train_log.txt'))
        
        print(f"\n[INFO] ========================================================")
        print(f"[INFO] 本次训练的所有输出结果将保存在独立文件夹: {experiment_dir}")
        print(f"[INFO] 训练日志已同步实时写入: train_log.txt")
        print(f"[INFO] 混合精度训练: {args.precision} | 导出权重精度: {args.checkpoint_dtype}")
        print(f"[INFO] ========================================================\n")
    else:
        experiment_dir = args.save_dir

    fourier_engine = PolyFourierConverter(
        pos_freqs=args.pos_freqs, w_min=args.w_min, w_max=args.w_max, 
        freq_type=args.freq_type, device=device, patch_size=args.patch_size
    )
    
    img_size = (fourier_engine.U.shape[0], fourier_engine.U.shape[1])
    
    # =========================================================================
    # 保存参数配置 config.json 供下游任务一键加载使用
    # =========================================================================
    if local_rank == 0:
        mae_config = {
            "geom_type": "polygon",  
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
            "freq_type": args.freq_type
        }
        with open(os.path.join(experiment_dir, 'poly_mae_config.json'), 'w', encoding='utf-8') as f:
            json.dump(mae_config, f, indent=4)
        print(f"[INFO] 已保存完整MAE模型与引擎配置至: {os.path.join(experiment_dir, 'poly_mae_config.json')}")
    
    # === 直接计算基于频率跨度面积 (du * dv) 的物理积分权重 ===
    H, W = fourier_engine.U.shape
    pad_h, pad_w = fourier_engine.pad_h, fourier_engine.pad_w
    valid_h = H - pad_h
    valid_w = W - pad_w
    Wx = fourier_engine.U[:valid_h, 0].clone()
    Wy = fourier_engine.V[0, :valid_w].clone()
    
    du = torch.zeros_like(Wx)
    for i in range(valid_h):
        if i == 0: du[i] = Wx[1] - Wx[0] if valid_h > 1 else 1.0
        elif i == valid_h - 1: du[i] = Wx[i] - Wx[i-1]
        else: du[i] = (Wx[i+1] - Wx[i-1]) / 2.0
        
    dv = torch.zeros_like(Wy)
    for j in range(valid_w):
        if j == 0: dv[j] = Wy[1] - Wy[0] if valid_w > 1 else 1.0
        elif j == valid_w - 1: dv[j] = Wy[j] - Wy[j-1]
        else: dv[j] = (Wy[j+1] - Wy[j-1]) / 2.0
        
    dU, dV = torch.meshgrid(du, dv, indexing='ij')
    freq_span_map = torch.zeros((H, W), device=device)
    freq_span_map[:valid_h, :valid_w] = dU * dV
    
    # 缓解极端高频跨度造成的惩罚爆炸，做平方根平滑
    freq_span_map = torch.sqrt(freq_span_map)
    # 归一化，保持整体Loss平均水平不变，防止破坏原有的学习率
    freq_span_map = freq_span_map / (freq_span_map.mean() + 1e-8)
    
    # 展平为 Patch 维度 [1, L, p*p]，等待广播计算
    freq_span_patches = patchify(freq_span_map.unsqueeze(0).unsqueeze(0), args.patch_size)
    save_dtype = precision_to_torch_dtype(args.checkpoint_dtype)
    
    model = MaskedAutoencoderViTPoly(
        img_size=img_size, patch_size=args.patch_size, in_chans=3,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        decoder_embed_dim=args.dec_embed_dim, decoder_depth=args.dec_depth, decoder_num_heads=args.dec_num_heads
    ).to(device)
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=args.min_lr / args.lr, total_iters=args.warmup_epochs)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.min_lr)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    full_data_list = torch.load(args.data_path, weights_only=False)
    total_size = len(full_data_list)
    val_size = max(1, int(total_size * args.val_ratio))
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_data_list = [full_data_list[i] for i in train_indices]
    val_data_list = [full_data_list[i] for i in val_indices]
    
    train_dataset = PolyMAEDataset(train_data_list, augment_times=args.augment_times)
    val_dataset = PolyMAEDataset(val_data_list, augment_times=1)
    
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=mae_collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=mae_collate_fn, num_workers=4, pin_memory=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=mae_collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=mae_collate_fn, num_workers=4, pin_memory=True)

    spatial_gt, spatial_icft_orig = None, None
    fixed_batch_tris = None
    fixed_lengths = None
    
    if local_rank == 0:
        fixed_tris_orig = val_dataset.data_list[0]
        random.seed(42)
        np.random.seed(42)
        fixed_tris_np = val_dataset.apply_augmentation(fixed_tris_orig)
        fixed_tris = torch.tensor(fixed_tris_np, dtype=torch.float32)
        
        fixed_batch_tris = fixed_tris.unsqueeze(0).to(device) 
        fixed_lengths = torch.tensor([fixed_tris.shape[0]], device=device)
        
        with torch.no_grad():
            mag_val_fix, phase_val_fix = fourier_engine.cft_polygon_batch(fixed_batch_tris, fixed_lengths)
            raw_mag_orig = torch.expm1(mag_val_fix)
            F_uv_orig = (raw_mag_orig * torch.exp(1j * phase_val_fix)).squeeze(1)
            spatial_icft_orig = fourier_engine.icft_2d(F_uv_orig)[0].detach().cpu()
            H_sp, W_sp = spatial_icft_orig.shape[-2], spatial_icft_orig.shape[-1]
            spatial_gt = rasterize_tris_to_grid(fixed_tris.cpu(), H_sp, W_sp)

    for epoch in range(args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
            
        model.train()
        train_total_loss, train_mag_loss, train_phase_loss = 0, 0, 0
        start_time = time.time()
        
        if local_rank == 0: print(f"\n--- Epoch [{epoch+1}/{args.epochs}] Started ---")
            
        for step, (batch_tris, lengths) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                mag, phase = fourier_engine.cft_polygon_batch(batch_tris, lengths)
                cos_phase = torch.cos(phase)
                sin_phase = torch.sin(phase)
                imgs = torch.cat([mag, cos_phase, sin_phase], dim=1)
            
            with autocast_context(device, train_precision):
                _, _, _, pred, mask = model(imgs, mask_ratio=args.mask_ratio)
            
            p = args.patch_size
            target_patches = patchify(imgs, p).float()
            pred = pred.float()
            mask = mask.float()
            
            target_mag_patches = target_patches[:, :, :p**2]
            target_cos_patches = target_patches[:, :, p**2:2*p**2]
            target_sin_patches = target_patches[:, :, 2*p**2:]
            
            pred_mag_patches = pred[:, :, :p**2]
            pred_cos_patches = pred[:, :, p**2:2*p**2]
            pred_sin_patches = pred[:, :, 2*p**2:]
            
            # --- 【非对称双轨 Loss 架构 (L1 Loss)】 ---
            mag_l1 = torch.abs(pred_mag_patches - target_mag_patches)
            loss_mag_base = (mag_l1.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-8)
            weighted_mag_loss = mag_l1 * freq_span_patches
            loss_mag_penalty = (weighted_mag_loss.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-8)
            loss_mag = loss_mag_base + args.weight_mag_hf * loss_mag_penalty
            
            phase_l1 = torch.abs(pred_cos_patches - target_cos_patches) + torch.abs(pred_sin_patches - target_sin_patches)
            loss_phase = (phase_l1.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-8)
            
            loss = args.weight_mag * loss_mag + args.weight_phase * loss_phase
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_total_loss += loss.item()
            train_mag_loss += loss_mag.item()
            train_phase_loss += loss_phase.item()
            
            if local_rank == 0 and step % 50 == 0:
                print(f"  -> Step [{step}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

        avg_train_loss = torch.tensor(train_total_loss / len(train_loader), device=device)
        avg_train_mag = torch.tensor(train_mag_loss / len(train_loader), device=device)
        avg_train_ph = torch.tensor(train_phase_loss / len(train_loader), device=device)
        
        model.eval()
        val_total_loss, val_mag_loss, val_phase_loss = 0, 0, 0
        with torch.no_grad():
            for val_batch_tris, val_lengths in val_loader:
                mag_v, phase_v = fourier_engine.cft_polygon_batch(val_batch_tris, val_lengths)
                cos_phase_v = torch.cos(phase_v)
                sin_phase_v = torch.sin(phase_v)
                imgs_v = torch.cat([mag_v, cos_phase_v, sin_phase_v], dim=1)
                
                with autocast_context(device, train_precision):
                    _, _, _, pred_v, mask_v = model(imgs_v, mask_ratio=args.mask_ratio)
                
                target_patches_v = patchify(imgs_v, args.patch_size).float()
                pred_v = pred_v.float()
                mask_v = mask_v.float()
                target_mag_patches_v = target_patches_v[:, :, :args.patch_size**2]
                target_cos_patches_v = target_patches_v[:, :, args.patch_size**2:2*args.patch_size**2]
                target_sin_patches_v = target_patches_v[:, :, 2*args.patch_size**2:]
                
                pred_mag_patches_v = pred_v[:, :, :args.patch_size**2]
                pred_cos_patches_v = pred_v[:, :, args.patch_size**2:2*args.patch_size**2]
                pred_sin_patches_v = pred_v[:, :, 2*args.patch_size**2:]
                
                mag_l1_v = torch.abs(pred_mag_patches_v - target_mag_patches_v)
                l_mag_base_v = (mag_l1_v.mean(dim=-1) * mask_v).sum() / (mask_v.sum() + 1e-8)
                l_mag_penalty_v = ((mag_l1_v * freq_span_patches).mean(dim=-1) * mask_v).sum() / (mask_v.sum() + 1e-8)
                l_mag_v = l_mag_base_v + args.weight_mag_hf * l_mag_penalty_v
                
                phase_l1_v = torch.abs(pred_cos_patches_v - target_cos_patches_v) + torch.abs(pred_sin_patches_v - target_sin_patches_v)
                l_phase_v = (phase_l1_v.mean(dim=-1) * mask_v).sum() / (mask_v.sum() + 1e-8)
                
                l_total_v = args.weight_mag * l_mag_v + args.weight_phase * l_phase_v
                val_total_loss += l_total_v.item()
                val_mag_loss += l_mag_v.item()
                val_phase_loss += l_phase_v.item()
                
        avg_val_loss = torch.tensor(val_total_loss / len(val_loader), device=device)
        avg_val_mag = torch.tensor(val_mag_loss / len(val_loader), device=device)
        avg_val_ph = torch.tensor(val_phase_loss / len(val_loader), device=device)

        if is_ddp:
            torch.distributed.all_reduce(avg_train_loss); torch.distributed.all_reduce(avg_val_loss)
            torch.distributed.all_reduce(avg_train_mag); torch.distributed.all_reduce(avg_val_mag)
            torch.distributed.all_reduce(avg_train_ph); torch.distributed.all_reduce(avg_val_ph)
            world_size = torch.distributed.get_world_size()
            avg_train_loss /= world_size; avg_val_loss /= world_size
            avg_train_mag /= world_size; avg_val_mag /= world_size
            avg_train_ph /= world_size; avg_val_ph /= world_size
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if local_rank == 0:
            print(f"Epoch {epoch+1} Completed in {time.time()-start_time:.2f}s | Current LR: {current_lr:.2e}")
            print(f"  [Train] Total: {avg_train_loss.item():.4f} | Mag(Dual): {avg_train_mag.item():.4f} | Phase(L1): {avg_train_ph.item():.4f}")
            print(f"  [Val]   Total: {avg_val_loss.item():.4f} | Mag(Dual): {avg_val_mag.item():.4f} | Phase(L1): {avg_val_ph.item():.4f}")
            
            with torch.no_grad():
                mag_val_fix, phase_val_fix = fourier_engine.cft_polygon_batch(fixed_batch_tris, fixed_lengths)
                cos_val_fix = torch.cos(phase_val_fix)
                sin_val_fix = torch.sin(phase_val_fix)
                imgs_val_fix = torch.cat([mag_val_fix, cos_val_fix, sin_val_fix], dim=1)
                
                with autocast_context(device, train_precision):
                    _, _, _, pred_val_fix, mask_val_fix = model(imgs_val_fix, mask_ratio=args.mask_ratio)
                pred_val_fix = pred_val_fix.float()
                mask_val_fix = mask_val_fix.float()
                
                p = args.patch_size
                h, w = mag_val_fix.shape[2], mag_val_fix.shape[3]
                h_p, w_p = h // p, w // p
                
                img_orig = imgs_val_fix[0].cpu() # [3, H, W]
                
                mask_map = mask_val_fix[0].cpu().reshape(h_p, w_p, 1, 1).expand(-1, -1, p, p)
                mask_map = mask_map.permute(0, 2, 1, 3).reshape(h, w)
                
                img_masked = img_orig.clone()
                img_masked[:, mask_map == 1] = torch.nan
                
                pred_img = pred_val_fix[0].cpu().reshape(h_p, w_p, 3, p, p)
                pred_img = torch.einsum('hwcpq->chpwq', pred_img).reshape(3, h, w)
                
                img_recon = img_orig.clone()
                img_recon[:, mask_map == 1] = pred_img[:, mask_map == 1]
                
                mag_recon_tensor = img_recon[0].unsqueeze(0).to(device)
                cos_recon_tensor = img_recon[1].unsqueeze(0).to(device)
                sin_recon_tensor = img_recon[2].unsqueeze(0).to(device)
                
                phase_recon_tensor = torch.atan2(sin_recon_tensor, cos_recon_tensor)
                
                raw_mag_recon = torch.expm1(mag_recon_tensor)
                F_uv_recon = raw_mag_recon * torch.exp(1j * phase_recon_tensor)
                
                spatial_icft_recon = fourier_engine.icft_2d(F_uv_recon)[0].squeeze().cpu()
                
                plot_reconstruction(img_orig, img_masked, img_recon, 
                                    spatial_gt, spatial_icft_orig.squeeze(), spatial_icft_recon, 
                                    epoch+1, experiment_dir)
            
            # =========================================================================
            # [修改点]: 每 20 个 Epoch 或 到达最后一个 Epoch 时强制保存！
            # =========================================================================
            if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs:
                model_to_save = model.module if is_ddp else model
                full_state_dict = _cast_state_dict_dtype(model_to_save.state_dict(), save_dtype)
                encoder_state_dict = _cast_state_dict_dtype(model_to_save.encoder.state_dict(), save_dtype)
                torch.save(full_state_dict, os.path.join(experiment_dir, f'mae_ckpt_{epoch+1}.pth'))
                torch.save(encoder_state_dict, os.path.join(experiment_dir, f'poly_encoder_epoch_{epoch+1}.pth'))
                print(f"  [Save] Checkpoints saved to {experiment_dir} (mae_ckpt & poly_encoder) at Epoch {epoch+1}")

        scheduler.step()

    if local_rank == 0:
        model_to_export = model.module if is_ddp else model
        sys.stdout.flush()
        export_final_artifacts(args, model_to_export, experiment_dir, run_timestamp)

    if is_ddp:
        torch.distributed.destroy_process_group()

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Poly2Vec MAE Training")
    parser.add_argument('--data_path', default='./data/processed/polygon_triangles_normalized.pt', type=str)
    parser.add_argument('--save_dir', default='./outputs/checkpoints/', type=str)
    
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--weight_mag', default=1.0, type=float)
    parser.add_argument('--weight_mag_hf', default=1.0, type=float, help='幅值的高频跨度惩罚权重 (alpha)')
    parser.add_argument('--weight_phase', default=1.0, type=float)
    
    parser.add_argument('--val_ratio', default=0.05, type=float)
    parser.add_argument('--warmup_epochs', default=15, type=int)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    
    parser.add_argument('--pos_freqs', default=31, type=int)
    parser.add_argument('--w_min', default=0.1, type=float)
    parser.add_argument('--w_max', default=100.0, type=float)
    parser.add_argument('--freq_type', default='geometric', type=str)
    
    parser.add_argument('--patch_size', default=2, type=int)
    parser.add_argument('--embed_dim', default=384, type=int)
    
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    
    parser.add_argument('--dec_embed_dim', default=128, type=int)
    parser.add_argument('--dec_depth', default=4, type=int)
    parser.add_argument('--dec_num_heads', default=4, type=int)
    
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=2.0e-3, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--augment_times', default=10, type=int)
    parser.add_argument('--precision', default='bf16', type=str, help='训练精度: fp32 | fp16 | bf16')
    parser.add_argument('--checkpoint_dtype', default='bf16', type=str, help='保存权重精度: fp32 | fp16 | bf16')
    parser.add_argument('--train_type', default='mae', type=str)
    parser.add_argument('--export_dir', default='./outputs/exports/', type=str)

    return parser

def run_cli(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpus = str(args.gpu).split(',')
    
    if len(gpus) > 1 and "LOCAL_RANK" not in os.environ:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(len(gpus)), sys.argv[0]]
        cmd.extend(sys.argv[1:])
        subprocess.run(cmd, start_new_session=True)
        sys.exit(0)
        
    main(args)

if __name__ == '__main__':
    run_cli()
