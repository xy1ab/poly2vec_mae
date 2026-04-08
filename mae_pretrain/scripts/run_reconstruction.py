#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_reconstruction.py
@Time    :   2026/04/07 14:34:03
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
import torch
import argparse
from pathlib import Path

from tqdm import tqdm
from datetime import datetime
import sys
import re
from typing import Any

import numpy as np
import matplotlib.path as mpltPath

import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)
from mae_pretrain.src.datasets.geometry_polygon import pad_triangle_batch
from mae_pretrain.src.datasets.registry import get_geometry_codec
from mae_pretrain.src.datasets.shard_io import load_triangle_shard, resolve_triangle_shard_paths
from mae_pretrain.src.datasets.sharded_pt_dataset import _ensure_numpy_float32
from mae_pretrain.src.models.factory import load_mae_model, load_pretrained_encoder
from mae_pretrain.src.models.decoder import TransUNetdecoder
from mae_pretrain.src.utils.config import load_config_any
from mae_pretrain.src.utils.filesystem import ensure_dir
from mae_pretrain.src.utils.precision import autocast_context, normalize_precision


def rasterize_triangles_pytorch(batch_tris, spatial_size=256):
    """
    针对 Batch 优化并排除退化三角形的栅格化函数
    """
    B, N, _, _ = batch_tris.shape
    device = batch_tris.device
    H = W = spatial_size

    # 生成网格坐标 (B, H, W, 2)
    y = torch.linspace(1, -1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    p = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, H * W, 2) 

    # 准备结果容器 (B, H*W)
    final_mask = torch.zeros((B, H * W), device=device, dtype=torch.bool)

    # 顶点提取 (B, N, 2)
    v0 = batch_tris[:, :, 0, :].unsqueeze(2) # (B, N, 1, 2)
    v1 = batch_tris[:, :, 1, :].unsqueeze(2)
    v2 = batch_tris[:, :, 2, :].unsqueeze(2)

    def edge_func(va, vb, vp):
        # (x2-x1)*(yP-y1) - (y2-y1)*(xP-x1)
        return (vb[..., 0] - va[..., 0]) * (vp[..., 1] - va[..., 1]) - \
               (vb[..., 1] - va[..., 1]) * (vp[..., 0] - va[..., 0])

    # 为了节省显存，可以分批处理 N 个三角形
    chunk_size = 100 
    for i in range(0, N, chunk_size):
        v0_c = v0[:, i:i+chunk_size]
        v1_c = v1[:, i:i+chunk_size]
        v2_c = v2[:, i:i+chunk_size]

        # 1. 计算边函数
        w0 = edge_func(v0_c, v1_c, p)
        w1 = edge_func(v1_c, v2_c, p)
        w2 = edge_func(v2_c, v0_c, p)

        # 2. 排除退化三角形（面积为 0 的三角形）
        # 面积近似于边函数之和。如果三个顶点重合，w 全为 0
        # 增加一个 epsilon 避免浮点误差导致的虚假覆盖
        eps = 1e-6
        is_inside = ((w0 > eps) & (w1 > eps) & (w2 > eps)) | \
                    ((w0 < -eps) & (w1 < -eps) & (w2 < -eps))

        # 3. 实时合并到结果中
        final_mask |= is_inside.any(dim=1)

    return final_mask.view(B, H, W).float()

def _default_device() -> str:
    """Resolve default runtime device string."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _extract_last_int(text: str) -> int | None:
    """Extract the last integer substring from text, if present."""
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])

def _resolve_user_path(path_str: str, project_root: Path) -> Path:
    """Resolve one user-provided path against cwd and project root."""
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (project_root / raw_path).resolve()

def _inject_repo_root() -> Path:
    """Inject repository root into `sys.path` for direct script execution.

    Returns:
        `mae_pretrain` project root path.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root

def _build_output_path(project_root: Path, data_dir: str, output_path: str | None) -> Path:
    """Resolve output path from explicit CLI path or one timestamped default."""
    if output_path:
        return Path(output_path).expanduser().resolve()

    data_name = Path(data_dir).expanduser().resolve().name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "outputs" / "forward_batch" / f"{data_name}_forward_batch_{timestamp}.pt"

def _resolve_model_artifacts(model_dir: str) -> tuple[Path, Path, Path, bool]:
    """Resolve encoder weight, MAE weight, and config under one model directory."""
    base = Path(model_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {base}")

    search_roots = [base]
    for subdir in ("best", "ckpt"):
        candidate = base / subdir
        if candidate.is_dir():
            search_roots.append(candidate)

    config_candidates: list[Path] = []
    for root in search_roots:
        for path in (root / "config.yaml", root / "config.yml", root / "poly_mae_config.json"):
            if path.exists() and path not in config_candidates:
                config_candidates.append(path)

    mae_candidates: list[Path] = []
    for root in search_roots:
        candidates = [
            path
            for path in root.glob("*.pth")
            if path.is_file() and ("encoder_decoder" in path.name.lower() or "mae" in path.name.lower())
        ]
        for path in candidates:
            if path not in mae_candidates:
                mae_candidates.append(path)

    encoder_candidates: list[Path] = []
    for root in search_roots:
        for path in (root / "encoder.pth", root / "encoder_best.pth"):
            if path.exists() and path not in encoder_candidates:
                encoder_candidates.append(path)
        for path in sorted(root.glob("*encoder*.pth")):
            name_lower = path.name.lower()
            if "encoder_decoder" in name_lower:
                continue
            if path not in encoder_candidates:
                encoder_candidates.append(path)

    if not config_candidates:
        raise FileNotFoundError(f"No config file found in model_dir: {base}")


    def _encoder_rank(path: Path) -> tuple[int, int, float, str]:
        name_lower = path.name.lower()
        if name_lower == "encoder.pth":
            priority = 0
        elif name_lower == "encoder_best.pth":
            priority = 1
        else:
            priority = 2
        suffix = _extract_last_int(path.stem)
        suffix_rank = -(suffix if suffix is not None else -1)
        mtime_rank = -float(path.stat().st_mtime)
        return (priority, suffix_rank, mtime_rank, name_lower)

    encoder_candidates = sorted(encoder_candidates, key=_encoder_rank)


    if encoder_candidates:
        return encoder_candidates[0], config_candidates[0], False

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch forward export."""
    parser = argparse.ArgumentParser(
        description="Batch forward triangulated polygon shards into embeddings and MAE frequency maps."
    )
    # parser.add_argument("--model_dir", type=str, required=True, help="Directory containing encoder/MAE checkpoints and config.")
    # parser.add_argument("--data_dir", type=str, required=True, help="Directory containing triangulated shard `.pt` files.")
    parser.add_argument("--model_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/ckpt/", help="Directory containing encoder/MAE checkpoints and config.")
    parser.add_argument("--data_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/data/processed/", help="Directory containing triangulated shard `.pt` files.")
    parser.add_argument("--output_path", type=str, default="", help="Output `.pt` file path.")
    parser.add_argument("--save_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_checkpoints", help="Output `.pt` file path.")
    parser.add_argument("--train_data_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/mae_pretrain/outputs/forward_batch/processed_forward_batch_20260407_174737.pt", help="Output `.pt` file path.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--device", type=str, default=_default_device(), help="Runtime device, e.g. cuda or cpu.")
    parser.add_argument("--precision", type=str, default="fp32", help="Runtime precision: fp32/bf16/fp16.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional export cap; 0 means all samples.")
    parser.add_argument("--epoch", type=int, default=1000, help="Optional export cap; 0 means all samples.")
    return parser

def getdata(args) -> None:
    """CLI main entrypoint."""
    project_root = _inject_repo_root()


    args.precision = normalize_precision(args.precision)
    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.max_samples < 0:
        raise ValueError(f"`max_samples` must be >= 0, got {args.max_samples}")

    args.model_dir = str(_resolve_user_path(args.model_dir, project_root))
    args.data_dir = str(_resolve_user_path(args.data_dir, project_root))
    output_path = _build_output_path(project_root, args.data_dir, args.output_path or None)
    ensure_dir(output_path.parent)

    requested_device = args.device or _default_device()
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, fallback to CPU for forward export.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    shard_paths = resolve_triangle_shard_paths(args.data_dir, warn_fn=lambda message: print(message))
    encoder_weight, config_path, encoder_fallback_to_mae = _resolve_model_artifacts(args.model_dir)

    config = load_config_any(config_path)
    geom_type = str(config.get("geom_type", "polygon")).lower()
    codec = get_geometry_codec(geom_type, config, device=str(device))

    encoder = load_pretrained_encoder(
        weight_path=encoder_weight,
        config_path=config_path,
        device=device,
        precision=args.precision,
    )

    encoder.eval()


    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "encoder_weight": str(Path(encoder_weight).expanduser().resolve()),
        "precision": args.precision,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "max_samples": int(args.max_samples),
        "spatial_size": 256,
        "geom_type": geom_type,
        "config": dict(config),
        "shard_paths": [str(path) for path in shard_paths],
        "processed_shard_count": 0,
        "sample_count": 0,
    }

    output_samples: list[dict[str, Any]] = []
    pending_tris: list[Any] = []
    exported_count = 0
    processed_shard_count = 0

    def flush_pending() -> None:
        """Run one inference batch and append output sample dicts."""
        nonlocal exported_count
        if not pending_tris:
            return

        batch_tris, lengths = pad_triangle_batch(pending_tris, device=device)
        with torch.no_grad():
            mag, phase = codec.cft_batch(batch_tris, lengths)
            imgs = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)
            true_imag = rasterize_triangles_pytorch(batch_tris)
            with autocast_context(device, args.precision):
                encoder_features = encoder(imgs)


            embeddings = encoder_features.float().cpu()


        start_index = exported_count + 1
        for offset, tri_np in enumerate(pending_tris):
            sample_index = start_index + offset
            output_samples.append(
                {
                    # "sample_index": int(sample_index),
                    # "triangles": torch.from_numpy(tri_np.astype("float32", copy=False)),
                    "embedding": embeddings[offset],
                    "imag": true_imag[offset],
                }
            )

        exported_count += len(pending_tris)
        pending_tris.clear()

    for shard_path in tqdm(shard_paths, desc="Reading shards"):
        processed_shard_count += 1
        shard_data = load_triangle_shard(shard_path)
        for sample in shard_data:
            tri_np = _ensure_numpy_float32(sample)
            pending_tris.append(tri_np)

            if len(pending_tris) >= args.batch_size:
                flush_pending()

            if args.max_samples > 0 and exported_count + len(pending_tris) >= args.max_samples:
                remaining = args.max_samples - exported_count
                if remaining < len(pending_tris):
                    pending_tris[:] = pending_tris[:remaining]
                flush_pending()
                break

        if args.max_samples > 0 and exported_count >= args.max_samples:
            break

    flush_pending()




    # metadata["processed_shard_count"] = int(processed_shard_count)
    metadata["sample_count"] = int(len(output_samples))
    if output_samples:
        metadata["embedding_dim"] = int(output_samples[0]["embedding"].numel())

    payload = {
        "metadata": metadata,
        "samples": output_samples,
    }
    torch.save(payload, output_path)

    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] Shards read: {processed_shard_count}")
    print(f"[DONE] Sample count: {len(output_samples)}")
    if output_samples:
        print(f"[DONE] Embedding dim: {int(output_samples[0]['embedding'].numel())}")
        print(f"[DONE] Frequency map shape: {tuple(output_samples[0]['freq_real'].shape)}")
        print(
            "[DONE] Sample schema: "
            "{'sample_index':int, 'triangles':(T,3,2), 'embedding':(D), 'freq_real':(H,W), 'freq_imag':(H,W)}"
        )
    else:
        print("[DONE] No samples exported.")

from torch.utils.data import DataLoader, Dataset,random_split
import segmentation_models_pytorch as smp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
class Res_Dataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_tensor = item['embedding'].detach().cpu().float()
        label_tensor = item['imag'].detach().cpu().float()
        # 读取时转回 float32 供模型计算
        return input_tensor, label_tensor

def calculate_iou(pred, target, threshold=0.5):
    """计算 IoU (交并比)，这是衡量几何修复质量的黄金标准"""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def total_variation_loss(img):
    """TV Loss: 严厉惩罚孤立像素点和噪点，强制输出大块平滑区域"""
    pixel_dif1 = img[:, :, 1:, :] - img[:, :, :-1, :]
    pixel_dif2 = img[:, :, :, 1:] - img[:, :, :, :-1]
    return torch.mean(torch.abs(pixel_dif1)) + torch.mean(torch.abs(pixel_dif2))

def spectral_consistency_loss(pred, target):
    """Spectral Loss: 强制预测图在频域上与真实图一致，促成闭环"""
    fft_pred = torch.fft.rfft2(pred)
    fft_target = torch.fft.rfft2(target)
    return torch.mean(torch.abs(fft_pred - fft_target))

def train(args):

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    data = torch.load(args.train_data_path, map_location='cpu')
    meta_data = data['metadata']
    data_samples = data['samples']
    dataset = Res_Dataset(data_samples)

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    
    # 锁定随机种子 42，确保多卡切分的集合完全一致，不会发生串数据的情况
    train_set, val_set = random_split(
        dataset,[train_len, val_len], 
        generator=torch.Generator().manual_seed(42)
    )
    
    
    # 为训练集和验证集增加 DistributedSampler
    train_sampler = DistributedSampler(train_set)
    val_sampler = DistributedSampler(val_set, shuffle=False)

    # DDP 环境下，DataLoader 不要写 shuffle=True，由 sampler 负责打乱
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)


    model = TransUNetdecoder(embed_dim=384).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    dice_loss = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    model.train()
    best_iou = 0.0
    for epoch in range(args.epoch):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        train_loss = 0
        # 只有主进程显示进度条
        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch} [Train]")
        else:
            pbar = train_loader
  
        
        for x, y in pbar:
            # inputs: [B, 513, 384], labels: [B, 1, 256, 256]
            x = x.to(device)
            y = y.unsqueeze(1).to(device)
            optimizer.zero_grad()
       
            pred = model(x) # Logits 输出
                
            # 计算三重 Loss
            l_dice = dice_loss(pred, y)
            l_tv = total_variation_loss(pred)
            l_spec = spectral_consistency_loss(pred, y)

            loss = l_dice + 0.1 * l_tv + 0.05 * l_spec
            # 反向传播
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
            if local_rank == 0:
                pbar.set_postfix({'TotLoss': f"{loss.item():.3f}", 'Dice': f"{l_dice.item():.3f}"})

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        if local_rank == 0:
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epoch} [Val  ]", leave=False)
        else:
            val_pbar = val_loader
            
        with torch.no_grad():
            for x, y in val_pbar:
                x = x.to(device)
                y = y.unsqueeze(1).to(device)
                pred = model(x)
                
                # 验证集也计算相同的 Loss 以便对比
                l_dice = dice_loss(pred, y)
                l_tv = total_variation_loss(pred)
                l_spec = spectral_consistency_loss(pred, y)
                loss = l_dice + 0.1 * l_tv + 0.05 * l_spec
                
                val_loss += loss.item()
                val_iou += calculate_iou(pred, y)

        metrics = torch.tensor([train_loss, val_loss, val_iou], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        # 全局 batch 总数 (每张卡的 batch 数量 × 显卡数)
        global_train_batches = len(train_loader) * world_size
        global_val_batches = len(val_loader) * world_size
        
        avg_train_loss = metrics[0].item() / global_train_batches
        avg_val_loss = metrics[1].item() / global_val_batches
        avg_val_iou = metrics[2].item() / global_val_batches
        # --- C. 成绩播报与存档 (仅主进程) ---
        if local_rank == 0:
            epoch_time = time.time() - start_time
            print(f"✨ Epoch {epoch+1} 结束 ({epoch_time:.1f}s)")
            print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

            # ⚠️ 在 DDP 下保存权重，需要用 model.module.state_dict()，这样日后单卡推理才能正常读取
            state_dict_to_save = model.module.state_dict()

            # 1. 保存最佳模型
            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                torch.save(state_dict_to_save, os.path.join(args.save_dir, 'v2_unet_best.pth'))
                print(f"   🏆 发现最佳 V2 模型！当前最高 IoU: {best_iou:.4f}，已保存。")

            # 2. 每5轮保存一个定期备份
            if (epoch + 1) % 20 == 0:
                ckpt_name = f'v2_unet_epoch_{epoch+1}.pth'
                torch.save(state_dict_to_save, os.path.join(args.save_dir, ckpt_name))
                print(f"   💾 定期备份已保存: {ckpt_name}")

    # 训练彻底结束后，保存最后一轮的快照
    if local_rank == 0:
        torch.save(model.module.state_dict(), os.path.join(args.save_dir, 'v2_unet_last.pth'))
        print(f"🎉 训练全部完成！最优验证 IoU 锁定在: {best_iou:.4f}")

    # 销毁进程组，退出 DDP
    dist.destroy_process_group()

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    # getdata(args)
    train(args)

