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
from torch.utils.data import DataLoader, Dataset,random_split
import segmentation_models_pytorch as smp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from mae_pretrain.src.models.decoder import TransUNetdecoder
import time
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

def _default_device() -> str:
    """Resolve default runtime device string."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch forward export."""
    parser = argparse.ArgumentParser(
        description="Batch forward triangulated polygon shards into embeddings and MAE frequency maps."
    )
    parser.add_argument("--save_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_checkpoints", help="Output `.pt` file path.")
    parser.add_argument("--data_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/mae_pretrain/outputs/forward_batch/processed_forward_batch_20260407_174737.pt", help="Output `.pt` file path.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--device", type=str, default=_default_device(), help="Runtime device, e.g. cuda or cpu.")
    parser.add_argument("--precision", type=str, default="fp32", help="Runtime precision: fp32/bf16/fp16.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional export cap; 0 means all samples.")
    parser.add_argument("--epoch", type=int, default=1000, help="Optional export cap; 0 means all samples.")
    return parser



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

    data = torch.load(args.data_path, map_location='cpu')
    meta_data= data['meta_data']
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
    train(args)

