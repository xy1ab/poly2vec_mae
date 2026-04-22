#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2026/04/21 13:49:13
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import os
import sys
import time
import math
import json
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
import argparse
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pytorch_msssim import ssim
import webdataset as wds
import glob


def load_total_samples(index_file, default_total=32272):
    if not index_file or not os.path.exists(index_file):
        return default_total
    with open(index_file, "r") as f:
        index = json.load(f)
    if isinstance(index, list):
        return sum(int(item["num_samples"]) for item in index)
    if isinstance(index, dict) and "num_samples" in index:
        return int(index["num_samples"])
    return default_total


def split_train_val_counts(total_samples, train_ratio=0.9):
    threshold = int(train_ratio * 100)
    full_cycles, remainder = divmod(total_samples, 100)
    train_samples = full_cycles * threshold + min(remainder, threshold)
    return train_samples, total_samples - train_samples


def get_wds_loader(url_pattern, batch_size, is_training=True, total_samples=32272, num_workers=4, split_by_rank=True):
    file_list = sorted(glob.glob(url_pattern))
    
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()} found {len(file_list)} shards for {url_pattern}")
        
    if not file_list:
        raise FileNotFoundError(f"未找到匹配的文件: {url_pattern}")

    # nodesplitter=wds.split_by_node 自动实现 DDP 分片
    shard_shuffle_buffer = 1000 if is_training else 0
    nodesplitter = wds.split_by_node if split_by_rank else None
    dataset = wds.WebDataset(file_list, shardshuffle=shard_shuffle_buffer, nodesplitter=nodesplitter, empty_check=False)

    dataset = (
        dataset
        .decode("torch")
        .to_tuple("input.npy", "label.npy") # 根据你保存的 key
        .batched(batch_size, partial=not is_training)
    )

    epoch_batches = None
    if dist.is_initialized():
        world_size = dist.get_world_size() if split_by_rank else 1
        if is_training:
            epoch_batches = total_samples // (batch_size * world_size)
            if epoch_batches < 1:
                raise ValueError("训练样本数小于全局 batch size，无法在 drop_last=True 下安全启动 DDP")
        else:
            epoch_batches = max(1, math.ceil(total_samples / batch_size))

    # WDS 已经完成了 batch 组装，DataLoader 的 batch_size 需设为 None
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=True)
    if epoch_batches is not None:
        loader = loader.with_epoch(nbatches=epoch_batches)
    return loader

def calculate_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def ssim_loss(pred, target, data_range=1.0, size_average=True):
    # ssim 默认返回的是相似度 (0 到 1 之间)，值越大越相似
    # Loss 需要定义为 1 - similarity
    return 1 - ssim(pred, target, data_range=data_range, size_average=size_average)

def spectral_loss(pred, target):
    fft_pred = torch.fft.rfft2(pred)
    fft_target = torch.fft.rfft2(target)
    return torch.mean(torch.abs(fft_pred - fft_target))

def setup_distributed():        
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    dist.init_process_group(backend='nccl', device_id=device)
    return local_rank, world_size, device
def build_arg_parser():
    parser = argparse.ArgumentParser(description="downstream_UNet")
    parser.add_argument("--index_file", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_traindataset/index_file.json")
    parser.add_argument("--data_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_traindataset")
    parser.add_argument("--save_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_ckpt")
    parser.add_argument("--test_data_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_ckpt")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--dice_weight", type=float, default=1.)
    parser.add_argument("--ssim_weight", type=float, default=0.5)
    parser.add_argument("--spec_weight", type=float, default=0.1)
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    # ==========================================
    # 📖 读取配置
    # ==========================================
    data_dir = args.data_dir
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    lr = float(args.lr)
    num_workers = int(args.num_workers)
    save_dir = args.save_dir
    save_freq = args.save_freq
    eval_every = args.eval_every

    dice_weight = float(args.dice_weight)
    ssim_weight = float(args.ssim_weight)
    spec_weight = float(args.spec_weight)

    # ==========================================
    # ⚙️ DDP 初始化
    # ==========================================
    local_rank, world_size, device = setup_distributed()

    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"🚀 训练启动！DDP进程数: {world_size}")
        print(f"📋 配置:")
        print(f"   数据路径: {data_dir}")
        print(f"   批次大小: {batch_size}")
        print(f"   训练轮数: {epochs}")
        print(f"   学习率: {lr}")
        print(f"   验证频率: 每 {eval_every} 轮")

    # ==========================================
    # 1. 数据加载
    # ==========================================
    train_urls = os.path.join(data_dir, "train","train-*.tar")
    val_urls = os.path.join(data_dir, "val", "val-*.tar")
    total_samples = load_total_samples(args.index_file)
    train_samples, val_samples = split_train_val_counts(total_samples)
    
    train_loader = get_wds_loader(train_urls, batch_size, is_training=True, total_samples=train_samples, num_workers=num_workers, split_by_rank=True)
    val_loader = get_wds_loader(val_urls, batch_size, is_training=False, total_samples=val_samples, num_workers=num_workers, split_by_rank=False)
    # ==========================================
    # 2. 模型初始化
    # ==========================================
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation='sigmoid'
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    dice_loss = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # ==========================================
    # 3. 训练循环
    # ==========================================
    best_iou = 0.0
    start_time_total = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        # --- A. 训练 ---
        model.train()
        train_loss = 0.0
        train_count = 0

        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        else:
            pbar = train_loader

        for input, label in pbar:
            input, label = input.to(device), label.float().to(device)
            optimizer.zero_grad()
            pred = model(input)

            l_dice = dice_loss(pred, label)
            l_ssim = ssim_loss(pred, label)
            l_spec = spectral_loss(pred, label)

            loss = dice_weight * l_dice + ssim_weight * l_ssim + spec_weight * l_spec

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_count += 1
            if local_rank == 0:
                pbar.set_postfix({"dice_loss": f"{l_dice.item():.3f}"})
        
        # 指标同步
        train_loss_tensor = torch.tensor(train_loss, device=device)
        train_count = torch.tensor(train_count, device=device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count)
        avg_train_loss = train_loss_tensor.item() / train_count.item()

        # --- B. 验证（每 eval_every 轮执行一次）---
        avg_val_loss = None
        avg_val_iou = None
        if (epoch + 1) % eval_every == 0:
            val_loss = 0.0
            val_iou = 0.0
            val_count = 0
            model.eval()
            with torch.no_grad():
                if local_rank == 0:
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
                else:
                    val_pbar = val_loader

                for input, label in val_pbar:
                    input, label = input.to(device), label.float().to(device)
                    pred = model(input)

                    l_dice = dice_loss(pred, label)
                    l_ssim = ssim_loss(pred, label)
                    l_spec = spectral_loss(pred, label)
                    loss = dice_weight * l_dice + ssim_weight * l_ssim + spec_weight * l_spec

                    val_loss += loss.item()
                    val_iou += calculate_iou(pred, label)
                    val_count += 1

            val_loss_tensor = torch.tensor(val_loss, device=device)
            val_iou_tensor = torch.tensor(val_iou, device=device)
            val_count_tensor = torch.tensor(val_count, device=device)

            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_iou_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / max(val_count_tensor.item(), 1)
            avg_val_iou = val_iou_tensor.item() / max(val_count_tensor.item(), 1)

        # --- C. 输出与保存 ---
        if local_rank == 0:
            epoch_time = time.time() - epoch_start
            print(f"✨ Epoch {epoch+1} 结束 ({epoch_time:.1f}s)")
            if avg_val_loss is None:
                print(f"   Train Loss: {avg_train_loss:.4f}")
            else:
                print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

            state_dict = model.module.state_dict()

            if avg_val_iou is not None and avg_val_iou > best_iou:
                best_iou = avg_val_iou
                torch.save(state_dict, os.path.join(save_dir, 'unet_best.pth'))
                print(f"   🏆 发现最佳模型！IoU: {best_iou:.4f}")

            if (epoch + 1) % save_freq == 0:
                torch.save(state_dict, os.path.join(save_dir, f'unet_epoch_{epoch+1}.pth'))
                print(f"   💾 已保存 checkpoint")

        dist.barrier()

    # 结束
    if local_rank == 0:
        total_h = (time.time() - start_time_total) / 3600.0
        print(f"\n🎉 训练完成！最佳 IoU: {best_iou:.4f} | 总耗时: {total_h:.2f}h")

    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
