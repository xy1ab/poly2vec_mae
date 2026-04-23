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
import time
import json
import hashlib
from functools import partial
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import segmentation_models_pytorch as smp
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR
from pytorch_msssim import ssim
import webdataset as wds
import glob


def load_samples(index_file):
    with open(index_file, "r") as f:
        index = json.load(f)
        train_info = index['train']
        val_info = index['val']        
    return train_info, val_info



def sample_belongs_to_rank(sample, rank, world_size):
    key = sample["__key__"]
    if not isinstance(key, str):
        key = key.decode("utf-8")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    sample_rank = int.from_bytes(digest, "little") % world_size
    return sample_rank == rank


def get_wds_loader(url_pattern, batch_size, total_samples, is_training=True, num_workers=4, split_by_rank=True, split_samples_by_rank=False):
    file_list = sorted(glob.glob(url_pattern))
    
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()} found {len(file_list)} shards for {url_pattern}")
        
    if not file_list:
        raise FileNotFoundError(f"未找到匹配的文件: {url_pattern}")

    # nodesplitter=wds.split_by_node 自动实现 DDP 分片
    shard_shuffle_buffer = 5000 if is_training else 0
    nodesplitter = wds.split_by_node if split_by_rank else None
    dataset = wds.WebDataset(file_list, shardshuffle=shard_shuffle_buffer, nodesplitter=nodesplitter, empty_check=False)

    if dist.is_initialized() and split_samples_by_rank:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dataset = dataset.select(partial(sample_belongs_to_rank, rank=rank, world_size=world_size))

    dataset = (
        dataset
        .decode("torch")
        .to_tuple("input.npy", "label.npy") # 根据你保存的 key
        .batched(batch_size, partial=not is_training)
    )

    epoch_batches = None
    if dist.is_initialized() and is_training:
        world_size = dist.get_world_size() if split_by_rank else 1
        epoch_batches = total_samples // (batch_size * world_size)
        if epoch_batches < 1:
            raise ValueError("训练样本数小于全局 batch size，无法在 drop_last=True 下安全启动 DDP")

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
    parser.add_argument("--save_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_ckpt")
    parser.add_argument("--test_data_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_ckpt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--dice_weight", type=float, default=1.)
    parser.add_argument("--bce_weight", type=float, default=1.)
    parser.add_argument("--ssim_weight", type=float, default=0.3)
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    # ==========================================
    # 📖 读取配置
    # ==========================================
    index_file = args.index_file
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    lr = float(args.lr)
    warmup_epochs = int(args.warmup_epochs)
    num_workers = int(args.num_workers)
    save_dir = args.save_dir
    save_freq = args.save_freq
    eval_every = args.eval_every

    dice_weight = float(args.dice_weight)
    bce_weight = float(args.bce_weight)
    ssim_weight = float(args.ssim_weight)

    # ==========================================
    # ⚙️ DDP 初始化
    # ==========================================
    local_rank, world_size, device = setup_distributed()

    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"🚀 训练启动！DDP进程数: {world_size}")
        print(f"📋 配置:")
        print(f"   数据路径: {index_file}")
        print(f"   批次大小: {batch_size}")
        print(f"   训练轮数: {epochs}")
        print(f"   学习率: {lr}")
        print(f"   验证频率: 每 {eval_every} 轮")

    # ==========================================
    # 1. 数据加载
    # ==========================================

    train_info, val_info = load_samples(index_file)
    
    train_urls = os.path.join(train_info['path'],"train-*.tar")
    val_urls = os.path.join(val_info['path'], "val-*.tar") 
    train_loader = get_wds_loader(train_urls, batch_size, total_samples=train_info['num_samples'], is_training=True, num_workers=num_workers, split_by_rank=True)
    val_loader = get_wds_loader(val_urls, batch_size, total_samples=val_info['num_samples'], is_training=False, num_workers=num_workers, split_by_rank=False, split_samples_by_rank=True)
    log_interval = train_info['num_samples'] // 5
    # ==========================================
    # 2. 模型初始化
    # ==========================================
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
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
            logits = model(input)
            pred = torch.sigmoid(logits)

            l_dice = dice_loss(logits, label)
            l_bce = bce_loss(logits, label)
            l_ssim = ssim_loss(pred, label)

            loss = dice_weight * l_dice + bce_weight * l_bce + ssim_weight * l_ssim

            loss.backward()
            optimizer.step()
            
            batch_samples = input.size(0)
            train_loss += loss.item() * batch_samples
            train_count += batch_samples
            if local_rank == 0 and train_count % log_interval == 0:
                print(f"dice_loss: {l_dice.item():.3f}, bce_loss: {l_bce.item():.3f}")
        
        # 指标同步
        train_loss_tensor = torch.tensor(train_loss, device=device)
        train_count = torch.tensor(train_count, device=device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_count)
        avg_train_loss = train_loss_tensor.item() / train_count.item()
        scheduler.step()

        # --- B. 验证（每 eval_every 轮执行一次）---
        avg_val_loss = None
        avg_val_iou = None
        if (epoch + 1) % eval_every == 0:
            val_loss = 0.0
            val_iou = 0.0
            val_count = 0
            model.eval()
            eval_model = model.module
            with torch.no_grad():
                if local_rank == 0:
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
                else:
                    val_pbar = val_loader

                for input, label in val_pbar:
                    input, label = input.to(device), label.float().to(device)
                    logits = eval_model(input)
                    pred = torch.sigmoid(logits)

                    l_dice = dice_loss(logits, label)
                    l_bce = bce_loss(logits, label)
                    l_ssim = ssim_loss(pred, label)
                    loss = dice_weight * l_dice + bce_weight * l_bce + ssim_weight * l_ssim

                    batch_samples = input.size(0)
                    val_loss += loss.item() * batch_samples
                    val_iou += calculate_iou(pred, label) * batch_samples
                    val_count += batch_samples

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
