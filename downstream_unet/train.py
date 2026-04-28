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
from dataloader import get_wds_loader, load_samples



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
    
    dist.init_process_group(backend='nccl')
    return local_rank, world_size, device
def build_arg_parser():
    parser = argparse.ArgumentParser(description="downstream_UNet")
    parser.add_argument("--index_file", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_traindataset/index_file.json")
    parser.add_argument("--save_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_ckpt")
    parser.add_argument("--test_data_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_ckpt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
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
    base_lr = float(args.lr)
    target_lr = base_lr * (world_size**0.5)
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"🚀 训练启动！DDP进程数: {world_size}")
        print(f"📋 配置:")
        print(f"   数据路径: {index_file}")
        print(f"   批次大小: {batch_size}")
        print(f"   训练轮数: {epochs}")
        print(f"   学习率: {target_lr}")
        print(f"   验证频率: 每 {eval_every} 轮")

    # ==========================================
    # 1. 数据加载
    # ==========================================

    train_info, val_info = load_samples(index_file)
    
    train_urls = os.path.join(train_info['path'],"train-*.tar")
    val_urls = os.path.join(val_info['path'], "val-*.tar") 
    train_loader = get_wds_loader(train_urls, batch_size, total_samples=train_info['num_samples'], is_training=True, num_workers=num_workers, split_by_rank=True)
    val_loader = get_wds_loader(val_urls, batch_size, total_samples=val_info['num_samples'], is_training=False, num_workers=num_workers, split_by_rank=False, split_samples_by_rank=True)
    log_interval = 100#train_info['num_samples'] // 5
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

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数量: {total_params / 1e6:.2f} M")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=target_lr, weight_decay=0.01)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
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

        for step, (input, label) in enumerate(train_loader):
            input, label = input.to(device), label.float().to(device)
            optimizer.zero_grad()
            logits = model(input)
            pred = torch.sigmoid(logits)

            l_dice = dice_loss(logits, label)
            l_bce = bce_loss(logits, label)
            l_ssim = ssim_loss(pred, label)

            loss = dice_weight * l_dice + bce_weight * l_bce + ssim_weight * l_ssim

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            

            batch_samples = input.size(0)
            train_loss += loss.item() * batch_samples
            train_count += batch_samples
            if local_rank == 0 and step % log_interval == 0:
                print(f"Epoch: {epoch}, Step: {step} | dice_loss: {l_dice.item():.3f}, bce_loss: {l_bce.item():.3f}")
        
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
      
                for step, (input, label) in enumerate(val_loader):
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
