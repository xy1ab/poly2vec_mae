# torchrun --nproc_per_node=4 train_0409.py --config configs/recons_unet_single.yaml
# python train_0409.py --config configs/recons_unet_single.yaml  # 单卡

import os
import sys
import time
from pathlib import Path
import torch_musa
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
import argparse
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _CURRENT_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib

    V2Dataset = importlib.import_module("downstream_unet.loaders.loader_single").V2Dataset
else:
    from .loaders.loader import V2Dataset


def calculate_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def total_variation_loss(img):
    pixel_dif1 = img[:, :, 1:, :] - img[:, :, :-1, :]
    pixel_dif2 = img[:, :, :, 1:] - img[:, :, :, :-1]
    return torch.mean(torch.abs(pixel_dif1)) + torch.mean(torch.abs(pixel_dif2))


def spectral_consistency_loss(pred, target):
    fft_pred = torch.fft.rfft2(pred)
    fft_target = torch.fft.rfft2(target)
    return torch.mean(torch.abs(fft_pred - fft_target))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="downstream_UNet")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # ==========================================
    # 📖 读取配置
    # ==========================================
    data_path = cfg['data']['data_path']
    test_dataset_path = cfg['data']['test_dataset']
    batch_size = int(cfg['training']['batch_size'])
    epochs = int(cfg['training']['epochs'])
    learning_rate = float(cfg['training']['learning_rate'])
    num_workers = int(cfg['training'].get('num_workers', 4))
    save_dir = cfg['logging']['save_dir']
    save_freq = cfg['logging'].get('save_freq', 20)
    val_freq = cfg['training'].get('val_freq', 1)  # 新增：验证频率

    dice_weight = float(cfg['loss'].get('dice_weight', 1.0))
    tv_weight = float(cfg['loss'].get('tv_weight', 0.1))
    spec_weight = float(cfg['loss'].get('spec_weight', 0.05))

    encoder_name = cfg['model'].get('encoder_name', 'resnet34')
    encoder_weights = cfg['model'].get('encoder_weights', None)
    in_channels = cfg['model'].get('in_channels', 3)
    classes = cfg['model'].get('classes', 1)

    # ==========================================
    # ⚙️ DDP 初始化
    # ==========================================
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.musa.set_device(local_rank)
        device = torch.device(f"musa:{local_rank}")
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device('musa' if torch.musa.is_available() else 'cpu')
        world_size = 1

    def print_rank0(*msg):
        if local_rank == 0:
            print(*msg)

    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        print(f"🚀 训练启动！DDP进程数: {world_size}")
        print(f"📋 配置:")
        print(f"   数据路径: {data_path}")
        print(f"   批次大小: {batch_size}")
        print(f"   训练轮数: {epochs}")
        print(f"   学习率: {learning_rate}")
        print(f"   验证频率: 每 {val_freq} 轮")

    # ==========================================
    # 1. 数据加载
    # ==========================================
    dataset = V2Dataset(data_path)
    total_len = len(dataset)
    if total_len == 0:
        raise ValueError(f"❌ 找不到数据: {data_path}")

    train_len = int(0.9 * total_len)
    val_len = int(0.05 * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    if local_rank == 0:
        torch.save(test_set.indices, test_dataset_path)
        print(f"📊 数据切分: 训练({train_len}) | 验证({val_len}) | 测试({test_len} 已封存)")

    if is_ddp:
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set, shuffle=False)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler,
                                num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    # ==========================================
    # 2. 模型初始化
    # ==========================================
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation='sigmoid'
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion_dice = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # ==========================================
    # 3. 训练循环
    # ==========================================
    best_iou = 0.0
    start_time_total = time.time()

    for epoch in range(epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        # --- A. 训练 ---
        model.train()
        train_loss = 0.0

        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        else:
            pbar = train_loader

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)

            l_dice = criterion_dice(pred, y)
            l_tv = total_variation_loss(pred)
            l_spec = spectral_consistency_loss(pred, y)

            loss = dice_weight * l_dice + tv_weight * l_tv + spec_weight * l_spec

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if local_rank == 0:
                pbar.set_postfix({'TotLoss': f"{loss.item():.3f}", 'Dice': f"{l_dice.item():.3f}"})

        # 清理 GPU 缓存（每轮结束）
        torch.musa.empty_cache()

        # --- B. 验证（每 val_freq 轮执行一次）---
        val_loss = 0.0
        val_iou = 0.0

        if (epoch + 1) % val_freq == 0:
            model.eval()
            with torch.no_grad():
                if local_rank == 0:
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val  ]", leave=False)
                else:
                    val_pbar = val_loader

                for x, y in val_pbar:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)

                    l_dice = criterion_dice(pred, y)
                    l_tv = total_variation_loss(pred)
                    l_spec = spectral_consistency_loss(pred, y)
                    loss = dice_weight * l_dice + tv_weight * l_tv + spec_weight * l_spec

                    val_loss += loss.item()
                    val_iou += calculate_iou(pred, y)

        # 指标同步
        train_loss_tensor = torch.tensor(train_loss, device=device)
        val_loss_tensor = torch.tensor(val_loss, device=device)
        val_iou_tensor = torch.tensor(val_iou, device=device)

        if is_ddp:
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_iou_tensor, op=dist.ReduceOp.SUM)

        global_train_batches = len(train_loader) * world_size
        global_val_batches = len(val_loader) * world_size

        avg_train_loss = train_loss_tensor.item() / global_train_batches
        avg_val_loss = val_loss_tensor.item() / global_val_batches if val_loss > 0 else 0
        avg_val_iou = val_iou_tensor.item() / global_val_batches if val_iou > 0 else 0

        # --- C. 输出与保存 ---
        if local_rank == 0:
            epoch_time = time.time() - epoch_start
            print(f"✨ Epoch {epoch+1} 结束 ({epoch_time:.1f}s)")
            print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

            state_dict = model.module.state_dict() if is_ddp else model.state_dict()

            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                torch.save(state_dict, os.path.join(save_dir, 'unet_best.pth'))
                print(f"   🏆 发现最佳模型！IoU: {best_iou:.4f}")

            if (epoch + 1) % save_freq == 0:
                torch.save(state_dict, os.path.join(save_dir, f'unet_epoch_{epoch+1}.pth'))
                print(f"   💾 已保存 checkpoint")

    # 结束
    if local_rank == 0:
        total_h = (time.time() - start_time_total) / 3600.0
        print(f"\n🎉 训练完成！最佳 IoU: {best_iou:.4f} | 总耗时: {total_h:.2f}h")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()