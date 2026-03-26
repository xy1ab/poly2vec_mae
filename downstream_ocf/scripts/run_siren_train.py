# export PYTHONPATH=$PYTHONPATH:.
# 单卡启动: python scripts/downstream/run_siren_train.py
# 多卡启动: torchrun --nproc_per_node=4 scripts/downstream/run_siren_train.py

import os
import sys
import yaml
import argparse
import time
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ---------------------------------------------------------
# 1. 兼容性补丁：防止加载 Numpy 2.x 数据时在 1.x 环境报错
# ---------------------------------------------------------
if not hasattr(numpy, '_core'):
    sys.modules['numpy._core'] = numpy.core

# ---------------------------------------------------------
# 2. 解决路径依赖
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.model_siren import FiLMSirenOCF
from src.loader_ocf import OCFDataset

# DDP 分布式训练库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ---------------------------------------------------------
# 📐 指标计算函数
# ---------------------------------------------------------
def calculate_iou(pred, target, threshold=0.5):
    """计算占用场点集的 Batch 平均 IoU"""
    p_bin = (pred > threshold).float()
    inter = (p_bin * target).sum(dim=1)
    union = p_bin.sum(dim=1) + target.sum(dim=1) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return iou.mean()

def main():
    # =========================================================
    # 3. 参数解析
    # =========================================================
    parser = argparse.ArgumentParser(description="Full-Blood SIREN OCF Training")
    parser.add_argument('--config', type=str, default='configs/recons.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_points', type=int, default=None)
    args = parser.parse_args()

    # 读取配置文件
    config_path = os.path.join(PROJECT_ROOT, args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 命令行覆盖逻辑
    if args.epochs: cfg['epochs'] = args.epochs
    if args.batch_size: cfg['batch_size'] = args.batch_size
    if args.lr: cfg['lr'] = args.lr
    if args.num_points: cfg['num_points'] = args.num_points

    # =========================================================
    # 4. DDP 环境初始化
    # =========================================================
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1

    # 仅允许主进程打印
    def print_info(*msg):
        if local_rank == 0: print(*msg)

    # =========================================================
    # 5. 数据准备
    # =========================================================
    save_dir = os.path.join(PROJECT_ROOT, cfg['save_dir'])
    if local_rank == 0: os.makedirs(save_dir, exist_ok=True)

    dataset = OCFDataset(
        data_path=os.path.join(PROJECT_ROOT, cfg['data_path']),
        num_points=cfg['num_points'],
        boundary_ratio=cfg['boundary_ratio'],
        jitter_std=cfg.get('jitter_std', 0.005)
    )
    
    total = len(dataset)
    torch.manual_seed(42)
    train_size = int(0.9 * total)
    val_size = int(0.05 * total)
    test_size = total - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    if local_rank == 0:
        indices_save_path = os.path.join(PROJECT_ROOT, cfg['test_indices_path'])
        os.makedirs(os.path.dirname(indices_save_path), exist_ok=True)
        torch.save(test_set.indices, indices_save_path)
        print_info(f"📊 数据就绪: 训练({len(train_set)}) | 验证({len(val_set)}) | 测试({len(test_set)} 封存)")

    # 适配 DDP 的数据加载器
    if is_ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    else:
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

    # =========================================================
    # 6. 初始化模型、优化器、调度器
    # =========================================================
    model = FiLMSirenOCF(
        embed_dim=cfg['embed_dim'], 
        hidden_dim=cfg['hidden_dim'], 
        num_layers=cfg['num_layers'],
        omega_0=cfg.get('omega_0', 30.0)
    ).to(device)
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=1e-5)
    # 动态调整学习率：如果 mIoU 5 轮不涨，学习率减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCELoss()

    # =========================================================
    # 7. 训练大循环
    # =========================================================
    best_iou = 0.0
    print_info(f"🚀 开始训练，总轮数: {cfg['epochs']}")

    for epoch in range(cfg['epochs']):
        if is_ddp: train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        # --- A. 训练阶段 ---
        model.train()
        epoch_train_loss = torch.tensor(0.0).to(device)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]") if local_rank == 0 else train_loader
        
        for v, p, y in pbar:
            v, p, y = v.to(device), p.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(p, v)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.detach()
            
            if local_rank == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- B. 验证阶段 ---
        model.eval()
        epoch_val_loss = torch.tensor(0.0).to(device)
        epoch_val_iou = torch.tensor(0.0).to(device)
        val_steps = 0
        
        with torch.no_grad():
            # 只有主进程看验证集进度
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False) if local_rank == 0 else val_loader
            for v, p, y in val_iter:
                v, p, y = v.to(device), p.to(device), y.to(device)
                preds = model(p, v)
                
                loss = criterion(preds, y)
                iou = calculate_iou(preds, y)
                
                epoch_val_loss += loss
                epoch_val_iou += iou
                val_steps += 1

        # --- C. 多卡同步成绩 ---
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / val_steps
        avg_val_iou = epoch_val_iou / val_steps
        
        if is_ddp:
            # 聚合所有显卡的成绩
            dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM); avg_train_loss /= world_size
            dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM); avg_val_loss /= world_size
            dist.all_reduce(avg_val_iou, op=dist.ReduceOp.SUM); avg_val_iou /= world_size
        
        # 调度器根据全局平均 IoU 调整学习率
        scheduler.step(avg_val_iou)

        # --- D. 存档与报告 ---
        if local_rank == 0:
            duration = time.time() - start_time
            print_info(f"✨ Epoch {epoch+1} 结束 | 耗时: {duration:.1f}s")
            print_info(f"   [Train] Loss: {avg_train_loss.item():.4f}")
            print_info(f"   [Val  ] Loss: {avg_val_loss.item():.4f} | mIoU: {avg_val_iou.item():.4f}")

            model_to_save = model.module if is_ddp else model
            
            # 保存 IoU 创纪录的模型
            if avg_val_iou.item() > best_iou:
                best_iou = avg_val_iou.item()
                torch.save(model_to_save.state_dict(), os.path.join(save_dir, 'siren_ocf_best.pth'))
                print_info(f"   🏆 发现最佳模型! mIoU: {best_iou:.4f}")

            # 每 10 轮强制定期快照
            if (epoch + 1) % 2 == 0:
                torch.save(model_to_save.state_dict(), os.path.join(save_dir, f'siren_ocf_ep{epoch+1}.pth'))

    print_info(f"\n🎉 训练圆满结束！最高验证 mIoU: {best_iou:.4f}")

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()