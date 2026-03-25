# export PYTHONPATH=$PYTHONPATH:.
# python scripts/downstream/run_siren_train.py

import os
import sys
import numpy
# 兼容性补丁：如果数据是用 numpy 2.x 存的，而当前环境是 1.x，这一步能防止崩溃
if not hasattr(numpy, '_core'):
    sys.modules['numpy._core'] = numpy.core
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time

# 关键：将项目根目录加入搜索路径，确保能 import src 文件夹
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.downstream.task_recons.model_siren import FiLMSirenOCF
from src.downstream.task_recons.loader_ocf import OCFDataset

def calculate_iou(pred, target, threshold=0.5):
    """
    计算占用场点集的 Batch 平均 IoU
    pred: [B, N, 1], target: [B, N, 1]
    """
    p_bin = (pred > threshold).float()
    inter = (p_bin * target).sum(dim=1)
    union = p_bin.sum(dim=1) + target.sum(dim=1) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def main():
    # ==========================================
    # 1. 加载配置 (读取 configs/downstream/recons.yaml)
    # ==========================================
    config_path = os.path.join(ROOT_DIR, "configs/downstream/recons.yaml")
    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 [OCF Training] 启动！")
    print(f"📍 实验结果将保存至: {os.path.abspath(save_dir)}")

    # ==========================================
    # 2. 数据准备 (30万级样本)
    # ==========================================
    dataset = OCFDataset(
        data_path=cfg['data_path'],
        num_points=cfg['num_points'],
        boundary_ratio=cfg['boundary_ratio']
    )
    
    total = len(dataset)
    torch.manual_seed(42) # 固定种子确保测试集不泄露
    
    # 按照 90/5/5 逻辑切分
    train_size = int(0.9 * total)
    val_size = int(0.05 * total)
    test_size = total - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    # 封存测试集索引（用于未来求交集应用）
    torch.save(test_set.indices, cfg['test_indices_path'])
    print(f"📊 数据就绪: 训练({len(train_set)}) | 验证({len(val_set)}) | 测试({len(test_set)} 封存)")

    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

    # ==========================================
    # 3. 初始化模型、优化器与损失函数
    # ==========================================
    model = FiLMSirenOCF(
        embed_dim=cfg['embed_dim'], 
        hidden_dim=cfg['hidden_dim'], 
        num_layers=cfg['num_layers'],
        omega_0=cfg['omega_0']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=1e-5)
    
    # 学习率调度器：当 Val Loss 停滞时自动减半学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.BCELoss()

    # ==========================================
    # 4. 核心训练循环
    # ==========================================
    best_val_iou = 0.0
    epochs = cfg['epochs']

    for epoch in range(epochs):
        start_time = time.time()
        
        # --- A. 训练阶段 ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for v, p, y in pbar:
            v, p, y = v.to(device), p.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(p, v)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- B. 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for v, p, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val  ]", leave=False):
                v, p, y = v.to(device), p.to(device), y.to(device)
                preds = model(p, v)
                
                val_loss += criterion(preds, y).item()
                val_iou += calculate_iou(preds, y)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        # 更新学习率调度器
        scheduler.step(avg_val_loss)

        # --- C. 日志播报与存档 ---
        epoch_time = time.time() - start_time
        print(f"✨ Epoch {epoch+1} 结束 ({epoch_time:.1f}s)")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_iou:.4f}")

        # 保存表现最好的模型
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), os.path.join(save_dir, 'siren_ocf_best.pth'))
            print(f"   🏆 发现更强模型！已保存至 siren_ocf_best.pth")

        # 每 10 轮保存一个定期快照
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'siren_ocf_epoch_{epoch+1}.pth'))

    print(f"\n🎉 训练全部完成！最高验证 mIoU: {best_val_iou:.4f}")

if __name__ == "__main__":
    main()