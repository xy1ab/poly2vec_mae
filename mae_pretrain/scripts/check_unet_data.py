#!/usr/bin/env python3
"""
检测生成的 U-Net 训练数据
"""

import torch
import numpy as np  # ✅ 添加这一行
import matplotlib.pyplot as plt
from pathlib import Path

def check_generated_data(data_path, num_samples=3):
    """检测生成的数据"""
    
    # 加载数据
    data = torch.load(data_path)
    print(f"✅ 加载成功: {data_path}")
    print(f"   样本数量: {len(data)}")
    
    # 检查第一个样本的结构
    sample = data[0]
    print(f"\n📦 样本结构:")
    print(f"   keys: {sample.keys()}")
    print(f"   input shape: {sample['input'].shape}")
    print(f"   label shape: {sample['label'].shape}")
    
    # 数值范围检查
    x = sample['input']
    y = sample['label']
    print(f"\n📊 数值范围:")
    print(f"   input: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
    print(f"   label: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
    print(f"   label 唯一值: {y.unique().tolist()}")
    
    # 检查是否有异常
    print(f"\n🔍 异常检测:")
    print(f"   input 包含 NaN: {torch.isnan(x).any().item()}")
    print(f"   input 包含 Inf: {torch.isinf(x).any().item()}")
    print(f"   label 包含 NaN: {torch.isnan(y).any().item()}")
    
    # 检查各通道
    print(f"\n🎨 各通道统计:")
    channels = ['模糊空间图', '幅值图', '相位图']
    for i, name in enumerate(channels):
        print(f"   {name}: min={x[i].min():.4f}, max={x[i].max():.4f}, mean={x[i].mean():.4f}")
    
    # 可视化前几个样本
    print(f"\n🖼️ 可视化前 {num_samples} 个样本...")
    
    for idx in range(min(num_samples, len(data))):
        sample = data[idx]
        x = sample['input']
        y = sample['label']
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # 模糊空间图
        im0 = axes[0].imshow(x[0], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Blurred Map\n{x[0].min():.2f}~{x[0].max():.2f}')
        plt.colorbar(im0, ax=axes[0])
        
        # 幅值图
        im1 = axes[1].imshow(x[1], cmap='viridis')
        axes[1].set_title(f'Magnitude\n{x[1].min():.2f}~{x[1].max():.2f}')
        plt.colorbar(im1, ax=axes[1])
        
        # 相位图
        im2 = axes[2].imshow(x[2], cmap='RdBu', vmin=-3.14, vmax=3.14)
        axes[2].set_title(f'Phase\n{x[2].min():.2f}~{x[2].max():.2f}')
        plt.colorbar(im2, ax=axes[2])
        
        # GT 掩码
        im3 = axes[3].imshow(y[0], cmap='gray', vmin=0, vmax=1)
        axes[3].set_title(f'GT Mask\n0: {(y[0]==0).sum():.0f}, 1: {(y[0]==1).sum():.0f}')
        plt.colorbar(im3, ax=axes[3])
        
        # 叠加图：模糊图 + GT 轮廓
        axes[4].imshow(x[0], cmap='gray', vmin=0, vmax=1)
        # 绘制 GT 轮廓
        from skimage import measure
        y_np = y[0].numpy()
        contours = measure.find_contours(y_np, 0.5)
        for contour in contours:
            axes[4].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1)
        axes[4].set_title('Blurred + GT Contour')
        axes[4].axis('off')
        
        plt.suptitle(f'Sample {idx}')
        plt.tight_layout()
        plt.savefig(f'check_sample_{idx}.png', dpi=150)
        plt.close()
        print(f"   ✅ 已保存: check_sample_{idx}.png")
    
    # 统计信息
    print(f"\n📈 统计信息:")
    all_ious = []
    for idx in range(min(100, len(data))):  # 只检查前100个
        sample = data[idx]
        pred_binary = (sample['input'][0] > 0.5).float()
        gt_binary = sample['label'][0]
        inter = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - inter
        iou = (inter + 1e-7) / (union + 1e-7)
        all_ious.append(iou.item())
    
    print(f"   基于模糊图二值化的 IoU (前100个):")
    print(f"     mean: {np.mean(all_ious):.4f}")
    print(f"     std: {np.std(all_ious):.4f}")
    print(f"     min: {np.min(all_ious):.4f}")
    print(f"     max: {np.max(all_ious):.4f}")
    
    print(f"\n✅ 检测完成！")

if __name__ == "__main__":
    # 修改为你的数据路径
    data_path = "./data/unet_dataset/all_samples.pt"
    check_generated_data(data_path, num_samples=3)