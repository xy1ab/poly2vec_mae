#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   eval.py
@Time    :   2026/04/25 15:10:39
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import os
import sys
from pathlib import Path
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import Subset
import time
import yaml
import argparse
from dataloader import get_wds_loader, load_samples


def calculate_metrics(pred, target, input_blur, threshold=0.5):
    """计算硬核数据指标"""
    pred_bin = (pred > threshold).astype(np.float32)
    
    # 1. IoU
    intersection = np.sum(pred_bin * target)
    union = np.sum(pred_bin) + np.sum(target) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # 2. MSE 改善对比
    mse_before = np.mean((input_blur - target) ** 2)
    mse_after = np.mean((pred - target) ** 2)
    
    # 3. 绝对错判的像素个数
    wrong_pixels = np.sum(np.abs(pred_bin - target))
    
    # 4. 犹豫像素数 (0.1 到 0.9 之间的模糊灰色地带)
    uncertain_before = np.sum((input_blur > 0.1) & (input_blur < 0.9))
    uncertain_after = np.sum((pred > 0.1) & (pred < 0.9))
    
    return iou, mse_before, mse_after, wrong_pixels, uncertain_before, uncertain_after


def build_arg_parser() -> argparse.ArgumentParser:
    """构建南湖平台标准 CLI 解析器"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_traindataset/index_file.json")
    parser.add_argument("--model_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_ckpt/unet_best.pth")
    parser.add_argument("--data_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_traindataset/val") 
    parser.add_argument("--vis_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_ckpe/vis")
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser

def main():
    args = build_arg_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_time = time.strftime("%Y%m%d_%H%M%S")
    
    # 可视化输出目录
    os.makedirs(args.vis_dir, exist_ok=True)
    report_path = os.path.join(args.vis_dir, f'evaluation_report_{current_time}.txt')
    
    # ========== 2. 加载测试集 ==========
    _, val_info = load_samples(args.index_file)
    val_urls = os.path.join(val_info['path'], "val-*.tar") 
    val_loader = get_wds_loader(val_urls, args.batch_size, total_samples=val_info['num_samples'], is_training=False, num_workers=args.num_workers, split_by_rank=False, split_samples_by_rank=True)

    print(f"\n📦 测试集 共抽取 {val_info['num_samples']} 道样本。")
    
    # ========== 3. 加载模型（从 YAML 读取模型参数） ==========
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"🤖 模型加载完毕: {args.model_path}")
    print("开始进行数据分析...\n" + "="*50)
    
    # ========== 4. 随机抽取测试样本 ==========

    num_samples = 2
    sample_indices = random.sample(range(val_loader)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(22, 6 * num_samples))
    plt.subplots_adjust(hspace=0.4)
    
    total_iou = 0.0
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("      Poly2Vec 频域引导模型 - 修复评估报告      \n")
        f.write("==================================================\n\n")
        
        for i, idx in enumerate(sample_indices):
            x, y = test_set[idx]
            # x: [3, 256, 256], y: [1, 256, 256]
            
            with torch.no_grad():
                pred = model(x.unsqueeze(0).to(device))
            
            # 转换为 numpy
            img_blur = x[0].numpy()       # 通道0: 模糊空间图
            img_mag = x[1].numpy()        # 通道1: 幅值图
            img_pred = pred.squeeze().cpu().numpy()
            img_gt = y.squeeze().numpy()
            
            iou, mse_bef, mse_aft, wrong_px, unc_bef, unc_aft = calculate_metrics(
                img_pred, img_gt, img_blur
            )
            total_iou += iou
            
            report_str = f"📊[测试样本 Index: {idx}]\n"
            report_str += f"   ▶ IoU (重合度)      : {iou * 100:.2f}%\n"
            report_str += f"   ▶ 像素误差 (MSE)    : 从 {mse_bef:.4f} 降至 {mse_aft:.4f}\n"
            report_str += f"   ▶ 错判像素数        : {int(wrong_px)} 个点画错 (总像素 65536)\n"
            report_str += f"   ▶ 边缘模糊点(灰色)  : 从 {int(unc_bef)} 个锐化至 {int(unc_aft)} 个\n"
            report_str += "-" * 50 + "\n"
            
            print(report_str, end="")
            f.write(report_str)
            
            # 绘图
            axes[i, 0].imshow(img_blur, cmap='plasma')
            axes[i, 0].set_title(f"Input Blur\nMSE: {mse_bef:.4f} | Unclear: {int(unc_bef)}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(img_mag, cmap='viridis')
            axes[i, 1].set_title("Frequency Magnitude")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(img_pred, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f"Prediction\nMSE: {mse_aft:.4f} | Unclear: {int(unc_aft)}\nIoU: {iou*100:.2f}%")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(img_gt, cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title(f"Ground Truth\nWrong: {int(wrong_px)}/65536")
            axes[i, 3].axis('off')
        
        avg_iou = total_iou / num_samples
        summary_str = f"\n🎯 总结: 抽测 {num_samples} 个样本，平均 IoU = {avg_iou * 100:.2f}%\n"
        print(summary_str)
        f.write(summary_str)
    
    img_save_path = os.path.join(vis_dir, f'test_report_{current_time}.png')
    plt.savefig(img_save_path, bbox_inches='tight', dpi=150)
    print(f"✅ 可视化图片: {img_save_path}")
    print(f"📄 评估报告: {report_path}")


if __name__ == '__main__':
    main()