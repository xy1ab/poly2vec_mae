import os
import sys
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from matplotlib.path import Path

# 解决路径依赖
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.model_siren import FiLMSirenOCF

def calculate_iou(pred_bin, gt_bin):
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return (intersection + 1e-6) / (union + 1e-6)

def render_gt_to_grid_fixed(tris_norm, res=256):
    # 1. 强制创建黑背景画布
    fig_tmp = plt.figure(figsize=(res/100, res/100), dpi=100, facecolor='black')
    ax_tmp = fig_tmp.add_axes([0, 0, 1, 1], facecolor='black')
    
    ax_tmp.set_xlim(-1, 1)
    ax_tmp.set_ylim(-1, 1)
    ax_tmp.axis('off')
    
    for tri in tris_norm:
        poly = plt.Polygon(tri, facecolor='white', edgecolor='white', linewidth=0.5)
        ax_tmp.add_patch(poly)
    
    # 2. 关键修复点：使用 buffer_rgba 替代已删除的 tostring_rgba
    fig_tmp.canvas.draw()
    rgba_buffer = fig_tmp.canvas.buffer_rgba() # 获取 RGBA 缓存
    data = np.frombuffer(rgba_buffer, dtype=np.uint8)
    
    # 3. 形状转换：RGBA 是 4 通道
    data = data.reshape(res, res, 4) 
    
    # 4. 提取红色通道 (index 0) 判定 0/1
    grid = (data[:, :, 0] > 127).astype(np.float32)
    
    # 5. 翻转 y 轴对齐坐标系
    grid = np.flipud(grid)
    plt.close(fig_tmp)
    return grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res = args.resolution

    # 1. 加载配置
    config_path = os.path.join(PROJECT_ROOT, "configs/recons.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. 加载数据 (处理 Numpy 兼容性)
    import numpy
    if not hasattr(numpy, '_core'): sys.modules['numpy._core'] = numpy.core
    
    data_path = os.path.join(PROJECT_ROOT, cfg['data_path'])
    print(f"📦 正在载入全量数据: {data_path}")
    all_data = torch.load(data_path, weights_only=False, map_location='cpu')
    test_indices = torch.load(os.path.join(PROJECT_ROOT, cfg['test_indices_path']))
    
    # 3. 加载模型
    model = FiLMSirenOCF(
        embed_dim=cfg['embed_dim'], 
        hidden_dim=cfg['hidden_dim'], 
        num_layers=cfg['num_layers']
    ).to(device)
    model_path = os.path.join(PROJECT_ROOT, cfg['save_dir'], 'siren_ocf_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. 生成网格
    x = np.linspace(-1, 1, res)
    y = np.linspace(1, -1, res)
    X, Y = np.meshgrid(x, y)
    grid_coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    grid_tensor = torch.from_numpy(grid_coords).float().unsqueeze(0).to(device)

    # 5. 抽样测试
    selected_indices = random.sample(list(test_indices), args.num_samples)
    fig, axes = plt.subplots(args.num_samples, 2, figsize=(10, 5 * args.num_samples))
    if args.num_samples == 1: axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(selected_indices):
        sample = all_data[idx]
        
        # --- 兼容性修复: 处理 Embedding ---
        v = sample['embedding']
        if not torch.is_tensor(v):
            v = torch.from_numpy(v)
        v = v.float().unsqueeze(0).to(device)
        
        # --- 兼容性修复: 处理三角形数据 ---
        tris_raw = sample['triangles']
        if torch.is_tensor(tris_raw):
            tris_raw = tris_raw.numpy()
            
        meta = sample['meta'] 
        if torch.is_tensor(meta):
            meta = meta.numpy()
        
        # 归一化缩放
        cx, cy, s = meta[0], meta[1], meta[2]
        tris_norm = (tris_raw - np.array([cx, cy])) / (s + 1e-9)
        
        # 诊断打印
        print(f"样本 {idx} | 归一化后坐标 Min/Max: {tris_norm.min():.2f} / {tris_norm.max():.2f}")

        # A. 模型预测
        with torch.no_grad():
            pred_prob = model(grid_tensor, v)
            pred_grid = pred_prob.reshape(res, res).cpu().numpy()
            pred_bin = (pred_grid > 0.5).astype(np.float32)

        # B. 渲染真值
        gt_grid = render_gt_to_grid_fixed(tris_norm, res=res)

        # C. 计算 IoU
        iou = calculate_iou(pred_bin, gt_grid)
        print(f"      -> 最终 IoU: {iou:.4f}")

        # D. 画图
        axes[i, 0].imshow(pred_bin, cmap='gray', origin='lower', extent=[-1, 1, -1, 1])
        axes[i, 0].set_title(f"Model Result (IoU: {iou:.4f})")
        axes[i, 1].imshow(gt_grid, cmap='gray', origin='lower', extent=[-1, 1, -1, 1])
        axes[i, 1].set_title(f"GT Label (Idx: {idx})")
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'eval_diagnostic_{int(time.time())}.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ 诊断完成！结果图路径: {save_path}")

if __name__ == "__main__":
    main()