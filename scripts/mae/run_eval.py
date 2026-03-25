import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from utils.fourier.engine import PolyFourierConverter
from utils.config.loader import load_yaml_config
from utils.geometry.rasterize import rasterize_triangles
from utils.io.filesystem import ensure_dir

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")
    
    # 1. 载入 .pt 文件数据
    print(f"载入数据文件: {args.data_path}")
    all_polys = torch.load(args.data_path, weights_only=False)
    
    if args.index >= len(all_polys):
        raise IndexError(f"指定的索引 {args.index} 越界，总共有 {len(all_polys)} 个多边形数据。")
        
    tris = all_polys[args.index]
    print(f"多边形 {args.index} 包含 {tris.shape[0]} 个三角形")
    
    # 2. 原始空间多边形栅格化
    orig_raster = rasterize_triangles(tris, spatial_size=args.spatial_size)
    
    # 3. 初始化 CFT 引擎
    engine = PolyFourierConverter(
        pos_freqs=args.pos_freqs, w_min=args.w_min, w_max=args.w_max, 
        freq_type=args.freq_type, device=device, patch_size=args.patch_size
    )
    
    batch_tris = torch.tensor(tris, dtype=torch.float32, device=device).unsqueeze(0) # [1, N, 3, 2]
    lengths = torch.tensor([tris.shape[0]], device=device)
    
    # 4. 执行 CFT 得到频域幅值和相位
    mag_log, phase = engine.cft_polygon_batch(batch_tris, lengths)
    
    # 【改动点】提取 cos 和 sin 用于可视化分析
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    
    # 还原真实傅里叶幅值用于逆变换
    raw_mag = torch.expm1(mag_log)
    F_uv = raw_mag * torch.exp(1j * phase)
    
    # 5. 执行 ICFT 进行空域重构
    recon_raster = engine.icft_2d(F_uv.squeeze(1), spatial_size=args.spatial_size)
    
    # 转移到CPU用于画图及误差计算
    mag_vis = mag_log.squeeze().cpu().numpy()
    cos_vis = cos_phase.squeeze().cpu().numpy()
    sin_vis = sin_phase.squeeze().cpu().numpy()
    recon_vis = recon_raster.squeeze().cpu().numpy()
    
    # 计算绝对误差图
    diff_vis = np.abs(orig_raster - recon_vis)
    
    # 6. 画图并保存 (【改动点】由 5 个子图扩展为 6 个子图)
    fig, axes = plt.subplots(1, 6, figsize=(36, 6))
    
    im0 = axes[0].imshow(orig_raster, cmap='gray', vmin=0, vmax=1, extent=[-1, 1, -1, 1])
    axes[0].set_title('Original Rasterized Polygon')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(mag_vis, cmap='viridis')
    axes[1].set_title('CFT Magnitude (log1p)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(cos_vis, cmap='viridis', vmin=-1, vmax=1)
    axes[2].set_title('CFT Cos(Phase)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    im3 = axes[3].imshow(sin_vis, cmap='viridis', vmin=-1, vmax=1)
    axes[3].set_title('CFT Sin(Phase)')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    im4 = axes[4].imshow(recon_vis, cmap='gray', vmin=0, vmax=1, extent=[-1, 1, -1, 1])
    axes[4].set_title('ICFT Reconstructed Field')
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    
    im5 = axes[5].imshow(diff_vis, cmap='Reds', vmin=0, vmax=1, extent=[-1, 1, -1, 1])
    axes[5].set_title('Absolute Difference')
    plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    ensure_dir(args.save_dir)
    save_path = os.path.join(args.save_dir, f'cft_visualize_idx_{args.index}.png')
    plt.savefig(save_path, dpi=150)
    print(f"可视化结果已保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "configs", "mae", "eval.yaml"),
        type=str
    )
    pre_args, remaining = pre_parser.parse_known_args()

    config_defaults = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description="CFT and ICFT Visualization")
    parser.add_argument('--config', type=str, default=pre_args.config)
    parser.add_argument('--index', type=int, default=0, help="提取.pt文件中多边形的索引")
    parser.add_argument('--data_path', type=str, default='./data/processed/polygon_triangles_normalized.pt')
    parser.add_argument('--save_dir', type=str, default='./outputs/checkpoints/eval')
    parser.add_argument('--spatial_size', type=int, default=256, help="栅格化以及逆变换图的分辨率大小")
    
    parser.add_argument('--pos_freqs', type=int, default=31)
    parser.add_argument('--w_min', type=float, default=0.1)
    parser.add_argument('--w_max', type=float, default=100.0)
    parser.add_argument('--freq_type', type=str, default='geometric')
    parser.add_argument('--patch_size', type=int, default=2)
    parser.set_defaults(**config_defaults)
    
    args = parser.parse_args(remaining)
    main(args)
