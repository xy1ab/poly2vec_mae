import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from mae_pretrain.src.datasets.geometry_polygon import PolyFourierConverter

def rasterize_triangles(tris, spatial_size=256):
    """将一组三角形空间坐标光栅化为 0/1 的二值图掩码"""
    x = np.linspace(-1, 1, spatial_size)
    y = np.linspace(1, -1, spatial_size) # matplotlib 图像坐标 y轴向下
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.flatten(), Y.flatten())).T
    
    mask = np.zeros(spatial_size * spatial_size, dtype=bool)
    for tri in tris:
        # 对每个三角形判断每个像素点是否在内部
        path = mpltPath.Path(tri)
        mask |= path.contains_points(points)
        
    return mask.reshape((spatial_size, spatial_size)).astype(np.float32)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")
    
    # 1. 载入 .pt 文件数据
    print(f"载入数据文件: {args.data_path}")
    # 添加 weights_only=False 以允许加载包含 numpy array 的本地安全列表
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
    
    # 还原真实傅里叶幅值用于逆变换 (因为前向传播内部做了 log1p)
    raw_mag = torch.expm1(mag_log)
    F_uv = raw_mag * torch.exp(1j * phase)
    
    # 5. 执行 ICFT 进行空域重构
    recon_raster = engine.icft_2d(F_uv.squeeze(1), spatial_size=args.spatial_size)
    
    # 转移到CPU用于画图及误差计算
    mag_vis = mag_log.squeeze().cpu().numpy()
    phase_vis = phase.squeeze().cpu().numpy()
    recon_vis = recon_raster.squeeze().cpu().numpy()
    
    # 计算绝对误差图
    diff_vis = np.abs(orig_raster - recon_vis)
    
    # 6. 画图并保存 (修改为5个子图)
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    
    # 图1：原始多边形二值栅格图 (统一为 gray 和 0-1 的范围)
    im0 = axes[0].imshow(orig_raster, cmap='gray', vmin=0, vmax=1, extent=[-1, 1, -1, 1])
    axes[0].set_title('Original Rasterized Polygon')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 图2：CFT 幅值
    im1 = axes[1].imshow(mag_vis, cmap='viridis')
    axes[1].set_title('CFT Magnitude (log1p)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 图3：CFT 相位
    im2 = axes[2].imshow(phase_vis, cmap='hsv')
    axes[2].set_title('CFT Phase')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # 图4：ICFT 重建图 (统一为 gray 和 0-1 的范围)
    im3 = axes[3].imshow(recon_vis, cmap='gray', vmin=0, vmax=1, extent=[-1, 1, -1, 1])
    axes[3].set_title('ICFT Reconstructed Field')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # 图5：重建差值图 (使用红色系突出误差)
    im4 = axes[4].imshow(diff_vis, cmap='Reds', vmin=0, vmax=1, extent=[-1, 1, -1, 1])
    axes[4].set_title('Absolute Difference')
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'cft_visualize_idx_{args.index}.png')
    plt.savefig(save_path, dpi=150)
    print(f"可视化结果已保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CFT and ICFT Visualization")
    parser.add_argument('--index', type=int, default=0, help="提取.pt文件中多边形的索引")
    parser.add_argument('--data_path', type=str, default='./data/polygon_triangles_normalized.pt')
    parser.add_argument('--save_dir', type=str, default='./vis_results')
    parser.add_argument('--spatial_size', type=int, default=256, help="栅格化以及逆变换图的分辨率大小")
    
    # 引擎频率参数，需要与你抽特征时保持一致
    parser.add_argument('--pos_freqs', type=int, default=5)
    parser.add_argument('--w_min', type=float, default=0.1)
    parser.add_argument('--w_max', type=float, default=100.0)
    # parser.add_argument('--freq_type', type=str, default='geometric')
    parser.add_argument('--freq_type', type=str, default='geometric')
    parser.add_argument('--patch_size', type=int, default=16)
    
    args = parser.parse_args()
    main(args)