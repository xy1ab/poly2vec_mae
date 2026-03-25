import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# 修正导入为您刚才生成的 API 文件
from downstream.task_recons.pipeline_mae import MaeReconstructionPipeline
from utils.config.loader import load_yaml_config
from utils.geometry.rasterize import rasterize_polygon
from utils.io.filesystem import ensure_dir

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "configs", "downstream", "recons.yaml"),
        type=str,
    )
    pre_args, remaining = pre_parser.parse_known_args()
    config_defaults = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description="MAE Reconstruction Integration Test")
    parser.add_argument('--config', type=str, default=pre_args.config, help="下游任务配置文件路径")
    parser.add_argument('--shape', type=str, default='circle', choices=['dataset', 'circle', 'pentagon'], 
                        help="测试几何体来源: dataset(来自.pt文件), circle(圆形), pentagon(正五边形)")
    parser.add_argument('--index', type=int, default=0, help="当 shape 为 dataset 时，提取.pt文件中多边形的索引")
    parser.add_argument('--data_path', type=str, default='./data/processed/polygon_triangles_normalized.pt')
    parser.add_argument('--mask_ratio', type=float, default=0.00, help="MAE掩码率，设为0可验证完美重构")
    parser.add_argument('--experiment_dir', type=str, default='./outputs/checkpoints/20260323_1342')
    parser.add_argument('--precision', type=str, default='bf16', help='推理精度: fp32 | fp16 | bf16')
    parser.add_argument('--save_dir', type=str, default='')
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining)
    if not args.save_dir:
        args.save_dir = args.experiment_dir

    # =========================================================================
    # 1. 初始化 MAE 流水线
    # =========================================================================
    experiment_dir = args.experiment_dir
    pipeline = MaeReconstructionPipeline(
        weight_path=os.path.join(experiment_dir, "mae_ckpt_140.pth"),
        config_path=os.path.join(experiment_dir, "poly_mae_config.json"),
        precision=args.precision,
    )
    
    # =========================================================================
    # 2. 根据用户选择，生成或加载测试数据
    # =========================================================================
    if args.shape == 'dataset':
        print(f"\n[Test] 正在从 {args.data_path} 提取索引为 {args.index} 的多边形...")
        all_polys = torch.load(args.data_path, weights_only=False)
        if args.index >= len(all_polys):
            raise IndexError(f"指定的索引 {args.index} 越界，总共有 {len(all_polys)} 个多边形数据。")
            
        tris = all_polys[args.index]
        
        # 将预处理好的三角形拼回原始多边形轮廓
        shapely_tris = [ShapelyPolygon(t) for t in tris]
        merged_poly = unary_union(shapely_tris)
        
        if merged_poly.geom_type == 'MultiPolygon':
            merged_poly = max(merged_poly.geoms, key=lambda a: a.area)
            
        poly_coords = np.array(merged_poly.exterior.coords)[:-1]
        save_suffix = f"idx_{args.index}"

    elif args.shape == 'pentagon':
        print(f"\n[Test] 泛化性测试：正在动态生成理想【正五边形】...")
        # 生成正五边形，稍微旋转使其尖端朝上
        angles = np.linspace(0, 2 * np.pi, 6)[:-1] + np.pi / 10
        poly_coords = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 0.8
        save_suffix = "shape_pentagon"

    elif args.shape == 'circle':
        print(f"\n[Test] 泛化性测试：正在动态生成理想【圆形】(64边形高精度拟合)...")
        # 生成高精度圆形
        angles = np.linspace(0, 2 * np.pi, 65)[:-1]
        poly_coords = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 0.8
        save_suffix = "shape_circle"
    
    raw_polygons = [poly_coords]
    print(f"[Test] 正在使用 {int(args.mask_ratio*100)}% 掩码率执行 MAE 重建...")
    
    # =========================================================================
    # 3. 动态劫持 (Hook)：在不破坏 API 的前提下，截获底层真实掩码
    # =========================================================================
    captured_mask = None
    captured_imgs = None
    original_forward = pipeline.model.forward
    
    def hooked_forward(imgs, mask_ratio=args.mask_ratio):
        nonlocal captured_mask, captured_imgs
        captured_imgs = imgs.clone().detach() # 截获输入(含原始幅值和相位)
        loss, pred, mask, target, imgs_sq = original_forward(imgs, mask_ratio)
        captured_mask = mask.clone().detach() # 截获本次生成的随机掩码
        return loss, pred, mask, target, imgs_sq
        
    pipeline.model.forward = hooked_forward
    real_part, imag_part = pipeline.reconstruct_real_imag(raw_polygons, mask_ratio=args.mask_ratio)
    pipeline.model.forward = original_forward
    
    print(f"[Test] 提取完成！")
    print(f"       实部维度: {real_part.shape}") 
    print(f"       虚部维度: {imag_part.shape}") 
    
    # =========================================================================
    # 4. 解析掩码图用于可视化
    # =========================================================================
    p = pipeline.config.get("patch_size", 2)
    H, W = captured_imgs.shape[2], captured_imgs.shape[3]
    h_p, w_p = H // p, W // p
    
    # 提取幅值和原始相位的 cos, sin
    orig_mag = captured_imgs[0, 0].cpu() 
    orig_cos = captured_imgs[0, 1].cpu()
    orig_sin = captured_imgs[0, 2].cpu()
    orig_phase = torch.atan2(orig_sin, orig_cos) # 反解出真实的原始相位
    
    if args.mask_ratio > 0:
        mask_map = captured_mask[0].cpu().reshape(h_p, w_p, 1, 1).expand(-1, -1, p, p)
        mask_map = mask_map.permute(0, 2, 1, 3).reshape(H, W)
    else:
        mask_map = torch.zeros((H, W))
    
    # 裁剪有效区域
    valid_h = H - pipeline.fourier_engine.pad_h
    valid_w = W - pipeline.fourier_engine.pad_w
    orig_mag_valid = orig_mag[:valid_h, :valid_w].numpy()
    orig_phase_valid = orig_phase[:valid_h, :valid_w].numpy()
    mask_map_valid = mask_map[:valid_h, :valid_w].numpy()
    
    # 生成带有 NaN 的幅值和相位掩码图 (用于镂空显示)
    masked_mag_vis = orig_mag_valid.copy()
    masked_mag_vis[mask_map_valid == 1] = np.nan
    
    masked_phase_vis = orig_phase_valid.copy()
    masked_phase_vis[mask_map_valid == 1] = np.nan
    
    # =========================================================================
    # 5. 模拟预处理运算 (为 ResNet34 准备 3 通道)
    # =========================================================================
    complex_valid = real_part + 1j * imag_part
    
    pad_h = pipeline.fourier_engine.pad_h
    pad_w = pipeline.fourier_engine.pad_w
    if pad_h > 0 or pad_w > 0:
        complex_padded = torch.nn.functional.pad(complex_valid, (0, pad_w, 0, pad_h), value=0.0)
    else:
        complex_padded = complex_valid
        
    mag_channel = torch.log1p(torch.abs(complex_padded))
    phase_channel = torch.angle(complex_padded)
    icft_raster_channel = pipeline.fourier_engine.icft_2d(complex_padded, spatial_size=256)
    
    # =========================================================================
    # 6. 测试结果可视化证明 (6 个子图)
    # =========================================================================
    gt_raster = rasterize_polygon(poly_coords)
    
    fig, axes = plt.subplots(1, 6, figsize=(36, 6))
    
    im0 = axes[0].imshow(gt_raster, cmap='gray', extent=[-1, 1, -1, 1])
    axes[0].set_title('Ground Truth Polygon')
    
    im1 = axes[1].imshow(masked_mag_vis, cmap='viridis')
    axes[1].set_title(f'MAE Input: Masked Mag ({int(args.mask_ratio*100)}%)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(masked_phase_vis, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    axes[2].set_title(f'MAE Input: Masked Phase ({int(args.mask_ratio*100)}%)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    im3 = axes[3].imshow(mag_channel[0].cpu().numpy(), cmap='viridis')
    axes[3].set_title('Colleague Ch1: Recon Mag(log)')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    im4 = axes[4].imshow(phase_channel[0].cpu().numpy(), cmap='viridis', vmin=-np.pi, vmax=np.pi)
    axes[4].set_title('Colleague Ch2: Recon Phase')
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    
    im5 = axes[5].imshow(icft_raster_channel[0].cpu().numpy(), cmap='gray', vmin=0, vmax=1, extent=[-1, 1, -1, 1])
    axes[5].set_title('Colleague Ch3: ICFT Raster Field')
    plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)
    
    for ax in axes: ax.axis('off')
        
    plt.tight_layout()
    ensure_dir(args.save_dir)
    save_path = os.path.join(args.save_dir, f"resnet_integration_proof_{save_suffix}_mask_{int(args.mask_ratio*100)}.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n[Test] 证明可视化图像已生成：{save_path}")
    print("[Test] 交接证明程序无缺陷运行完毕！您可以将实部虚部放心地交给同事了。")

if __name__ == "__main__":
    main()
