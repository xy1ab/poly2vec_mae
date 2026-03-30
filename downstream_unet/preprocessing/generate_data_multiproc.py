import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.path as mpltPath
import argparse
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_polygon import PolyFourierConverter
from config import FourierConfig as cfg

# --- 核心改动 1: 将单次任务封装成独立函数 ---
def process_single_sample(i, tris, save_dir, engine):
    """处理单个样本的函数，供多进程调用"""
    save_path = os.path.join(save_dir, f"pair_{i}.pt")
    if os.path.exists(save_path):
        return
    
    # 1. 制作 Label (CPU 计算)
    y_np = rasterize_triangles(tris, cfg.SPATIAL_SIZE)
    y_tensor = torch.from_numpy(y_np).unsqueeze(0).half()

    # 2. 制作 3 通道 Input
    # 确保在 CPU 上运算以避免多进程 CUDA 初始化冲突
    batch_tris = torch.tensor(tris, dtype=torch.float32).unsqueeze(0)
    lengths = torch.tensor([tris.shape[0]])

    with torch.no_grad():
        # 获取解析频谱
        mag_log, phase = engine.cft_polygon_batch(batch_tris, lengths)
        
        # 逆变换出模糊空间图 (通道1)
        raw_mag = torch.expm1(mag_log)
        F_uv = raw_mag * torch.exp(1j * phase)
        x_spatial = engine.icft_2d(F_uv.squeeze(1), spatial_size=cfg.SPATIAL_SIZE)
        x_spatial = x_spatial.unsqueeze(1)
        
        # 插值缩放
        mag_map = F.interpolate(mag_log, size=(cfg.SPATIAL_SIZE, cfg.SPATIAL_SIZE), mode='bilinear', align_corners=False)
        phase_map = F.interpolate(phase, size=(cfg.SPATIAL_SIZE, cfg.SPATIAL_SIZE), mode='bilinear', align_corners=False)
        
        # 拼接并转为半精度以节省空间
        x_3channel = torch.cat([x_spatial, mag_map, phase_map], dim=1).squeeze(0).half()

    # 3. 存盘
    torch.save({'input': x_3channel, 'label': y_tensor}, save_path)

def rasterize_triangles(tris, spatial_size=256):
    x = np.linspace(-1, 1, spatial_size)
    y = np.linspace(1, -1, spatial_size)
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.flatten(), Y.flatten())).T
    mask = np.zeros(spatial_size * spatial_size, dtype=bool)
    for tri in tris:
        path = mpltPath.Path(tri)
        mask |= path.contains_points(points)
    return mask.reshape((spatial_size, spatial_size)).astype(np.float32)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="downstream_UNet")
    parser.add_argument("--data_path", type=str, default="./data/processed/polygon_triangles_normalized.pt")
    parser.add_argument("--save_dir", type=str, default="./data/unet_dataset2")
    parser.add_argument("--num_workers", type=int, default=cpu_count()-2, help="使用的 CPU 核心数")
    return parser

def main():
    TEST_MODE = False
    parser = build_arg_parser()
    args = parser.parse_args()

    # --- 核心改动 2: 并行时强制使用 CPU ---
    # 在大规模并行时，CPU 的多核总吞吐量往往高于单个 GPU 的数据交换开销
    device = torch.device('cpu') 
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"初始化傅里叶引擎 (CPU Mode)... 并行核心数: {args.num_workers}")
    engine = PolyFourierConverter(
        pos_freqs=cfg.POS_FREQS, w_min=cfg.W_MIN, w_max=cfg.W_MAX,
        freq_type=cfg.FREQ_TYPE, patch_size=cfg.PATCH_SIZE, device=device
    )
    
    all_polys = torch.load(args.data_path, weights_only=False)
    total_samples = 100 if TEST_MODE else len(all_polys)
    
    # 准备索引列表
    indices = list(range(total_samples))
    
    # --- 核心改动 3: 使用 Pool 进行并行分发 ---
    # 使用 partial 固定不需要变的参数
    worker_func = partial(
        process_single_sample, 
        save_dir=args.save_dir, 
        engine=engine
    )

    print(f"🚀 开始多核并行生成数据, 目标数量: {total_samples}")
    
    # 使用 tqdm 显示并行进度
    with Pool(processes=args.num_workers) as pool:
        # 使用 imap_unordered 效率最高
        list(tqdm(pool.imap_unordered(worker_wrapper, [(i, all_polys[i], worker_func) for i in indices]), total=total_samples))

# 为了解决 Pool 无法直接传递复杂对象的问题，写一个包装器
def worker_wrapper(args_tuple):
    i, tris, func = args_tuple
    return func(i, tris)

if __name__ == "__main__":
    main()