# preprocessing/generate_data.py
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.path as mpltPath
import argparse
import sys
# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上退两级回到根目录 (preprocessing -> downstream_unet -> root)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.append(project_root)

from mae_pretrain.src.datasets.geometry_polygon import PolyFourierConverter
from config import FourierConfig as cfg

def rasterize_triangles(tris, spatial_size=256):
    """栅格化为二值清晰图 (Data A)"""
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
    parser.add_argument("--save_dir", type=str, default="./data/unet_dataset")
    return parser

def main():
    # ==========================================
    # 🌟 快速测试开关：True 只生成 100 个，False 生成全部
    TEST_MODE = False
    # ==========================================
    parser = build_arg_parser()
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("初始化傅里叶引擎...")
    engine = PolyFourierConverter(
        pos_freqs=cfg.POS_FREQS, w_min=cfg.W_MIN, w_max=cfg.W_MAX,
        freq_type=cfg.FREQ_TYPE, patch_size=cfg.PATCH_SIZE, device=device
    )
    
    all_polys = torch.load(args.data_path, weights_only=False)
    total_samples = 100 if TEST_MODE else len(all_polys)
    
    print(f"🚀 开始生成3通道输入数据, 目标数量: {total_samples}")
    
    for i in tqdm(range(total_samples)):
        save_path = os.path.join(args.save_dir, f"pair_{i}.pt")
        if os.path.exists(save_path): continue
        
        tris = all_polys[i]
        
        # 1. 制作 Label (干净的 0/1 图)
        y_np = rasterize_triangles(tris, cfg.SPATIAL_SIZE)
        y_tensor = torch.from_numpy(y_np).unsqueeze(0).half() #[1, 256, 256]
        
        # 2. 制作3通道 Input
        batch_tris = torch.tensor(tris, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([tris.shape[0]]).to(device)
        
        with torch.no_grad():
            # 获取解析频谱
            mag_log, phase = engine.cft_polygon_batch(batch_tris, lengths)
            
            # 逆变换出模糊空间图 (通道1)
            raw_mag = torch.expm1(mag_log)
            F_uv = raw_mag * torch.exp(1j * phase)
            x_spatial = engine.icft_2d(F_uv.squeeze(1), spatial_size=cfg.SPATIAL_SIZE)
            x_spatial = x_spatial.unsqueeze(1) #[1, 1, 256, 256]
            
            # 使用双线性插值, 将频域特征强制缩放到 256x256 (通道2, 通道3)
            mag_map = F.interpolate(mag_log, size=(cfg.SPATIAL_SIZE, cfg.SPATIAL_SIZE), mode='bilinear', align_corners=False)
            phase_map = F.interpolate(phase, size=(cfg.SPATIAL_SIZE, cfg.SPATIAL_SIZE), mode='bilinear', align_corners=False)
            
            # 拼接成 3 通道输入
            x_3channel = torch.cat([x_spatial, mag_map, phase_map], dim=1).squeeze(0).cpu().half() # [3, 256, 256]
        
        # 3. 存盘
        torch.save({'input': x_3channel, 'label': y_tensor}, save_path)

if __name__ == "__main__":
    main()