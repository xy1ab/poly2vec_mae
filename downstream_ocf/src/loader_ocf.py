import torch
from torch.utils.data import Dataset
import numpy as np
from matplotlib.path import Path

class OCFDataset(Dataset):
    def __init__(self, data_path, num_points=1024, boundary_ratio=0.7, jitter_std=0.01):
        print(f"📦 正在载入数据: {data_path}")
        # weights_only=False 是必须的，map_location='cpu' 节省显存
        self.data = torch.load(data_path, weights_only=False, map_location='cpu')
        self.num_points = num_points
        self.boundary_ratio = boundary_ratio
        self.jitter_std = jitter_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        v = sample['embedding']
        if isinstance(v, np.ndarray): v = torch.from_numpy(v)
            
        tris_raw = sample['triangles'] # 原始大坐标
        meta = sample['meta']          # [x, y, L, N]
        
        # --- 执行归一化 ---
        cx, cy, s = meta[0], meta[1], meta[2]
        
        # 转换成 numpy 方便计算
        if torch.is_tensor(tris_raw): tris_raw = tris_raw.numpy()
        if torch.is_tensor(meta): meta = meta.numpy()

        # 对三角形坐标进行中心化和比例缩放
        # 公式：(原始坐标 - 中心点) / 边长
        # 这样多边形就会分布在 [-1, 1] 附近的中心区域
        tris_norm = (tris_raw - np.array([cx, cy])) / (s + 1e-9)
        
        # --- 智能采样 (在归一化后的空间进行) ---
        n_boundary = int(self.num_points * self.boundary_ratio)
        n_uniform = self.num_points - n_boundary
        
        verts = tris_norm.reshape(-1, 2)
        
        # 1. 边界采样：在放大的多边形边缘撒点
        idx_v = np.random.choice(len(verts), n_boundary)
        p_boundary = verts[idx_v] + np.random.normal(0, self.jitter_std, (n_boundary, 2))
        
        # 2. 全局均匀采样：在整个 [-1, 1] 空间撒点
        p_uniform = np.random.uniform(-1, 1, (n_uniform, 2))
        
        p_combined = np.concatenate([p_boundary, p_uniform], axis=0).astype(np.float32)
        p_combined = np.clip(p_combined, -1.0, 1.0) 

        # --- 3. 标签判定 ---
        # 现在采样点和三角形都在同一个“放大后”的坐标系了
        labels = np.zeros(self.num_points, dtype=bool)
        for tri in tris_norm:
            labels |= Path(tri).contains_points(p_combined)

        return v.float(), torch.from_numpy(p_combined), torch.from_numpy(labels.astype(np.float32)).unsqueeze(-1)