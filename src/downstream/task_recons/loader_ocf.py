import torch
from torch.utils.data import Dataset
import numpy as np
from matplotlib.path import Path

class OCFDataset(Dataset):
    def __init__(self, data_path, num_points=1024, boundary_ratio=0.7):
        print(f"正在载入30万级全量数据: {data_path}")
        self.data = torch.load(data_path, weights_only=False)
        self.num_points = num_points
        self.boundary_ratio = boundary_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 1. 提取 384维 Embedding
        v = sample['embedding'] 
        if isinstance(v, np.ndarray): v = torch.from_numpy(v)
            
        # 2. 提取三角形数据用于空间采样
        tris = sample['triangles'] 
        
        # 采样
        n_boundary = int(self.num_points * self.boundary_ratio)
        n_uniform = self.num_points - n_boundary
        verts = tris.reshape(-1, 2)
        
        # 边界点
        idx_v = np.random.choice(len(verts), n_boundary)
        p_boundary = verts[idx_v] + np.random.normal(0, 0.005, (n_boundary, 2))
        # 全局点
        p_uniform = np.random.uniform(-1, 1, (n_uniform, 2))
        p_combined = np.concatenate([p_boundary, p_uniform], axis=0).astype(np.float32)

        # 3. 计算 0/1 标签
        path = Path(verts) 
        labels = path.contains_points(p_combined).astype(np.float32)

        return v, torch.from_numpy(p_combined), torch.from_numpy(labels).unsqueeze(-1)