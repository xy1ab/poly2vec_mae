import torch
from torch.utils.data import Dataset
import numpy as np
import random
import math

class PolyMAEDataset(Dataset):
    """
    负责加载变长多边形三角形列表，并支持实时数据增强。
    """
    # 【改动点】__init__ 不再接收路径，而是直接接收切分好的实体 data_list
    def __init__(self, data_list, geom_type='polygon', augment_times=1):
        super().__init__()
        self.data_list = data_list # List of numpy arrays [N, 3, 2]
        self.geom_type = geom_type
        self.augment_times = augment_times
        self.total_len = len(self.data_list) * self.augment_times

    def __len__(self):
        return self.total_len

    def apply_augmentation(self, tris):
        """对坐标系 [-1, 1] 应用增强：旋转、缩放、平移(抖动)"""
        # 旋转
        angle = random.uniform(0, 2 * math.pi)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # 应用旋转
        tris_shape = tris.shape
        tris_flat = tris.reshape(-1, 2).dot(rot_matrix)
        
        # 获取当前包围盒
        min_c = tris_flat.min(axis=0)
        max_c = tris_flat.max(axis=0)
        
        # 缩放 (确保不越界)，范围 0.5 到 1.0
        scale = random.uniform(0.5, 1.0)
        tris_flat = tris_flat * scale
        min_c *= scale
        max_c *= scale
        
        # 抖动 (确保缩放加平移后在 [-1, 1] 内)
        max_tx = 1.0 - max_c[0]
        min_tx = -1.0 - min_c[0]
        max_ty = 1.0 - max_c[1]
        min_ty = -1.0 - min_c[1]
        
        tx = random.uniform(min_tx, max_tx) if max_tx >= min_tx else 0
        ty = random.uniform(min_ty, max_ty) if max_ty >= min_ty else 0
        
        tris_flat += np.array([tx, ty])
        
        return tris_flat.reshape(tris_shape).astype(np.float32)

    def __getitem__(self, idx):
        real_idx = idx % len(self.data_list)
        tris = self.data_list[real_idx] # [N, 3, 2]
        
        # idx > 0 说明是被扩增出来的副本
        if idx >= len(self.data_list):
            tris = self.apply_augmentation(tris)
            
        return torch.tensor(tris, dtype=torch.float32)

def mae_collate_fn(batch):
    """
    处理变长三角形数组：用 0 填充到当前 batch 的最大三角形数 (max_N)。
    返回 padding 后的 Tensor 和 真实长度的 Tensor。
    """
    lengths = torch.tensor([item.shape[0] for item in batch])
    max_len = lengths.max().item()
    
    padded_batch = torch.zeros((len(batch), max_len, 3, 2), dtype=torch.float32)
    for i, item in enumerate(batch):
        padded_batch[i, :lengths[i], :, :] = item
        
    return padded_batch, lengths