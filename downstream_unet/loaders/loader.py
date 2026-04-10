# loaders/loader.py
import torch
from torch.utils.data import Dataset

class V2Dataset(Dataset):
    def __init__(self, data_path):
        """
        Args:
            data_path: 单个 .pt 文件路径，包含所有样本
        """
        self.data = torch.load(data_path, map_location='cpu')
        print(f"✅ 加载数据: {data_path}, 共 {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample['input'].float(), sample['label'].float()