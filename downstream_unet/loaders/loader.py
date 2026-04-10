# loaders/loader.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
class V2Dataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_path: 单个 .pt 文件路径，包含所有样本
        """
        shards = sorted(Path(data_dir).glob("shard_*.pt"))
        all_samples = []
        for shard in shards:
            data = torch.load(shard)
            all_samples.extend(data)
        self.data = all_samples
        print(f"✅ 加载数据: {data_dir}, 共 {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample['input'].float(), sample['label'].float()