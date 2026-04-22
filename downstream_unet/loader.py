#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F
import json
import os
import bisect

# class UnetDataset(Dataset):
#     def __init__(self, index_file, data_dir=None):
        
#         if not os.path.exists(index_file):
#             if data_dir and os.path.exists(data_dir):
#                 print(f"⚠️ 未找到索引文件 {index_file}，正在从 {data_dir} 构建...")
#                 self._build_index(data_dir, index_file)
#             else:
#                 raise FileNotFoundError(f"找不到索引文件且未提供有效的 data_dir: {data_dir}")
            
#         with open(index_file, 'r') as f:
#             self.index = json.load(f)
#         # 计算累计长度用于快速定位
#         self.cumulative_lengths = [0]
#         for item in self.index:
#             self.cumulative_lengths.append(self.cumulative_lengths[-1] + item['num_samples'])
        
#         self.total_len = self.cumulative_lengths[-1]
#         self.current_shard_path = None
#         self.current_shard = None
        
#     def __len__(self):
#         return self.total_len
#     def __getitem__(self, idx):
#         # 1. 快速定位到分片索引 (使用二分查找效率更高)
#         shard_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
#         shard_info = self.index[shard_idx]
#         local_idx = idx - self.cumulative_lengths[shard_idx]
        
#         # 2. 懒加载：仅在必要时加载
#         if self.current_shard_path != shard_info['path']:
#             self.current_shard_path = shard_info['path']
#             self.current_shard = torch.load(self.current_shard_path, weights_only=False)
        
#         item = self.current_shard[local_idx]
#         return (torch.as_tensor(item['icft'], dtype=torch.float32).unsqueeze(0),
#                 torch.as_tensor(item['rec_label'], dtype=torch.float32).unsqueeze(0))
    
#     def _build_index(self, data_dir, output_file="dataset_index.json"):
#         shards = sorted(Path(data_dir).glob("tri_forward*.pt"))
#         index = []
        
#         print("正在扫描数据集，构建索引...")
#         for shard in shards:
#             # 这里只读文件属性或使用一种轻量级方法统计样本数
#             # 如果 pt 是 torch.save 存的列表，可能无法避免加载，但我们可以只加载一次
#             data = torch.load(shard, weights_only=False)
#             index.append({
#                 'path': str(shard),
#                 'num_samples': len(data)
#             })
        
#         with open(output_file, 'w') as f:
#             json.dump(index, f)
#         print(f"✅ 索引已保存至 {output_file}")
