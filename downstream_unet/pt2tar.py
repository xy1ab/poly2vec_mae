#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pt2tar.py
@Time    :   2026/04/22 09:56:27
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import torch
import webdataset as wds
import glob
import os
from tqdm import tqdm
import numpy as np
import json
def convert_pt_to_tar(input_dir, output_dir, train_ratio=0.9, max_size_bytes=1 * 1024**3):
    """
    input_dir: 包含 .pt 文件的文件夹
    output_dir: .tar 文件输出目录
    max_size_bytes: 每个 .tar 包的大小上限 (默认 5GB)
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    
    # 定义输出格式，例如 shard-000000.tar
    train_sink = wds.ShardWriter(os.path.join(train_dir, "train-%06d.tar"), maxsize=max_size_bytes)
    val_sink = wds.ShardWriter(os.path.join(val_dir, "val-%06d.tar"), maxsize=max_size_bytes)
    
    # 初始化计数器
    train_count = 0
    val_count = 0
    total_samples = 0
    # 使用 WebDataset 的 ShardWriter 进行流式写入
    for pt_file in tqdm(pt_files, desc="Converting & Splitting"):
        data_list = torch.load(pt_file, map_location='cpu', weights_only=False)
        
        for i, sample in enumerate(data_list):
            key = f"{os.path.basename(pt_file).replace('.pt', '')}_{i:06d}"
            sample_dict = {
                "__key__": key,
                "input.npy": sample['icft'][np.newaxis, ...],
                "label.npy": sample['rec_label'][np.newaxis, ...]
            }
            
            # 使用确定性逻辑进行分配，确保一致性
            # 这里简单地根据总样本计数进行分配，也可以用 hash(key) % 100 < train_ratio*100
            if (total_samples % 100) < (train_ratio * 100):
                train_sink.write(sample_dict)
                train_count += 1
            else:
                val_sink.write(sample_dict)
                val_count += 1
            
            total_samples += 1
            
    train_sink.close()
    val_sink.close()
    print(f"转换完成，处理了 {total_samples} 个样本。")
    # 构造并保存 index_file.json
    index_data = {
        "train": {
            "num_samples": train_count,
            "path": os.path.abspath(train_dir)
        },
        "val": {
            "num_samples": val_count,
            "path": os.path.abspath(val_dir)
        }
    }
    index_path = os.path.join(output_dir, "index_file.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4)

    print(f"转换完成。")
    print(f"训练集样本数: {train_count}, 验证集样本数: {val_count}")
    print(f"索引文件已保存至: {index_path}")
    
if __name__ == '__main__':
    # 使用示例
    convert_pt_to_tar("/mnt/git-data/HB/poly2vec_mae/outputs/unet_traindataset", "/mnt/git-data/HB/poly2vec_mae/outputs/unet_traindataset")

