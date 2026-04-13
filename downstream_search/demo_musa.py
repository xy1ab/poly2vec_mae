#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   demo.py
@Time    :   2026/04/13 15:01:45
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

from method import run_spatial_filter_ddp,restore_bbox
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import argparse
import time
    

def setup_dist():
    """
    初始化分布式环境并返回核心配置参数
    """
    local_rank  = int(os.environ["LOCAL_RANK"])

    
    # 设置当前进程对应的 GPU 设备
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        # 大规模集群（128卡）通信握手可能较慢，timeout 设长一点
        dist.init_process_group(backend="nccl",device_id=device)
    global_rank = dist.get_rank()
    world_size  = dist.get_world_size() 
    return local_rank, global_rank, world_size, device

def get_library_init(data_path,batch_size):
    all_data = torch.load(data_path,weights_only=False)
    all_data_tensor = torch.stack([torch.from_numpy(a) for a in all_data])
    library_bboxes = restore_bbox(all_data_tensor)
    dataset = TensorDataset(library_bboxes)
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch forward export."""
    parser = argparse.ArgumentParser(
        description="Batch forward triangulated polygon shards into embeddings and MAE frequency maps."
    )
    parser.add_argument("--data_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/data/processed/osm/25_hangzhou_landuse_meta.pt", help="Directory containing triangulated shard `.pt` files.")
    parser.add_argument("--output_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/", help="Output `.pt` file path.")
    parser.add_argument("--batch_size", type=int, default=32768, help="Inference batch size.")    
    return parser

if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    args.local_rank, args.global_rank, args.world_size, args.device = setup_dist()
    
    if args.global_rank == 0:
        print(f"🚀 集群启动成功 | 总规模: {args.world_size} 卡 | 当前设备: {args.device}")

    loader = get_library_init(args.data_path, args.batch_size)

    # Spatial 数据 (BBox: minx, miny, maxx, maxy)
    ## test data
    M = 100
    data = torch.rand(M, 4).to(args.device)
    data[:, [0, 2]], _ = torch.sort(data[:, [0, 2]], dim=1)
    data[:, [1, 3]], _ = torch.sort(data[:, [1, 3]], dim=1)

    small_queries = data

    start = time.perf_counter()
    run_spatial_filter_ddp(args,small_queries, loader)
    spatial_time = (time.perf_counter() - start) * 1000 / 10
    print(f"{'Spatial Filter':<20} | {len(loader.dataset):<10} | {args.world_size:<10} | {spatial_time:.2f}")

    dist.destroy_process_group()
    


