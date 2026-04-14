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
    real_data_len = len(library_bboxes)
    dataset = TensorDataset(library_bboxes)
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,num_workers=0)
    return loader, real_data_len

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

    loader,real_data_len = get_library_init(args.data_path, args.batch_size)

    # Spatial 数据 (BBox: minx, miny, maxx, maxy)
    ## test data
    dx = torch.linspace(118.02257356500002, 122.8342026900001, 1001)
    dy = torch.linspace(27.143422685000075, 31.182556060000024, 1001)

    # 2. 提取左右边界和上下边界
    x_left = dx[:-1]
    x_right = dx[1:]
    y_bottom = dy[:-1]
    y_top = dy[1:]

    # 3. 生成网格坐标矩阵
    # indexing='ij' 表示第一个维度对应 x，第二个维度对应 y
    X_min, Y_min = torch.meshgrid(x_left, y_bottom, indexing='ij')
    X_max, Y_max = torch.meshgrid(x_right, y_top, indexing='ij')

    # 4. 展平并合并为 [1000000, 4]
    # 使用 reshape(-1) 或 flatten() 展开，stack 进行维度堆叠
    bboxes = torch.stack([
        X_min.reshape(-1), 
        Y_min.reshape(-1), 
        X_max.reshape(-1), 
        Y_max.reshape(-1)
    ], dim=1)
    if args.global_rank == 0:
        print(f"BBox 数组形状: {bboxes.shape}")
    small_queries = bboxes.to(args.device,non_blocking=True)


    ## warm-up
    run_spatial_filter_ddp(args, small_queries[:100], loader, real_data_len=real_data_len, count=False)
    torch.cuda.synchronize()
    if args.global_rank == 0:
        print("warm-up over")
    ## 
    start = time.perf_counter()
    run_spatial_filter_ddp(args,small_queries, loader,real_data_len = real_data_len)
    spatial_time = (time.perf_counter() - start) * 1000 / 10
    dist.barrier()
    if args.global_rank == 0:
        print(f"{'Spatial Filter':<20} | {len(loader.dataset):<10} | {args.world_size:<10} | {spatial_time:.2f}")
    dist.destroy_process_group()
    import sys
    sys.exit(0)
    


