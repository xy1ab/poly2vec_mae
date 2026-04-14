#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   method.py
@Time    :   2026/04/13 10:59:42
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os
from tqdm import tqdm
from pathlib import Path

# def ddp_knn_worker(rank, world_size, queries, library,batch_size, k=5):
#     setup(rank, world_size)
    
#     # 1. 准备数据：将 library 放在每张卡的显存中
#     device = torch.device(f"cuda:{rank}")
#     library = library.to(device)
#     l_norm = torch.sum(library**2, dim=1, keepdim=True)
    
#     # 2. 包装 queries 为分布式数据集
#     dataset = TensorDataset(queries)
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
#     loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
#     results_val = []
#     results_idx = []

#     # 3. 检索逻辑 (Inference Mode 提升性能)
#     with torch.inference_mode():
#         for (q_batch,) in loader:
#             q_batch = q_batch.to(device)
#             q_norm = torch.sum(q_batch**2, dim=1, keepdim=True)
            
#             # L2 距离矩阵运算
#             dists = q_norm + l_norm.t() - 2 * torch.matmul(q_batch, library.t())
#             topk_values, topk_indices = torch.topk(dists, k, largest=False)
            
#             results_val.append(topk_values)
#             results_idx.append(topk_indices)

#     # 4. 汇总所有显卡的结果 (Optional)
#     # 在生产环境下，通常每张卡直接将结果写入磁盘或数据库，避免主卡内存溢出
#     # 如果需要汇总：
#     all_vals = [torch.zeros_like(torch.cat(results_val)) for _ in range(world_size)]
#     dist.all_gather(all_vals, torch.cat(results_val))
    
#     cleanup()

@torch.compile
def musa_spatial_filter_kernel(queries, library_bboxes):
    
    q = queries.unsqueeze(1)
    lib = library_bboxes.unsqueeze(0)
    return (q[..., 0] < lib[..., 2]) & \
           (q[..., 2] > lib[..., 0]) & \
           (q[..., 1] < lib[..., 3]) & \
           (q[..., 3] > lib[..., 1])




def restore_bbox(center_data):
    # center_data 形状假设为 [N, 6]，每一行是 [x, y, dx, dy]
    x = center_data[:, 0]
    y = center_data[:, 1]
    dx = center_data[:, 3]
    dy = center_data[:, 4]
    
    minx = x - dx / 2
    maxx = x + dx / 2
    miny = y - dy / 2
    maxy = y + dy / 2
    
    # 堆叠成 [N, 4] 的 BBox 格式
    return torch.stack([minx, miny, maxx, maxy], dim=1)

def run_spatial_filter_ddp(args, small_queries, loader, real_data_len, count=True):
    local_hit_indices = []
    pbar = None
    if args.global_rank == 0:
        pbar = tqdm(total=real_data_len, desc="Global Processing", unit="sample")
    print(f"[Rank {args.global_rank}] 开始处理...")

    base_num = real_data_len // args.world_size
    extras = real_data_len % args.world_size
    # 每个 rank 真正应该负责的数据量
    # 前 extras 个 rank 多拿一个，后面的拿 base_num 个
    legit_count = base_num + (1 if args.global_rank < extras else 0)
    processed_in_rank = 0 # 当前进程已处理的有效数据计数
    num_queries = small_queries.size(0)
    global_query_counts = torch.zeros(num_queries, dtype=torch.long, device=args.device)
    query_chunk_size = 100000
    with torch.inference_mode():
        for (b_batch,) in loader:
            current_batch_size = b_batch.size(0)
            if processed_in_rank >= legit_count:
                break # 这个 Batch 里的全是补齐给我的，但我不需要了
            if processed_in_rank + current_batch_size > legit_count:
                # 这个 batch 跨界了，截取前半部分
                actual_needed = legit_count - processed_in_rank
            else:
                actual_needed = current_batch_size
            b_batch = b_batch[:actual_needed].to(args.device, non_blocking=True)
            for start_idx in range(0, num_queries, query_chunk_size):
                end_idx = min(start_idx + query_chunk_size, num_queries)
                
                # 提取当前块的 Query: [chunk_size, 4]
                query_chunk = small_queries[start_idx:end_idx]
                
                # 计算掩码: [chunk_size, 8068]
                # 此时显存占用从 7.5GB 降到了 0.75GB 左右
                cross_mask = musa_spatial_filter_kernel(query_chunk, b_batch)
                
                # 累加命中数
                global_query_counts[start_idx:end_idx] += cross_mask.sum(dim=1)
                
                # 强制释放临时中间变量（可选，由 Python GC 处理）
                del cross_mask


            
            # # 调用你的空间过滤器方法
            # cross_mask = musa_spatial_filter_kernel(small_queries, b_batch)
            # # hit_in_batch = cross_mask.any(dim=1) ## 相加找到总数
            # # hit_in_batch = cross_mask.sum().item()
            # global_query_counts += cross_mask.sum(dim=1)
            # if hit_in_batch.any():
            #     hits = b_batch[hit_in_batch].cpu()
            #     local_hit_indices.append(hits)
            processed_in_rank += actual_needed
            # 更新进度条
            if args.global_rank == 0 and pbar is not None:
                pbar.update(actual_needed * args.world_size)
    if args.global_rank == 0: pbar.close()
    # 合并本地计算结果
    # if local_hit_indices:
    #     local_result = torch.cat(local_hit_indices, dim=0)
    # else:
    #     local_result = torch.empty((0, 4))
    if count:
        # local_count = torch.tensor([local_result.shape[0]], device=args.device)
        dist.all_reduce(global_query_counts, op=dist.ReduceOp.SUM)
        # global_total = global_query_counts.item()
        if args.global_rank == 0:
            print(f"📊 汇总统计完成:")
            print(f"  - 全局命中总数: {global_query_counts.sum()}")
    # else:
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     torch.save(local_result, os.path.join(args.output_dir, f"output_rank_{args.global_rank}.pt"))

    #     dist.barrier()
    #     if args.global_rank == 0:
    #         print(f"所有 {args.world_size} 张卡处理完成，结果已保存在 {args.output_dir}")



if __name__ == '__main__':
    pass