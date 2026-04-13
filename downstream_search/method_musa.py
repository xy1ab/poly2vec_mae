#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   method.py
@Time    :   2026/04/13 10:59:42
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
import torch_musa
import torch
import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os
from tqdm import tqdm


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

def run_spatial_filter_ddp(args, small_queries, loader):
    local_hit_indices = []
    pbar = None
    if args.global_rank == 0:
        pbar = tqdm(total=len(loader), desc="Global Processing", unit="batch")
    print(f"[Rank {args.global_rank}] 开始处理...")

    with torch.inference_mode():
        for (b_batch,) in loader:
            b_batch = b_batch.to(args.device)
            # 调用你的空间过滤器方法
            cross_mask = musa_spatial_filter_kernel(small_queries, b_batch)
            hit_in_batch = cross_mask.any(dim=0) ## 相加找到总数
            # 结果回传 CPU 释放显存
            if hit_in_batch.any():
                local_hit_indices.append(b_batch[hit_in_batch].cpu())
            # 更新进度条
            if pbar:
                pbar.update(1)
    # 合并本地计算结果
    if local_hit_indices:
        local_result = torch.cat(local_hit_indices, dim=0)
    else:
        local_result = torch.empty((0, 4))

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(local_result, os.path.join(args.output_dir, f"output_rank_{args.global_rank}.pt"))

    dist.barrier()
    if args.global_rank == 0:
        print(f"所有 {args.world_size} 张卡处理完成，结果已保存在 {args.output_dir}")



if __name__ == '__main__':
    pass