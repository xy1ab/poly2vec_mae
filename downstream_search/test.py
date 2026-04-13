import torch_musa
import torch
import time
import pandas as pd

# --- 被测函数定义 ---

def musa_knn_search(query, library, k=5):
    # L2距离：||a-b||^2 = ||a||^2 + ||b||^2 - 2ab'
    q_norm = torch.sum(query**2, dim=1, keepdim=True)
    l_norm = torch.sum(library**2, dim=1, keepdim=True)
    dists = q_norm + l_norm.t() - 2 * torch.matmul(query, library.t())
    topk_values, topk_indices = torch.topk(dists, k, largest=False)
    return topk_values, topk_indices

def musa_spatial_filter(queries, tree_bboxes):
    # 空间相交判断 (Overlap Check)
    # queries/tree_bboxes: [N, 4] -> (minx, miny, maxx, maxy)
    res_mask = (queries[:, 0:1] < tree_bboxes[:, 2]) & \
               (queries[:, 2:3] > tree_bboxes[:, 0]) & \
               (queries[:, 1:2] < tree_bboxes[:, 3]) & \
               (queries[:, 3:4] > tree_bboxes[:, 1])
    return res_mask

# --- 测试框架 ---

def benchmark():
    results = []
    # 测试规模：1万, 5万, 10万条数据
    sizes = [10000, 50000, 100000]
    dim = 256  # 向量维度
    k = 10
    
    print(f"{'Task':<20} | {'Size':<10} | {'Device':<10} | {'Time (ms)':<10}")
    print("-" * 60)

    for n in sizes:
        # 1. 数据准备
        # KNN 数据
        xq = torch.randn(100, dim)
        xb = torch.randn(n, dim)
        # Spatial 数据 (BBox: minx, miny, maxx, maxy)
        sq = torch.rand(100, 4) * 100
        sb = torch.rand(n, 4) * 100

        for device_name in ["cpu", "musa"]:
            device = torch.device(device_name)
            
            # 将数据移动到设备
            xq_d, xb_d = xq.to(device), xb.to(device)
            sq_d, sb_d = sq.to(device), sb.to(device)

            # --- 测试 KNN ---
            # 预热 (Warm up)
            _ = musa_knn_search(xq_d, xb_d, k)
            torch.musa.synchronize() if device_name == "musa" else None
            
            start = time.perf_counter()
            for _ in range(10): # 运行10次取平均
                _ = musa_knn_search(xq_d, xb_d, k)
            if device_name == "musa": torch.musa.synchronize()
            knn_time = (time.perf_counter() - start) * 1000 / 10
            
            # --- 测试 Spatial Filter ---
            _ = musa_spatial_filter(sq_d, sb_d)
            torch.musa.synchronize() if device_name == "musa" else None

            start = time.perf_counter()
            for _ in range(10):
                _ = musa_spatial_filter(sq_d, sb_d)
            if device_name == "musa": torch.musa.synchronize()
            spatial_time = (time.perf_counter() - start) * 1000 / 10

            print(f"{'KNN Search':<20} | {n:<10} | {device_name:<10} | {knn_time:.2f}")
            print(f"{'Spatial Filter':<20} | {n:<10} | {device_name:<10} | {spatial_time:.2f}")
            
            results.append({"task": "KNN", "size": n, "device": device_name, "ms": knn_time})
            results.append({"task": "Spatial", "size": n, "device": device_name, "ms": spatial_time})

    return results

if __name__ == "__main__":
    if not torch.musa.is_available():
        print("错误：未检测到 MUSA 设备，请检查驱动和 torch_musa。")
    else:
        res = benchmark()
        # 这里可以进一步计算加速比 (CPU_ms / MUSA_ms)