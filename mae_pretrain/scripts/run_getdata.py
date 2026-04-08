#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_reconstruction_musa.py
@Time    :   2026/04/07 14:34:03
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import torch
import argparse
from pathlib import Path

from tqdm import tqdm
from datetime import datetime
import sys
import re
from typing import Any

import numpy as np
import matplotlib.path as mpltPath

import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)
from mae_pretrain.src.datasets.geometry_polygon import pad_triangle_batch
from mae_pretrain.src.datasets.registry import get_geometry_codec
from mae_pretrain.src.datasets.shard_io import load_triangle_shard, resolve_triangle_shard_paths
from mae_pretrain.src.datasets.sharded_pt_dataset import _ensure_numpy_float32
from mae_pretrain.src.models.factory import load_mae_model, load_pretrained_encoder
from mae_pretrain.src.models.decoder import TransUNetdecoder
from mae_pretrain.src.utils.config import load_config_any
from mae_pretrain.src.utils.filesystem import ensure_dir
from mae_pretrain.src.utils.precision import autocast_context, normalize_precision


def rasterize_triangles_pytorch(batch_tris, spatial_size=256):
    """
    针对 Batch 优化并排除退化三角形的栅格化函数
    """
    B, N, _, _ = batch_tris.shape
    device = batch_tris.device
    H = W = spatial_size

    # 生成网格坐标 (B, H, W, 2)
    y = torch.linspace(1, -1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    p = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, H * W, 2) 

    # 准备结果容器 (B, H*W)
    final_mask = torch.zeros((B, H * W), device=device, dtype=torch.bool)

    # 顶点提取 (B, N, 2)
    v0 = batch_tris[:, :, 0, :].unsqueeze(2) # (B, N, 1, 2)
    v1 = batch_tris[:, :, 1, :].unsqueeze(2)
    v2 = batch_tris[:, :, 2, :].unsqueeze(2)

    def edge_func(va, vb, vp):
        # (x2-x1)*(yP-y1) - (y2-y1)*(xP-x1)
        return (vb[..., 0] - va[..., 0]) * (vp[..., 1] - va[..., 1]) - \
               (vb[..., 1] - va[..., 1]) * (vp[..., 0] - va[..., 0])

    # 为了节省显存，可以分批处理 N 个三角形
    chunk_size = 100 
    for i in range(0, N, chunk_size):
        v0_c = v0[:, i:i+chunk_size]
        v1_c = v1[:, i:i+chunk_size]
        v2_c = v2[:, i:i+chunk_size]

        # 1. 计算边函数
        w0 = edge_func(v0_c, v1_c, p)
        w1 = edge_func(v1_c, v2_c, p)
        w2 = edge_func(v2_c, v0_c, p)

        # 2. 排除退化三角形（面积为 0 的三角形）
        # 面积近似于边函数之和。如果三个顶点重合，w 全为 0
        # 增加一个 epsilon 避免浮点误差导致的虚假覆盖
        eps = 1e-6
        is_inside = ((w0 > eps) & (w1 > eps) & (w2 > eps)) | \
                    ((w0 < -eps) & (w1 < -eps) & (w2 < -eps))

        # 3. 实时合并到结果中
        final_mask |= is_inside.any(dim=1)

    return final_mask.view(B, H, W).float()

def _default_device() -> str:
    """Resolve default runtime device string."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _extract_last_int(text: str) -> int | None:
    """Extract the last integer substring from text, if present."""
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])

def _resolve_user_path(path_str: str, project_root: Path) -> Path:
    """Resolve one user-provided path against cwd and project root."""
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (project_root / raw_path).resolve()

def _inject_repo_root() -> Path:
    """Inject repository root into `sys.path` for direct script execution.

    Returns:
        `mae_pretrain` project root path.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root

def _build_output_path(project_root: Path, data_dir: str, output_path: str | None) -> Path:
    """Resolve output path from explicit CLI path or one timestamped default."""
    if output_path:
        return Path(output_path).expanduser().resolve()

    data_name = Path(data_dir).expanduser().resolve().name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "outputs" / "forward_batch" / f"{data_name}_forward_batch_{timestamp}.pt"

def _resolve_model_artifacts(model_dir: str) -> tuple[Path, Path, Path, bool]:
    """Resolve encoder weight, MAE weight, and config under one model directory."""
    base = Path(model_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {base}")

    search_roots = [base]
    for subdir in ("best", "ckpt"):
        candidate = base / subdir
        if candidate.is_dir():
            search_roots.append(candidate)

    config_candidates: list[Path] = []
    for root in search_roots:
        for path in (root / "config.yaml", root / "config.yml", root / "poly_mae_config.json"):
            if path.exists() and path not in config_candidates:
                config_candidates.append(path)

    mae_candidates: list[Path] = []
    for root in search_roots:
        candidates = [
            path
            for path in root.glob("*.pth")
            if path.is_file() and ("encoder_decoder" in path.name.lower() or "mae" in path.name.lower())
        ]
        for path in candidates:
            if path not in mae_candidates:
                mae_candidates.append(path)

    encoder_candidates: list[Path] = []
    for root in search_roots:
        for path in (root / "encoder.pth", root / "encoder_best.pth"):
            if path.exists() and path not in encoder_candidates:
                encoder_candidates.append(path)
        for path in sorted(root.glob("*encoder*.pth")):
            name_lower = path.name.lower()
            if "encoder_decoder" in name_lower:
                continue
            if path not in encoder_candidates:
                encoder_candidates.append(path)

    if not config_candidates:
        raise FileNotFoundError(f"No config file found in model_dir: {base}")


    def _encoder_rank(path: Path) -> tuple[int, int, float, str]:
        name_lower = path.name.lower()
        if name_lower == "encoder.pth":
            priority = 0
        elif name_lower == "encoder_best.pth":
            priority = 1
        else:
            priority = 2
        suffix = _extract_last_int(path.stem)
        suffix_rank = -(suffix if suffix is not None else -1)
        mtime_rank = -float(path.stat().st_mtime)
        return (priority, suffix_rank, mtime_rank, name_lower)

    encoder_candidates = sorted(encoder_candidates, key=_encoder_rank)


    if encoder_candidates:
        return encoder_candidates[0], config_candidates[0], False

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch forward export."""
    parser = argparse.ArgumentParser(
        description="Batch forward triangulated polygon shards into embeddings and MAE frequency maps."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing encoder/MAE checkpoints and config.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing triangulated shard `.pt` files.")
    parser.add_argument("--output_path", type=str, default="", help="Output `.pt` file path.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--device", type=str, default=_default_device(), help="Runtime device, e.g. cuda or cpu.")
    parser.add_argument("--precision", type=str, default="fp32", help="Runtime precision: fp32/bf16/fp16.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional export cap; 0 means all samples.")
    return parser

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
    

def cleanup():
    dist.destroy_process_group()

def worker(rank, world_size, args):

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank} initialized.")
    # 1. 设备初始化
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    project_root = _inject_repo_root()
    
    # 2. 路径处理：每个进程生成独特的文件名
    args.model_dir = str(_resolve_user_path(args.model_dir, project_root))
    args.data_dir = str(_resolve_user_path(args.data_dir, project_root))
    base_output_path = _build_output_path(project_root, args.data_dir, args.output_path or None)
    
    # 每个进程的文件名：output_name.part0.pt, output_name.part1.pt ...
    rank_output_path = base_output_path.parent / f"{base_output_path.stem}.part{rank}{base_output_path.suffix}"
    
    if rank == 0:
        ensure_dir(base_output_path.parent)
    dist.barrier() # 确保目录创建完毕

    # 3. 数据分片 (Data Sharding)
    all_shard_paths = resolve_triangle_shard_paths(args.data_dir, warn_fn=lambda m: print(m) if rank == 0 else None)
    my_shard_paths = all_shard_paths[rank::world_size] # 均匀分配

    # 4. 加载模型与插件
    encoder_weight, config_path, _ = _resolve_model_artifacts(args.model_dir)
    config = load_config_any(config_path)
    geom_type = str(config.get("geom_type", "polygon")).lower()
    codec = get_geometry_codec(geom_type, config, device=str(device))
    
    encoder = load_pretrained_encoder(
        weight_path=encoder_weight, config_path=config_path,
        device=device, precision=args.precision
    )
    encoder.eval()

    # 5. 局部变量
    local_samples = []
    exported_count = 0
    pending_tris = []

    def flush_pending_to_memory():
        nonlocal exported_count
        if not pending_tris: return
        
        batch_tris, lengths = pad_triangle_batch(pending_tris, device=device)
        with torch.no_grad():
            # 特征提取
            mag, phase = codec.cft_batch(batch_tris, lengths)
            imgs = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)
            true_imag = rasterize_triangles_pytorch(batch_tris)
            
            with autocast_context(device, args.precision):
                encoder_features = encoder(imgs)
            
            # 关键：立即转回 CPU 释放显存
            embeddings = encoder_features.float().cpu()
            imgs_cpu = true_imag.cpu()

        for i in range(len(pending_tris)):
            local_samples.append({
                "embedding": embeddings[i],
                "imag": imgs_cpu[i],
            })
        
        exported_count += len(pending_tris)
        pending_tris.clear()

    # 6. 开始循环
    with tqdm(my_shard_paths, desc=f"Rank {rank}", position=rank, disable=False) as pbar:
        for shard_path in pbar:
            shard_data = load_triangle_shard(shard_path)
            for sample in shard_data:
                pending_tris.append(_ensure_numpy_float32(sample))
                
                if len(pending_tris) >= args.batch_size:
                    flush_pending_to_memory()
                
                # 检查是否达到总样本限制 (近似平分)
                if args.max_samples > 0 and (exported_count * world_size) >= args.max_samples:
                    break
            if args.max_samples > 0 and (exported_count * world_size) >= args.max_samples:
                break

    flush_pending_to_memory()

    # 7. 独立保存本进程文件
    metadata = {
        "rank": rank,
        "world_size": world_size,
        "created_at": datetime.now().isoformat(),
        "sample_count": len(local_samples),
        "precision": args.precision,
    }
    
    payload = {"metadata": metadata, "samples": local_samples}
    torch.save(payload, rank_output_path)
    print(f"Rank {rank} saved {len(local_samples)} samples to {rank_output_path.name}")

    dist.barrier()
    if rank == 0:
        print(f"\n[DONE] All workers finished. Files are saved as {base_output_path.stem}.partX.pt")
    
    cleanup()

def getdata(args) -> None:
    """入口函数"""
    world_size = torch.cuda.device_count()
    if world_size == 0:
        world_size = 1 # 支持 CPU 模式
    """每个 GPU 进程执行的逻辑"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print(f"[INFO] Launching {world_size} processes...")
    mp.spawn(
        worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )



if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    getdata(args)


