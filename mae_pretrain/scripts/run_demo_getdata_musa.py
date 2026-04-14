#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_getdata.py
@Time    :   2026/04/09 14:06:10
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import torch_musa
import torch
import argparse
from pathlib import Path
import sys
from typing import Any
import re
import os
from tqdm import tqdm
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)
from mae_pretrain.src.datasets.sharded_pt_dataset import _ensure_numpy_float32
from mae_pretrain.src.datasets.geometry_polygon import pad_triangle_batch
from mae_pretrain.src.datasets.registry import get_geometry_codec
from mae_pretrain.src.models.factory import load_mae_model, load_pretrained_encoder
from mae_pretrain.src.utils.config import load_config_any
from mae_pretrain.src.datasets.shard_io import load_triangle_shard, resolve_triangle_shard_paths
from mae_pretrain.src.utils.filesystem import ensure_dir
from mae_pretrain.src.utils.precision import autocast_context, normalize_precision
from datetime import datetime
import torch.nn.functional as F

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

def _extract_last_int(text: str) -> int | None:
    """Extract the last integer substring from text, if present."""
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])

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

def _default_device() -> str:
    """Resolve default runtime device string."""
    try:
        import torch

        return "musa" if torch.musa.is_available() else "cpu"
    except Exception:
        return "cpu"


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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch forward export."""
    parser = argparse.ArgumentParser(
        description="Batch forward triangulated polygon shards into embeddings and MAE frequency maps."
    )
    parser.add_argument("--model_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/ckpt/", help="Directory containing encoder/MAE checkpoints and config.")
    parser.add_argument("--data_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/data/processed", help="Directory containing triangulated shard `.pt` files.")
    parser.add_argument("--output_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/", help="Output `.pt` file path.")
    parser.add_argument("--precision", type=str, default="fp32", help="Runtime precision: fp32/bf16/fp16.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional export cap; 0 means all samples.")
    parser.add_argument("--device", type=str, default=_default_device(), help="Runtime device, e.g. musa or cpu.")
    
    return parser


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from pathlib import Path
import sys
from typing import Any
import re
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch.nn.functional as F
import joblib

def setup_dist(args):
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        print("Not running in distributed mode, falling back to single GPU.")
        args.distributed = False
        return False

    args.distributed = True
    torch.musa.set_device(args.gpu)
    device = torch.device(f"musa:{args.gpu}")
    dist.init_process_group(
        backend="mccl", 
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
        device_id=device # 较新版本的 PyTorch 支持直接绑定
    )
    return True

def getdata(args) -> None:
    project_root = _inject_repo_root()
    
    # 1. 分布式初始化
    is_dist = setup_dist(args)
    rank = args.rank if is_dist else 0
    world_size = args.world_size if is_dist else 1
    device = torch.device(f"musa:{args.gpu}" if is_dist else args.device)

    # 2. 准备模型和工具
    args.model_dir = str(_resolve_user_path(args.model_dir, project_root))
    encoder_weight, config_path, _ = _resolve_model_artifacts(args.model_dir)
    config = load_config_any(config_path)
    geom_type = str(config.get("geom_type", "polygon")).lower()
    
    codec = get_geometry_codec(geom_type, config, device=str(device))
    encoder = load_pretrained_encoder(
        weight_path=encoder_weight,
        config_path=config_path,
        device=device,
        precision=args.precision,
    )
    encoder.eval()
    
    # 如果是分布式，包装模型（虽是推理，DDP 也能统一管理）
    if is_dist:
        encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
        # 推理模式下通常直接用原本模型即可，但为了兼容性可不包 DDP 仅用 .to(device)
    
    # 3. 读取数据 (所有进程都会读取，但后续只处理属于自己的部分)
    shard_paths = resolve_triangle_shard_paths(args.data_dir)
    all_tris = []
    if rank == 0:
        print(f"Loading shards from {args.data_dir}...")
    
    for shard_path in shard_paths:
        shard_data = load_triangle_shard(shard_path)
        for sample in shard_data:
            all_tris.append(_ensure_numpy_float32(sample))
            if args.max_samples > 0 and len(all_tris) >= args.max_samples:
                break
        if args.max_samples > 0 and len(all_tris) >= args.max_samples:
            break

    # 4. 数据分配 (按 Rank 切分数据)
    # 计算当前进程负责的索引范围
    total_samples = len(all_tris)
    samples_per_rank = (total_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, total_samples)
    my_tris = all_tris[start_idx:end_idx]

    # 5. 推理循环
    local_output_samples = []
    pbar = tqdm(total=len(my_tris), desc=f"Rank {rank} Inferring") if rank == 0 else None

    for i in range(0, len(my_tris), args.batch_size):
        batch = my_tris[i : i + args.batch_size]
        batch_tris, lengths = pad_triangle_batch(batch, device=device)
        
        with torch.no_grad():
            with autocast_context(device, args.precision):
                mag, phase = codec.cft_batch(batch_tris, lengths)
                imgs = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)
                encoder_features = encoder(imgs)
                

            # 存入本地结果
            embeddings = encoder_features.float().cpu()
            
            for j in range(len(batch)):
                local_output_samples.append({
                    "sample_index": start_idx + i + j,
                    "embedding": embeddings[j][0].clone(),
                })
        
        if pbar: pbar.update(len(batch))



    
    # 例如：output_shard_rank_0.pt, output_shard_rank_1.pt
    rank_output_path = os.path.join(args.output_dir, f"emb_rank{rank}.joblib")
    ensure_dir(Path(rank_output_path).parent)

    # 构造当前进程的 Payload
    metadata_local = {
        "rank": rank,
        "world_size": world_size,
        "sample_count": len(local_output_samples),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "precision": args.precision,
        "device": str(device),
    }
    payload = {
        "metadata": metadata_local,
        "samples": local_output_samples,
    }
    # 执行保存
    joblib.dump(payload, rank_output_path)
    print(f"[Rank {rank}] Saved {len(local_output_samples)} samples to {rank_output_path}")


    if is_dist:
        dist.barrier() # 等待所有卡写完
        if rank == 0:
            print(f"\n[DONE] All ranks finished. Files are saved as {args.output_dir}_rankX.joblib")
        dist.destroy_process_group()

def read_embeddings_from_shard(shard_path: str) -> list[dict[str, Any]]:
    """读取单个 shard 输出文件中的 embedding 样本列表"""
    data = joblib.load(shard_path)
    return data["samples"]

if __name__ == '__main__':
    parser = build_arg_parser()
    # 自动获取 local_rank 用于分布式
    args = parser.parse_args()
    getdata(args)
    # aa = read_embeddings_from_shard("/mnt/git-data/HB/poly2vec_mae/outputs/emb_rank0.joblib")