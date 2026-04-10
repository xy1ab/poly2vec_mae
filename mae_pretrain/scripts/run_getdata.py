#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_getdata.py
@Time    :   2026/04/09 14:06:10
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


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


def _build_output_path(project_root: Path, data_dir: str, output_path: str | None) -> Path:
    """Resolve output path from explicit CLI path or one timestamped default."""
    if output_path:
        return Path(output_path).expanduser().resolve()

    data_name = Path(data_dir).expanduser().resolve().name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "outputs" / "forward_batch" / f"{data_name}_forward_batch_{timestamp}.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch forward export."""
    parser = argparse.ArgumentParser(
        description="Batch forward triangulated polygon shards into embeddings and MAE frequency maps."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing encoder/MAE checkpoints and config.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing triangulated shard `.pt` files.")
    parser.add_argument("--output_path", type=str, default="", help="Output `.pt` file path.")
    parser.add_argument("--precision", type=str, default="fp32", help="Runtime precision: fp32/bf16/fp16.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional export cap; 0 means all samples.")
    parser.add_argument("--device", type=str, default=_default_device(), help="Runtime device, e.g. musa or cpu.")
    
    return parser


def getdata(args) -> None:
    """CLI main entrypoint."""
    project_root = _inject_repo_root()

    args.precision = normalize_precision(args.precision)
    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.max_samples < 0:
        raise ValueError(f"`max_samples` must be >= 0, got {args.max_samples}")

    args.model_dir = str(_resolve_user_path(args.model_dir, project_root))
    args.data_dir = str(_resolve_user_path(args.data_dir, project_root))
    output_path = _build_output_path(project_root, args.data_dir, args.output_path or None)
    ensure_dir(output_path.parent)

    requested_device = args.device or _default_device()
    if str(requested_device).startswith("musa") and not torch.musa.is_available():
        print("[WARN] MUSA unavailable, fallback to CPU for forward export.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    shard_paths = resolve_triangle_shard_paths(args.data_dir, warn_fn=lambda message: print(message))
    encoder_weight, config_path, encoder_fallback_to_mae = _resolve_model_artifacts(args.model_dir)

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


    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "encoder_weight": str(Path(encoder_weight).expanduser().resolve()),
        "precision": args.precision,
        "device": str(device),
        "spatial_size": 256,
        "geom_type": geom_type,
        "config": dict(config),
        "shard_paths": [str(path) for path in shard_paths],
        "processed_shard_count": 0,
        "sample_count": 0,
    }

    output_samples: list[dict[str, Any]] = []
    pending_tris: list[Any] = []
    exported_count = 0
    processed_shard_count = 0

    def flush_pending() -> None:
        """Run one inference batch and append output sample dicts."""
        nonlocal exported_count
        if not pending_tris:
            return

        batch_tris, lengths = pad_triangle_batch(pending_tris, device=device)
        with torch.no_grad():
            mag, phase = codec.cft_batch(batch_tris, lengths)
            imgs = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)
            # true_imag = rasterize_triangles_pytorch(batch_tris)
            with autocast_context(device, args.precision):
                encoder_features = encoder(imgs)

            embeddings = encoder_features.float().cpu()
            # 逆变换出模糊空间图 (通道1)
            raw_mag = torch.expm1(mag)
            # F_uv = raw_mag * torch.exp(1j * phase)
            F_uv_real = raw_mag * torch.cos(phase)
            F_uv_imag = raw_mag * torch.sin(phase)
            # x_spatial = engine.icft_2d(F_uv.squeeze(1), spatial_size=cfg.SPATIAL_SIZE)
            x_spatial = codec.icft_2d(F_uv_real.squeeze(1), F_uv_imag.squeeze(1), spatial_size=256)
            x_spatial = x_spatial.unsqueeze(1) #[1, 1, 256, 256]
            
            # 使用双线性插值, 将频域特征强制缩放到 256x256 (通道2, 通道3)
            mag_map = F.interpolate(mag, size=(256, 256), mode='bilinear', align_corners=False)
            phase_map = F.interpolate(phase, size=(256, 256), mode='bilinear', align_corners=False)
            
            # 拼接成 3 通道输入
            x_3channel = torch.cat([x_spatial, mag_map, phase_map], dim=1).squeeze(0).cpu().half() # [3, 256, 256]


        start_index = exported_count + 1
        for offset, tri_np in enumerate(pending_tris):
            sample_index = start_index + offset
            output_samples.append(
                {
                    "sample_index": int(sample_index),
                    # "triangles": torch.from_numpy(tri_np.astype("float32", copy=False)),
                    "embedding": embeddings[offset],
                    "x_3channel": x_3channel[offset],
                }
            )

        exported_count += len(pending_tris)
        pending_tris.clear()

    for shard_path in tqdm(shard_paths, desc="Reading shards"):
        processed_shard_count += 1
        shard_data = load_triangle_shard(shard_path)
        for sample in shard_data:
            tri_np = _ensure_numpy_float32(sample)
            pending_tris.append(tri_np)

            if len(pending_tris) >= args.batch_size:
                flush_pending()

            if args.max_samples > 0 and exported_count + len(pending_tris) >= args.max_samples:
                remaining = args.max_samples - exported_count
                if remaining < len(pending_tris):
                    pending_tris[:] = pending_tris[:remaining]
                flush_pending()
                break

        if args.max_samples > 0 and exported_count >= args.max_samples:
            break

    flush_pending()




    # metadata["processed_shard_count"] = int(processed_shard_count)
    metadata["sample_count"] = int(len(output_samples))
    if output_samples:
        metadata["embedding_dim"] = int(output_samples[0]["embedding"].numel())

    payload = {
        "metadata": metadata,
        "samples": output_samples,
    }
    torch.save(payload, output_path)

    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] Shards read: {processed_shard_count}")
    print(f"[DONE] Sample count: {len(output_samples)}")
    if output_samples:
        print(f"[DONE] Embedding dim: {int(output_samples[0]['embedding'].numel())}")
    else:
        print("[DONE] No samples exported.")


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    getdata(args)