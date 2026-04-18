#!/usr/bin/env python3
"""
Generate U-Net training data with shard saving
每个分片 10000 个样本，支持断点续传
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.path as mpltPath
import gc

_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from scripts.run_eval import _inject_repo_root, _resolve_model_paths, mag_phase_to_real_imag

_inject_repo_root()

from mae_pretrain.src.datasets.registry import get_geometry_codec
from mae_pretrain.src.models.factory import load_mae_model
from mae_pretrain.src.utils.precision import autocast_context, normalize_precision
import torch.distributed as dist
import os
SHARD_SIZE = 10000  # 每个分片 10000 个样本

def rasterize_triangles(tris, spatial_size=256) -> np.ndarray:
    if hasattr(tris, "detach"):
        tris_np = tris.detach().cpu().numpy()
    else:
        tris_np = np.asarray(tris)
    x = np.linspace(-1, 1, spatial_size)
    y = np.linspace(1, -1, spatial_size)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    mask = np.zeros(spatial_size * spatial_size, dtype=bool)
    for tri in tris_np:
        path = mpltPath.Path(tri)
        mask = mask | path.contains_points(points)
    return mask.reshape(spatial_size, spatial_size).astype(np.float32)


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}")
    )
    return local_rank

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/data/processed/test")
    parser.add_argument("--mae_model_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/ckpt")
    parser.add_argument("--save_dir", type=str, default="./data/unet_dataset/ddp_test")
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--spatial_size", type=int, default=256)
    parser.add_argument("--precision", type=str, default="fp32")
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    local_rank = setup_distributed()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    all_files = sorted(list(Path(args.data_dir).glob("*.pt")))
    # all_files = sorted(list(Path(args.data_dir).glob("*_tri_part*.pt")))
    my_files = np.array_split(all_files, world_size)[rank]


    args.precision = normalize_precision(args.precision)

    ckpt_path, cfg_path = _resolve_model_paths(args.mae_model_dir)
    model, model_cfg = load_mae_model(
        weight_path=ckpt_path, config_path=cfg_path,
        device=device, precision=args.precision,
    )
    model.eval()

    patch_size = int(model_cfg.get("patch_size", 2))
    codec_cfg = {
        "geom_type": "polygon",
        "pos_freqs": int(model_cfg.get("pos_freqs", 31)),
        "w_min": float(model_cfg.get("w_min", 0.1)),
        "w_max": float(model_cfg.get("w_max", 100.0)),
        "freq_type": str(model_cfg.get("freq_type", "geometric")),
        "patch_size": patch_size,
    }
    codec = get_geometry_codec("polygon", codec_cfg, device=str(device))

    # all_polys = torch.load(args.data_path, map_location="cpu", weights_only=False)
    # total = len(all_polys)
    
    # start_idx = args.start_idx
    # end_idx = total if args.num_samples == -1 else min(start_idx + args.num_samples, total)
    # num_samples = end_idx - start_idx

    save_dir = Path(args.save_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    # print(f"[INFO] 总样本数: {total}")
    # print(f"[INFO] 从第 {start_idx} 个样本开始")
    # print(f"[INFO] 生成 {num_samples} 个样本")
    # print(f"[INFO] 每个分片 {SHARD_SIZE} 个样本")

    # 计算起始分片 ID
    # start_shard = start_idx // SHARD_SIZE
    # current_shard = start_shard
    all_samples = []
    current_shard = 0

    # pbar = tqdm(range(start_idx, end_idx), desc="Generating", initial=start_idx, total=end_idx)
    pbar = tqdm(my_files, desc=f"Rank {rank} Processing", disable=(rank != 0))
    for f_path in pbar:
        # 逐个文件加载，避免 OOM
        data = torch.load(f_path, map_location="cpu", weights_only=False)
        for tris in data:
            batch_tris = torch.as_tensor(tris, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([tris.shape[0]], device=device)

            with torch.no_grad():
                mag_fix, phase_fix = codec.cft_batch(batch_tris, lengths)
                imgs_fix = torch.cat([mag_fix, torch.cos(phase_fix), torch.sin(phase_fix)], dim=1)

                with autocast_context(device, args.precision):
                    pred_fix, mask_fix = model(imgs_fix, mask_ratio=args.mask_ratio)

                pred_fix = pred_fix.float()
                mask_fix = mask_fix.float()
                
                h, w = 64, 32
                h_p, w_p = h // patch_size, w // patch_size

                img_orig = imgs_fix[0].detach().cpu()
                pred_img = pred_fix[0].detach().cpu().reshape(h_p, w_p, 3, patch_size, patch_size)
                pred_img = torch.einsum("hwcpq->chpwq", pred_img).reshape(3, h, w)
                mask_map = mask_fix[0].detach().cpu().reshape(h_p, w_p, 1, 1).expand(-1, -1, patch_size, patch_size)
                mask_map = mask_map.permute(0, 2, 1, 3).reshape(h, w)

                img_recon = img_orig.clone()
                img_recon[:, mask_map == 1] = pred_img[:, mask_map == 1]

                mag_recon = img_recon[0].unsqueeze(0).to(device)
                cos_recon = img_recon[1].unsqueeze(0).to(device)
                sin_recon = img_recon[2].unsqueeze(0).to(device)
                phase_recon = torch.atan2(sin_recon, cos_recon)

                real_recon, imag_recon = mag_phase_to_real_imag(mag_recon, phase_recon)
                spatial_recon = codec.icft_2d(real_recon, imag_recon, spatial_size=args.spatial_size)[0].detach().cpu()

                mag_upsampled = F.interpolate(
                    mag_recon.unsqueeze(1), size=(args.spatial_size, args.spatial_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0).cpu()
                phase_upsampled = F.interpolate(
                    phase_recon.unsqueeze(1), size=(args.spatial_size, args.spatial_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0).cpu()

                x_input = torch.stack([spatial_recon, mag_upsampled, phase_upsampled], dim=0)
                y_label = rasterize_triangles(tris, spatial_size=args.spatial_size)
                y_label = torch.from_numpy(y_label).float()

                all_samples.append({'input': x_input.float(), 'label': y_label.unsqueeze(0)})

                # 检查是否需要保存分片
                if len(all_samples) >= SHARD_SIZE:
                    shard_path = save_dir / f"shard_r{rank}_{current_shard:04d}.pt"
                    torch.save(all_samples, shard_path)
                    print(f"\n   💾 已保存分片: {shard_path} ({len(all_samples)} 个样本)")
                    all_samples = []
                    current_shard += 1
                    torch.cuda.empty_cache()
               

                # 更新进度条
                if rank == 0:
                    pbar.set_postfix({"shard": current_shard, "buf": len(all_samples)})

    # 保存剩余样本
    if all_samples:
        torch.save(all_samples, save_dir / f"shard_r{rank}_final.pt")
        print(f"\n   💾 已保存最后分片: {shard_path} ({len(all_samples)} 个样本)")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()