"""
Generate U-Net training data with checkpoint saving
支持断点续传，每 5000 个样本自动保存
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.path as mpltPath

sys.path.insert(0, '/home/cx/cz/poly2vec_mae/mae_pretrain')
from scripts.run_eval import _inject_repo_root, _resolve_model_paths, mag_phase_to_real_imag

_inject_repo_root()

from mae_pretrain.src.datasets.registry import get_geometry_codec
from mae_pretrain.src.models.factory import load_mae_model
from mae_pretrain.src.utils.precision import autocast_context, normalize_precision

SAVE_INTERVAL = 5000  # 每 5000 个样本保存一次

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

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mae_model_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./data/unet_dataset")
    parser.add_argument("--output_file", type=str, default="all_samples.pt")
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--spatial_size", type=int, default=256)
    parser.add_argument("--precision", type=str, default="bf16")
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    all_polys = torch.load(args.data_path, map_location="cpu", weights_only=False)
    total = len(all_polys)
    num_samples = total if args.num_samples == -1 else min(args.num_samples, total)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / args.output_file

    # ========== 断点续传：加载已生成的样本 ==========
    all_samples = []
    start_idx = 0
    if output_path.exists():
        all_samples = torch.load(output_path)
        start_idx = len(all_samples)
        print(f"[INFO] 发现已有 {start_idx} 个样本，从第 {start_idx} 个继续生成")
    else:
        print(f"[INFO] 从头开始生成 {num_samples} 个样本")

    if start_idx >= num_samples:
        print(f"[INFO] 已完成！共 {len(all_samples)} 个样本")
        return

    print(f"[INFO] 需要生成 {num_samples - start_idx} 个新样本")

    pbar = tqdm(range(start_idx, num_samples), desc="Generating", initial=start_idx, total=num_samples)

    for idx in pbar:
        tris = all_polys[idx]
        batch_tris = torch.as_tensor(tris, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([tris.shape[0]], device=device)

        with torch.no_grad():
            mag_fix, phase_fix = codec.cft_batch(batch_tris, lengths)
            imgs_fix = torch.cat([mag_fix, torch.cos(phase_fix), torch.sin(phase_fix)], dim=1)

            with autocast_context(device, args.precision):
                pred_fix, mask_fix = model(imgs_fix, mask_ratio=args.mask_ratio)

            pred_fix = pred_fix.float()
            mask_fix = mask_fix.float()  # ✅ 添加这行，确保 mask_fix 是 float
            
            h, w = 64, 32
            h_p, w_p = h // patch_size, w // patch_size

            # ✅ 1. 保存原始图像
            img_orig = imgs_fix[0].detach().cpu()

            # ✅ 2. 重建的 patch 图像
            pred_img = pred_fix[0].detach().cpu().reshape(h_p, w_p, 3, patch_size, patch_size)
            pred_img = torch.einsum("hwcpq->chpwq", pred_img).reshape(3, h, w)

            # ✅ 3. 生成掩码图
            mask_map = mask_fix[0].detach().cpu().reshape(h_p, w_p, 1, 1).expand(-1, -1, patch_size, patch_size)
            mask_map = mask_map.permute(0, 2, 1, 3).reshape(h, w)

            # ✅ 4. 组合原始和重建（关键修复）
            img_recon = img_orig.clone()
            img_recon[:, mask_map == 1] = pred_img[:, mask_map == 1]

            # ✅ 5. 后续处理
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

            # ========== 每 N 个样本清理一次 ==========
            if (idx + 1) % 5000 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                print(f"   🧹 已清理 GPU 缓存和内存")
            # ======================================

        # ========== 每 SAVE_INTERVAL 个样本保存一次 ==========
        if (idx + 1) % SAVE_INTERVAL == 0:
            torch.save(all_samples, output_path)
            pbar.set_postfix({"saved": len(all_samples)})

    # 最终保存
    torch.save(all_samples, output_path)
    print(f"\n✅ 完成！共 {len(all_samples)} 个样本，保存至: {output_path}")

if __name__ == "__main__":
    main()