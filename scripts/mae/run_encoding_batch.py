import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mae_core.model import load_pretrained_encoder
from utils.config.loader import load_json_config, load_yaml_config
from utils.fourier.engine import build_poly_fourier_converter_from_config
from utils.io.filesystem import ensure_dir
from utils.io.precision import autocast_context, normalize_precision


def _iter_vector_files(input_dirs, recursive=True):
    suffixes = (".shp", ".geojson")
    all_files = []
    for base_dir in input_dirs:
        root = Path(base_dir)
        if not root.exists():
            continue
        if recursive:
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in suffixes:
                    all_files.append(p)
        else:
            for p in root.iterdir():
                if p.is_file() and p.suffix.lower() in suffixes:
                    all_files.append(p)
    return sorted(set(all_files))


def _expand_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return []


def _limit_normalize(coords, eps=1e-6):
    max_abs = float(np.max(np.abs(coords)))
    if max_abs < eps:
        return None
    return (coords / max_abs).astype(np.float32)


def _augment_triangles(tris, rng, scale_min=0.5, scale_max=1.0):
    pts = tris.reshape(-1, 2).astype(np.float32, copy=True)

    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    pts = pts.dot(rot)

    min_c = pts.min(axis=0)
    max_c = pts.max(axis=0)
    span = np.max(max_c - min_c)
    if span <= 1e-6:
        return tris.astype(np.float32, copy=True)

    fit_scale = min(1.0, (2.0 - 1e-6) / span)
    random_scale = float(rng.uniform(scale_min, scale_max))
    scale = min(random_scale, fit_scale)
    pts *= scale

    min_c = pts.min(axis=0)
    max_c = pts.max(axis=0)
    tx_low, tx_high = -1.0 - min_c[0], 1.0 - max_c[0]
    ty_low, ty_high = -1.0 - min_c[1], 1.0 - max_c[1]

    tx = float(rng.uniform(tx_low, tx_high)) if tx_low <= tx_high else float((tx_low + tx_high) * 0.5)
    ty = float(rng.uniform(ty_low, ty_high)) if ty_low <= ty_high else float((ty_low + ty_high) * 0.5)
    pts += np.array([tx, ty], dtype=np.float32)

    return pts.reshape(tris.shape).astype(np.float32)


def _build_meta(tris, node_count):
    pts = tris.reshape(-1, 2)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    center = (min_xy + max_xy) * 0.5
    side_len = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
    return np.array([center[0], center[1], side_len, float(node_count)], dtype=np.float32)


def _pad_triangles(triangles_np_list, device):
    lengths = torch.tensor([arr.shape[0] for arr in triangles_np_list], dtype=torch.long, device=device)
    max_len = int(lengths.max().item())
    batch = torch.zeros((len(triangles_np_list), max_len, 3, 2), dtype=torch.float32, device=device)
    for i, arr in enumerate(triangles_np_list):
        tri_tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        batch[i, : arr.shape[0]] = tri_tensor
    return batch, lengths


@torch.no_grad()
def _encode_triangle_batch(triangles_np_list, encoder, fourier_engine, device, precision):
    batch_tris, lengths = _pad_triangles(triangles_np_list, device=device)
    mag, phase = fourier_engine.cft_polygon_batch(batch_tris, lengths)
    imgs = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)
    with autocast_context(device, precision):
        features = encoder(imgs)
    return features[:, 0, :].float().cpu()


def _resolve_model_artifacts(model_dir):
    base = Path(model_dir).expanduser()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"模型目录不存在或不是文件夹: {base}")

    config_candidates = []
    preferred_configs = [
        base / "config.yaml",
        base / "config.yml",
        base / "poly_mae_config.json",
    ]
    for p in preferred_configs:
        if p.exists():
            config_candidates.append(p)
    config_candidates.extend(sorted(p for p in base.glob("*.yaml") if p not in config_candidates))
    config_candidates.extend(sorted(p for p in base.glob("*.yml") if p not in config_candidates))
    config_candidates.extend(sorted(p for p in base.glob("*.json") if p not in config_candidates))
    if not config_candidates:
        raise FileNotFoundError(
            f"在模型目录中未找到配置文件: {base}\n"
            f"请确保目录下包含 config.yaml（推荐）/ config.yml / poly_mae_config.json。"
        )
    encoder_config = config_candidates[0]

    weight_candidates = []
    preferred_weights = [
        base / "encoder.pth",
    ]
    for p in preferred_weights:
        if p.exists():
            weight_candidates.append(p)
    weight_candidates.extend(sorted(p for p in base.glob("poly_encoder_epoch_*.pth") if p not in weight_candidates))
    weight_candidates.extend(sorted(p for p in base.glob("*encoder*.pth") if p not in weight_candidates))
    if not weight_candidates:
        available_pth = sorted(p.name for p in base.glob("*.pth"))
        raise FileNotFoundError(
            f"在模型目录中未找到可用 encoder 权重文件: {base}\n"
            f"优先查找: encoder.pth / poly_encoder_epoch_*.pth\n"
            f"当前目录 .pth 文件: {available_pth if available_pth else '无'}"
        )
    encoder_weight = weight_candidates[0]

    return encoder_weight, encoder_config


def _load_config_any(config_path):
    path = str(config_path).lower()
    if path.endswith(".yaml") or path.endswith(".yml"):
        return load_yaml_config(config_path)
    return load_json_config(config_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Batch encode vector polygons into downstream samples")
    parser.add_argument("--data_dir", type=str, required=True, help="原始矢量数据目录（脚本自动搜索 .shp/.geojson）")
    parser.add_argument("--recursive", dest="recursive", action="store_true", help="递归扫描数据目录（默认开启）")
    parser.add_argument("--no_recursive", dest="recursive", action="store_false", help="仅扫描数据目录顶层")
    parser.set_defaults(recursive=True)
    parser.add_argument("--augment_times", type=int, default=10, help="每个原始样本额外增强次数（总数=1+augment_times）")
    parser.add_argument("--scale_min", type=float, default=0.5, help="增强缩放下限")
    parser.add_argument("--scale_max", type=float, default=1.0, help="增强缩放上限")
    parser.add_argument("--batch_size", type=int, default=256, help="编码批大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="bf16", help="编码精度: fp32|fp16|bf16")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录（脚本自动寻找配置和 encoder 权重）")
    parser.add_argument("--output_path", type=str, default="", help="输出 .pt 路径；留空则写入 data/emb/encoded_samples_<timestamp>.pt")
    parser.add_argument("--dry_run", action="store_true", help="仅检查路径/文件可读性与样本统计，不执行编码与保存")
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.precision = normalize_precision(args.precision)
    # 允许 GDAL 自动恢复/重建缺失的 .shx 索引，避免常见 shp 读取失败。
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

    if args.augment_times < 0:
        raise ValueError("--augment_times 不能为负数。")
    if args.scale_min <= 0 or args.scale_max <= 0 or args.scale_min > args.scale_max:
        raise ValueError("请保证 0 < scale_min <= scale_max。")

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    vector_files = _iter_vector_files([args.data_dir], recursive=args.recursive)
    if not vector_files:
        raise FileNotFoundError(
            f"未在输入目录中找到任何 .shp/.geojson 文件。\n"
            f"data_dir={Path(args.data_dir).expanduser()} | recursive={args.recursive}"
        )
    print(f"[Info] 发现矢量文件 {len(vector_files)} 个。")

    encoder_weight, encoder_config = _resolve_model_artifacts(args.model_dir)
    if not encoder_weight.exists():
        raise FileNotFoundError(f"找不到 encoder 权重: {encoder_weight}")
    if not encoder_config.exists():
        raise FileNotFoundError(f"找不到模型配置: {encoder_config}")
    print(f"[Info] 使用 encoder 权重: {encoder_weight}")
    print(f"[Info] 使用模型配置: {encoder_config}")

    config = _load_config_any(str(encoder_config))
    triangulator = build_poly_fourier_converter_from_config(config, device="cpu")

    base_records = []
    skipped_invalid = 0
    for file_path in tqdm(vector_files, desc="Reading vectors"):
        try:
            gdf = gpd.read_file(file_path)
        except Exception as exc:
            print(f"[Warn] 读取失败 {file_path}: {exc}")
            continue

        for row_idx, geom in enumerate(gdf.geometry):
            polygons = _expand_polygons(geom)
            for part_idx, poly in enumerate(polygons):
                coords = np.asarray(poly.exterior.coords, dtype=np.float32)
                if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
                    coords = coords[:-1]
                node_count = int(coords.shape[0])
                if node_count < 3:
                    skipped_invalid += 1
                    continue

                norm_coords = _limit_normalize(coords)
                if norm_coords is None:
                    skipped_invalid += 1
                    continue

                tris = triangulator.triangulate_polygon(norm_coords)
                if tris.shape[0] == 0:
                    skipped_invalid += 1
                    continue

                base_records.append(
                    {
                        "triangles": tris.astype(np.float32),
                        "node_count": node_count,
                        "src_file": str(file_path),
                        "row_idx": row_idx,
                        "part_idx": part_idx,
                    }
                )

    if not base_records:
        raise RuntimeError("没有可用的有效 Polygon 样本，请检查输入数据。")

    total_targets = len(base_records) * (args.augment_times + 1)
    print(f"[Info] 有效原始样本数: {len(base_records)}")
    print(f"[Info] 计划产出样本数(含原始+增强): {total_targets}")
    print(f"[Info] 过滤无效样本数: {skipped_invalid}")

    if args.dry_run:
        print("[DryRun] 检查完成：不执行编码与保存。")
        return

    encoder = load_pretrained_encoder(
        str(encoder_weight),
        str(encoder_config),
        device=device,
        precision=args.precision,
    )
    encoder.eval()
    fourier_engine = build_poly_fourier_converter_from_config(config, device=device)

    pending_tris = []
    pending_meta = []
    output_samples = []

    def flush_pending():
        if not pending_tris:
            return
        emb = _encode_triangle_batch(
            pending_tris,
            encoder=encoder,
            fourier_engine=fourier_engine,
            device=device,
            precision=args.precision,
        )
        for i, tri_np in enumerate(pending_tris):
            sample = {
                "meta": torch.from_numpy(pending_meta[i]),                # (4,)
                "embedding": emb[i],                                      # (N,)
                "triangles": torch.from_numpy(tri_np.astype(np.float32)), # (T,3,2)
            }
            output_samples.append(sample)
        pending_tris.clear()
        pending_meta.clear()

    for rec in tqdm(base_records, desc="Augment + Encode"):
        base_tri = rec["triangles"]
        node_count = rec["node_count"]

        for aug_id in range(args.augment_times + 1):
            if aug_id == 0:
                tri_aug = base_tri.astype(np.float32, copy=True)
            else:
                tri_aug = _augment_triangles(
                    base_tri,
                    rng=rng,
                    scale_min=args.scale_min,
                    scale_max=args.scale_max,
                )

            meta = _build_meta(tri_aug, node_count=node_count)
            pending_tris.append(tri_aug)
            pending_meta.append(meta)

            if len(pending_tris) >= args.batch_size:
                flush_pending()

    flush_pending()

    if args.output_path:
        output_path = Path(args.output_path).expanduser()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = Path(PROJECT_ROOT) / "data" / "emb" / f"encoded_samples_{timestamp}.pt"

    ensure_dir(output_path.parent)
    torch.save(output_samples, str(output_path))

    if output_samples:
        emb_dim = int(output_samples[0]["embedding"].numel())
        print(f"[Done] 样本已保存至: {output_path}")
        print(f"[Done] 样本总数: {len(output_samples)}")
        print(f"[Done] embedding 维度: {emb_dim}")
        print(f"[Done] 样本结构: {{'meta':(4), 'embedding':({emb_dim}), 'triangles':(T,3,2)}}")
    else:
        print("[Done] 未生成任何样本。")


if __name__ == "__main__":
    main()
