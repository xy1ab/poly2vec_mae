#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_demo.py
@Time    :   2026/04/23 11:30:01
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''



import os
import time
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import numpy as np
import cv2  # 引入 OpenCV
import segmentation_models_pytorch as smp
from tqdm import tqdm
from pathlib import Path
import sys
import torch.nn.functional as F
if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CURRENT_DIR.parent
    _REPO_ROOT = _PROJECT_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

def _preflight_validate_index_shards(ind_shards: list[Path], helpers_module) -> int:
    """Validate index shard payload schema before launching decode workers."""
    import torch

    total_samples = 0
    for shard_path in ind_shards:
        samples = helpers_module.load_torch_list(shard_path)
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(f"Index shard sample must be dict: {shard_path}#{sample_index}")
            if "indices" not in sample or "meta" not in sample:
                raise KeyError(f"Index shard sample missing `indices`/`meta`: {shard_path}#{sample_index}")
            helpers_module.normalize_indices_grid(sample["indices"])
        total_samples += len(samples)
    return total_samples

def _setup_distributed(args, helpers):
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(torch.device(device))
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}")
    )   
    return rank, local_rank, world_size, device

def _build_rank_output_path(output_dir: str, input_shard_path: Path, rank: int) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    in_path = Path(input_shard_path).expanduser().resolve()
    stem_path = output_root / f"ind2img_{in_path.stem}.pt"
    return stem_path.with_name(f"{stem_path.stem}_rank{rank:03d}{stem_path.suffix}")


def post_process_results(sigmoid_output, original_sizes, threshold=0.5):
    """
    sigmoid_output: [B, 1, 256, 256] 概率值 (0-1)
    original_sizes: list of (h, w)
    """
    final_masks = []
    
    for i in range(sigmoid_output.shape[0]):
        # 1. 提取单张概率图 [1, 1, 256, 256]
        prob_map = sigmoid_output[i].unsqueeze(0)
        
        # 2. 插值回原始分辨率 (使用双线性插值)
        # 注意：这里我们是在概率图层面插值
        resized_prob = F.interpolate(
            prob_map, 
            size=(original_sizes[i],original_sizes[i]), 
            mode='bilinear', 
            align_corners=False
        )
        # 3. 在原分辨率下执行二值化
        binary_mask = (resized_prob > threshold)
        final_masks.append(binary_mask.squeeze(0))
    return final_masks

def _process_one_batch(pipeline, ds_model, helpers, start_index: int, batch_samples, resolution:int=5, nicft: int=256):
    """Process one index batch and return output records with global start index."""
    import torch

    batch_indices = []
    batch_meta = []
    batch_sample_indices = []
    for local_index, sample in enumerate(batch_samples):
        if not isinstance(sample, dict):
            raise TypeError(f"Index batch sample must be dict: local#{local_index}")
        if "indices" not in sample or "meta" not in sample:
            raise KeyError(f"Index batch sample missing `indices`/`meta`: local#{local_index}")
        batch_indices.append(helpers.normalize_indices_grid(sample["indices"]))
        batch_meta.append(sample["meta"])
        batch_sample_indices.append(int(sample.get("sample_index", start_index + local_index)))

    indices_batch = torch.stack(batch_indices, dim=0)
    # indices_batch_u16 = helpers.to_uint16_indices(indices_batch, context="ind2img")
    real_batch, imag_batch = pipeline.decode_indices(indices_batch)

    real_for_icft = real_batch.to(pipeline.device)
    imag_for_icft = imag_batch.to(pipeline.device)
    target_h = int(pipeline.codec.converter.U.shape[0])
    target_w = int(pipeline.codec.converter.U.shape[1])
    if real_for_icft.shape[1] > target_h or real_for_icft.shape[2] > target_w:
        raise ValueError(
            "Decoded valid frequency grid is larger than codec full grid: "
            f"decoded={tuple(real_for_icft.shape)}, full=({target_h}, {target_w})"
        )
    if real_for_icft.shape[1] != target_h or real_for_icft.shape[2] != target_w:
        padded_real = torch.zeros(
            (real_for_icft.shape[0], target_h, target_w),
            dtype=real_for_icft.dtype,
            device=real_for_icft.device,
        )
        padded_imag = torch.zeros(
            (imag_for_icft.shape[0], target_h, target_w),
            dtype=imag_for_icft.dtype,
            device=imag_for_icft.device,
        )
        padded_real[:, : real_for_icft.shape[1], : real_for_icft.shape[2]] = real_for_icft
        padded_imag[:, : imag_for_icft.shape[1], : imag_for_icft.shape[2]] = imag_for_icft
        real_for_icft = padded_real
        imag_for_icft = padded_imag
    icft_batch = pipeline.codec.icft_2d(
        f_uv_real=real_for_icft,
        f_uv_imag=imag_for_icft,
        spatial_size=int(nicft),
    ).float()
    with torch.no_grad():
        logits = ds_model(icft_batch.unsqueeze(1))
        pred = torch.sigmoid(logits).cpu()
    
    sample_nicfts = []
    if int(resolution) > 0:
        for metadata in batch_meta:
            dL = float(metadata[2])
            nicft = int(np.ceil(dL * 118000.0 / float(resolution)))
            sample_nicfts.append(nicft)
    pred_bin = post_process_results(pred, sample_nicfts)

    records = []
    for sample_offset in range(len(batch_samples)):
        record = {
            "sample_index": int(batch_sample_indices[sample_offset]),
            "meta_data": batch_meta[sample_offset],
            "pred_bin": pred_bin[sample_offset]
        }
        records.append(record)

    return {
        "records": records,
    }

## TODO
def research_index_require(sample_index_list, shard_dir):
    pass


def main():

    if __package__ in {None, ""}:
        import importlib

        helpers = importlib.import_module("rvqae_pretrain.scripts.batch_infer_common")
        pipeline_module = importlib.import_module("rvqae_pretrain.src.engine.pipeline")
    else:
        from . import batch_infer_common as helpers
        from ..src.engine import pipeline as pipeline_module

    parser = argparse.ArgumentParser(description="High-Performance DDP Inference with OpenCV")
    parser.add_argument("--ind_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/emb", help="Directory containing tri2ind shard files.")
    parser.add_argument("--model_dir", type=str, default='/mnt/git-data/HB/poly2vec_mae/outputs/20260417_1945/best')
    parser.add_argument("--downstream_model_path", type=str, default='/mnt/git-data/HB/poly2vec_mae/outputs/unet_ckpt/unet_best.pth')
    parser.add_argument("--output_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/demo")
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--num_workers", type=int, default=4) 
    args = parser.parse_args()
    
    rank, local_rank, world_size, device = _setup_distributed(args, helpers)
    is_rank0 = rank == 0
    decoder_path, quantizer_path, config_path = helpers.resolve_decode_paths(args.model_dir)
    ind_shards = helpers.resolve_ind_shards(args.ind_dir)

    total_samples = 0
    if is_rank0:
        total_samples = _preflight_validate_index_shards(ind_shards, helpers)
        print(
            f"[INFO] Index preflight passed: {len(ind_shards)} shards, "
            f"total_samples={total_samples}, world_size={world_size}, device={device}"
        )
        # helpers.clear_task_outputs(
        #     args.output_dir,
        #     task_prefix="ind2img",
        #     manifest_name="ind2img.manifest.json",
        # )

    dist.barrier(device_ids=[torch.device(device).index])


    ## 加载模型
    pipeline = pipeline_module.PolyRvqDecodePipeline(
        decoder_path=str(decoder_path),
        quantizer_path=str(quantizer_path),
        config_path=str(config_path),
        device=str(device),
        precision="fp32",
    )

    ds_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None
    ).to(device)
    
    ds_model.load_state_dict(torch.load(args.downstream_model_path, map_location=device))
    ds_model.eval()


    ## 解码数据
    if is_rank0:
        total_progress = tqdm(total=total_samples, desc="ind2img total", unit="sample", position=0)

    for shard_index, ind_path in enumerate(ind_shards, start=1):
        if is_rank0:
            print(f"[INFO] Loading shard {shard_index}/{len(ind_shards)}: {Path(ind_path).name}")
        samples = helpers.load_torch_list(ind_path)
        shard_size = len(samples)
        rank_outputs = []
        rank_sample_count = 0
        shard_progress = (
            tqdm(
                total=shard_size,
                desc=f"ind2img shard {shard_index}/{len(ind_shards)}",
                unit="sample",
                leave=False,
                position=1,
            )
            if is_rank0
            else None
        )

        num_batches = (shard_size + args.batch_size - 1) // args.batch_size
        if is_rank0:
            print(
                f"[INFO] Processing shard {shard_index}/{len(ind_shards)}: "
                f"samples={shard_size}, batches={num_batches}, world_size={world_size}, output_mode=rank_part"
            )
        for round_start in range(0, num_batches, world_size):
            batch_index = round_start + rank

            if batch_index < num_batches:
                start = batch_index * args.batch_size
                end = min(start + args.batch_size, shard_size)
                batch_samples = samples[start:end]
                local_payload = _process_one_batch(
                    pipeline=pipeline,
                    ds_model=ds_model,
                    helpers=helpers,
                    start_index=int(start),
                    batch_samples=batch_samples,
                )
                records = list(local_payload["records"])
                if len(records) != int(local_payload["sample_count"]):
                    raise RuntimeError("Local ind2img batch result length is inconsistent.")
                rank_outputs.extend(records)
                rank_sample_count += len(records)

            if is_rank0:
                round_sample_count = 0
                for round_rank in range(world_size):
                    round_batch_index = round_start + round_rank
                    if round_batch_index >= num_batches:
                        continue
                    start_index = round_batch_index * args.batch_size
                    end_index = min(start_index + args.batch_size, shard_size)
                    round_sample_count += end_index - start_index
                shard_progress.update(round_sample_count)
                total_progress.update(round_sample_count)

        output_path = _build_rank_output_path(args.output_dir, ind_path, rank)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[RANK {rank}] Saving part shard {shard_index}/{len(ind_shards)}: "
            f"{output_path.name} ({rank_sample_count} samples)",
            flush=True,
        )
        torch.save(rank_outputs, output_path)

    # # 2. 文件夹准备
    # current_time = time.strftime("%Y%m%d_%H%M%S")
    # run_save_dir = os.path.join(args.save_dir, f"run_cv2_{current_time}")
    # vis_dir = os.path.join(run_save_dir, "plots")
    
    # if is_rank0:
    #     os.makedirs(vis_dir, exist_ok=True)
    # dist.barrier()

    # # 3. 数据加载 (优化：预提取映射)
    # if is_rank0: 
    #     print(f"📦 Loading data from {args.data_path}...")

    # data = torch.load(args.data_path, map_location='cpu')
    # data_samples = data['samples']
    
    # # 提取 Tensor
    # inputs = torch.stack([item['x_3channel'] for item in data_samples])
    
    # # --- 关键优化点 ---
    # # 预先提取业务自定义的 sample_index，建立映射表
    # # data_samples 是个 list，这步操作很快
    # all_sample_indices = [item.get('sample_index', i) for i, item in enumerate(data_samples)]
    
    # dataset = TensorDataset(inputs)
    # sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    
    # # 性能优化：pin_memory 在 DDP 下非常重要
    # loader = DataLoader(
    #     dataset, 
    #     batch_size=args.batch_size, 
    #     sampler=sampler, 
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=False
    # )

    # # 4. 加载模型并包装 DDP
    # if is_master: print("🤖 Loading model...")
    # model = smp.Unet(
    #     encoder_name="resnet34", encoder_weights=None,
    #     in_channels=3, classes=1, activation='sigmoid'
    # ).to(device)
    # state_dict = torch.load(args.model_path, map_location=device)
    # model.load_state_dict(state_dict)
    # model = DDP(model, device_ids=[local_rank])
    # model.eval()

    # # 5. 推理并**保存所有图片**
    # start_time = time.time()
    # if is_master: print(f"🚀 开始推理 (总计 {len(dataset)} 张图片, 并行保存)...")
    
    # # 获取当前进程负责的数据集索引顺序
    # rank_indices = list(sampler)
    # dataset_len = len(dataset)

    # with torch.no_grad():
    #     # 进度条处理
    #     pbar = tqdm(loader, disable=not is_master, desc="Infer & CV2 Save")
        
    #     for batch_idx, (batch_x,) in enumerate(pbar):
    #         batch_x = batch_x.to(device)
    #         pred = model(batch_x)
            
    #         # 将 Tensor 转为 Numpy 移回 CPU，方便 OpenCV 处理
    #         batch_x_cpu = batch_x.cpu().numpy()
    #         pred_cpu = pred.cpu().numpy() # 形状 (B, 1, H, W)
            
    #         # --- 核心保存逻辑 ---
    #         batch_size_actual = batch_x.size(0)
    #         for i in range(batch_size_actual):
    #             # 计算在 loader 里的线性偏移
    #             local_ptr = batch_idx * args.batch_size + i
                
    #             # 越界检查 (处理 DDP 自动 Padding 的重复数据)
    #             if local_ptr >= len(rank_indices):
    #                 continue
                
    #             # a. 获取在整个 Dataset 中的原始位置
    #             raw_idx = rank_indices[local_ptr]
                
    #             # 只有当全局索引在原数据集长度范围内时才保存 (去除 padding 数据)
    #             if raw_idx < dataset_len:
    #                 # b. 获取你要求的真正的 sample_index
    #                 true_sid = all_sample_indices[raw_idx]
                    
    #                 # c. 调用 OpenCV 高性能保存函数
    #                 # 传入：sample_index, 输入Tensor(CPU), 预测Tensor(CPU), Rank号, 目录
    #                 # 注意 pred_cpu[i][0] 取出的是 (H, W) 的 NumPy 数组
    #                 save_single_image_opencv(
    #                     true_sid, 
    #                     batch_x_cpu[i], 
    #                     pred_cpu[i][0], 
    #                     global_rank, 
    #                     vis_dir
    #                 )

    # # 确保所有图片保存完毕
    # dist.barrier()

    # # 6. 聚合 PT 结果 (保持不变，这步数据量相对较小)
    # if is_master:
    #     print(f"\n✅ 图片保存完成，耗时: {time.time() - start_time:.1f}s")
    #     print("📊 正在收集推理矩阵...")

    # if is_master:
    #     print(f"🏁 DDP 全量图片保存任务完成！")
    #     print(f"⏱️ 总耗时: {time.time() - start_time:.2f}s")

    dist.destroy_process_group()

if __name__ == '__main__':
    # OpenCV 不需要 switch backend，它天然支持多进程
    main()