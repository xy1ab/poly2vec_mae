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
import argparse
from datetime import datetime
import re
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import segmentation_models_pytorch as smp

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CURRENT_DIR.parent
    _REPO_ROOT = _PROJECT_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib
    helpers = importlib.import_module("rvqae_pretrain.scripts.batch_infer_common")
    pipeline_module = importlib.import_module("rvqae_pretrain.src.engine.pipeline")
    ensure_cuda_runtime_libs = importlib.import_module(
        "rvqae_pretrain.scripts.runtime_bootstrap"
    ).ensure_cuda_runtime_libs
else:
    from .runtime_bootstrap import ensure_cuda_runtime_libs
    from . import batch_infer_common as helpers
    from ..src.engine import pipeline as pipeline_module




def post_process_results(sigmoid_output, original_sizes, threshold=0.5):
    """
    sigmoid_output: [B, 1, 256, 256] 概率值 (0-1)
    original_sizes: list of (h, w)
    """
    final_masks = []
    import torch.nn.functional as F
    
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
            "request_position": int(start_index + sample_offset),
            "sample_index": int(batch_sample_indices[sample_offset]),
            "meta_data": batch_meta[sample_offset],
            "pred_bin": pred_bin[sample_offset]
        }
        records.append(record)

    return {
        "sample_count": int(len(records)),
        "records": records,
    }

def _parse_sample_index_file(path: str | Path) -> list[int]:
    index_path = Path(path).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"`sample_index_file` does not exist: {index_path}")

    values: list[int] = []
    with index_path.open("r", encoding="utf-8") as fp:
        for line_number, raw_line in enumerate(fp, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            for token in re.split(r"[\s,]+", line):
                if token:
                    try:
                        values.append(int(token))
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid sample index `{token}` at {index_path}:{line_number}"
                        ) from exc
    if not values:
        raise ValueError(f"`sample_index_file` contains no sample indices: {index_path}")
    return values


def research_index_require(sample_index_list, shard_dir, helpers, device, show_progress=True):
    """Build a GPU index pool from all shards, then return requested samples in input order."""
    import torch

    required_indices = [int(item) for item in sample_index_list]
    ind_shards = helpers.resolve_ind_shards(shard_dir)

    index_pool = {}
    for shard_path in tqdm(ind_shards, desc="Loading index pool", unit="shard", disable=not show_progress):
        samples = helpers.load_torch_list(shard_path)
        for local_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(f"Index shard sample must be dict: {shard_path}#{local_index}")
            if "sample_index" not in sample:
                raise KeyError(f"Index shard sample missing `sample_index`: {shard_path}#{local_index}")
            if "indices" not in sample or "meta" not in sample:
                raise KeyError(f"Index shard sample missing `indices`/`meta`: {shard_path}#{local_index}")

            sample_index = int(sample["sample_index"])
            if sample_index in index_pool:
                raise ValueError(f"Duplicate sample_index found in index pool: {sample_index}")

            index_pool[sample_index] = {
                "sample_index": sample_index,
                "indices": helpers.normalize_indices_grid(sample["indices"]).to(device=device, dtype=torch.long),
                "meta": sample["meta"],
            }

    missing = [sample_index for sample_index in required_indices if sample_index not in index_pool]
    if missing:
        preview = missing[:20]
        raise KeyError(f"Requested sample_index not found in index pool: {preview}")

    return [index_pool[sample_index] for sample_index in required_indices]

class demo():
    def __init__(self, args):
        self.args = args
        self.rank, self.local_rank, self.world_size, self.device = self._setup_distributed()

        is_rank0 = self.rank == 0

        self.decoder_path, self.quantizer_path, self.config_path = helpers.resolve_decode_paths(self.args.model_dir)
        run_dir_obj = [None]
        if is_rank0:
            print(f"world_size={self.world_size}, device={self.device}, output_dir={self.args.output_dir}")
        run_dir_obj[0] = str(self.args.output_dir)
        dist.broadcast_object_list(run_dir_obj, src=0)
        self.run_dir = Path(run_dir_obj[0])

        dist.barrier(device_ids=[torch.device(self.device).index])
        self.pipeline, self.ds_model = self.init()

    def _setup_distributed(self):
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

    def init(self):
        ## 加载模型
        pipeline = pipeline_module.PolyRvqDecodePipeline(
            decoder_path=str(self.decoder_path),
            quantizer_path=str(self.quantizer_path),
            config_path=str(self.config_path),
            device=str(self.device),
            precision="fp32",
        )

        ds_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=1,
            activation=None
        ).to(self.device)
        
        ds_model.load_state_dict(torch.load(self.args.downstream_model_path, map_location=self.device))
        ds_model.eval()

        return pipeline, ds_model


    def eval(self, sample_index_list):
        is_rank0 = self.rank == 0
        requested_samples = research_index_require(
            sample_index_list=sample_index_list,
            shard_dir=self.args.ind_dir,
            helpers=helpers,
            device=self.device,
            show_progress=is_rank0,
        )
        requested_count = len(requested_samples)
        
        if is_rank0:
            total_progress = tqdm(total=requested_count, desc="demo total", unit="sample", position=0)
        else:
            total_progress = None

        rank_outputs = []
        rank_sample_count = 0
        num_batches = (requested_count + self.args.batch_size - 1) // self.args.batch_size
        for round_start in range(0, num_batches, self.world_size):
            batch_index = round_start + self.rank
            if batch_index < num_batches:
                start = batch_index * self.args.batch_size
                end = min(start + self.args.batch_size, requested_count)
                local_payload = _process_one_batch(
                    pipeline=self.pipeline,
                    ds_model=self.ds_model,
                    helpers=helpers,
                    start_index=int(start),
                    batch_samples=requested_samples[start:end],
                    resolution=int(self.args.resolution),
                )
                records = list(local_payload["records"])
                if len(records) != int(local_payload["sample_count"]):
                    raise RuntimeError("Local demo batch result length is inconsistent.")
                rank_outputs.extend(records)
                rank_sample_count += len(records)

            if is_rank0:
                round_sample_count = 0
                for round_rank in range(self.world_size):
                    round_batch_index = round_start + round_rank
                    if round_batch_index >= num_batches:
                        continue
                    start_index = round_batch_index * self.args.batch_size
                    end_index = min(start_index + self.args.batch_size, requested_count)
                    round_sample_count += end_index - start_index
                total_progress.update(round_sample_count)

        output_path = self.run_dir / f"demo_rank{self.rank:03d}.pt"
        torch.save(rank_outputs, output_path)
        print(f"[RANK {self.rank}] Saved demo part: {output_path.name} ({rank_sample_count} samples)", flush=True)

        # dist.barrier(device_ids=[torch.device(self.device).index])

        # if is_rank0:
        #     total_progress.close()
        #     part_records = []
        #     for part_rank in range(self.world_size):
        #         part_path = self.run_dir / f"demo_rank{part_rank:03d}.pt"
        #         if not part_path.is_file():
        #             raise FileNotFoundError(f"Missing rank output part: {part_path}")
        #         part_samples = 0
        #         for batch_index in range(part_rank, num_batches, self.world_size):
        #             start = batch_index * args.batch_size
        #             end = min(start + args.batch_size, requested_count)
        #             part_samples += end - start
        #         part_records.append(
        #             {
        #                 "path": str(part_path.resolve()),
        #                 "rank": int(part_rank),
        #                 "sample_count": int(part_samples),
        #                 "size_bytes": int(part_path.stat().st_size),
        #             }
        #         )
            # manifest_path = helpers.write_task_manifest(
            #     output_dir=self.run_dir,
            #     manifest_name="demo.manifest.json",
            #     metadata={
            #         "created_at": datetime.now().isoformat(timespec="seconds"),
            #         "ind_dir": str(Path(args.ind_dir).expanduser().resolve()),
            #         # "sample_index_file": str(Path(args.sample_index_file).expanduser().resolve()),
            #         "model_dir": str(Path(args.model_dir).expanduser().resolve()),
            #         "downstream_model_path": str(Path(args.downstream_model_path).expanduser().resolve()),
            #         "decoder_path": str(decoder_path),
            #         "quantizer_path": str(quantizer_path),
            #         "config_path": str(config_path),
            #         "batch_size": int(args.batch_size),
            #         "resolution": int(args.resolution),
            #         "world_size": int(world_size),
            #         "requested_sample_indices": [int(item) for item in sample_index_list],
            #         "output_mode": "rank_part",
            #     },
            #     shard_records=part_records,
            # )
            # print(f"[INFO] Saved demo results to: {self.run_dir}")
            # print(f"[INFO] Manifest: {manifest_path}")

        # dist.barrier(device_ids=[torch.device(self.device).index])

def main():
    ensure_cuda_runtime_libs()

    parser = argparse.ArgumentParser(description="Retrieve RVQ indices by sample_index and run RVQAE+UNet demo inference.")
    parser.add_argument("--ind_dir", type=str, required=True, help="Directory containing tri2ind shard files.")
    # parser.add_argument("--sample_index_file", type=str, required=True, help="Text file containing requested sample_index values.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--downstream_model_path", type=str, required=True, help="Path to downstream UNet checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory for timestamped demo results.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.resolution <= 0:
        raise ValueError(f"`resolution` must be > 0, got {args.resolution}")
    
    demo_test = demo(args)
    demo_test.init()

    ## get index request
    sample_index_list = [i for i in range(100)]#_parse_sample_index_file(args.sample_index_file)

    demo_test.eval(sample_index_list=sample_index_list)


    dist.destroy_process_group()

if __name__ == '__main__':
    main()
