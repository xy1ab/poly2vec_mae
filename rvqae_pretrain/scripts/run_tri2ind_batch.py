"""Batch encode triangle shards to RVQ indices with paired meta export."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CURRENT_DIR.parent
    _REPO_ROOT = _PROJECT_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib

    ensure_cuda_runtime_libs = importlib.import_module(
        "rvqae_pretrain.scripts.runtime_bootstrap"
    ).ensure_cuda_runtime_libs
else:
    from .runtime_bootstrap import ensure_cuda_runtime_libs


def _inject_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


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
    stem_path = output_root / f"tri2ind_{in_path.stem}.pt"
    return stem_path.with_name(f"{stem_path.stem}_rank{rank:03d}{stem_path.suffix}")


def _process_one_batch(pipeline, helpers, start_index: int, tri_batch, meta_batch):
    """Process one triangle batch and return output records with global start index."""
    import torch

    indices_batch = pipeline.quantize_triangles(tri_batch)
    if indices_batch.ndim == 3:
        indices_batch = indices_batch.unsqueeze(1)
    indices_batch = helpers.to_uint16_indices(indices_batch, context="tri2ind")

    records = []
    for sample_offset in range(len(tri_batch)):
        records.append(
            {
                "sample_index": int(start_index + sample_offset),
                "indices": indices_batch[sample_offset].cpu(),
                "meta": torch.as_tensor(meta_batch[sample_offset], dtype=torch.float32).cpu(),
            }
        )
    return {
        "start_index": int(start_index),
        "sample_count": int(len(records)),
        "records": records,
    }


def main() -> None:
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()
    import torch
    import torch.distributed as dist

    if __package__ in {None, ""}:
        import importlib

        helpers = importlib.import_module("rvqae_pretrain.scripts.batch_infer_common")
        pipeline_module = importlib.import_module("rvqae_pretrain.src.engine.pipeline")
    else:
        from . import batch_infer_common as helpers
        from ..src.engine import pipeline as pipeline_module

    parser = argparse.ArgumentParser(description="Encode triangle shards to RVQ indices (with meta).")
    parser.add_argument("--tri_dir", type=str, required=True, help="Directory containing triangle+meta shard files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tri2ind shards.")
    parser.add_argument("--batch_size", type=int, default=512, help="Per-rank inference micro-batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Runtime device type, usually `cuda`.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")

    rank = 0
    world_size = 1
    shard_records = []
    total_progress = None

    try:
        rank, local_rank, world_size, device = _setup_distributed(args, helpers)
        is_rank0 = rank == 0

        checkpoint_path, config_path = helpers.resolve_model_paths(args.model_dir)
        tri_meta_pairs = helpers.resolve_tri_meta_pairs(args.tri_dir)

        total_samples = 0
        if is_rank0:
            total_samples = helpers.preflight_validate_tri_meta_pairs(tri_meta_pairs)
            print(
                f"[INFO] Tri/meta preflight passed: {len(tri_meta_pairs)} shard pairs, "
                f"total_samples={total_samples}, world_size={world_size}, device={device}"
            )
            helpers.clear_task_outputs(
                args.output_dir,
                task_prefix="tri2ind",
                manifest_name="tri2ind.manifest.json",
            )

        dist.barrier(device_ids=[torch.device(device).index])

        pipeline = pipeline_module.PolyRvqAePipeline(
            weight_path=str(checkpoint_path),
            config_path=str(config_path),
            device=str(device),
            precision="fp32",
        )

        if is_rank0:
            total_progress = tqdm(total=total_samples, desc="tri2ind total", unit="sample", position=0)

        for shard_index, pair in enumerate(tri_meta_pairs, start=1):
            if is_rank0:
                print(f"[INFO] Loading shard {shard_index}/{len(tri_meta_pairs)}: {pair.tri_path.name}")
            tri_samples = helpers.load_torch_list(pair.tri_path)
            meta_samples = helpers.load_torch_list(pair.meta_path)
            if len(tri_samples) != len(meta_samples):
                raise ValueError(
                    f"Triangle/meta sample count mismatch for {pair.tri_path.name} and {pair.meta_path.name}: "
                    f"{len(tri_samples)} vs {len(meta_samples)}"
            )

            shard_size = len(tri_samples)
            rank_outputs = []
            rank_sample_count = 0
            shard_progress = (
                tqdm(
                    total=shard_size,
                    desc=f"tri2ind shard {shard_index}/{len(tri_meta_pairs)}",
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
                    f"[INFO] Processing shard {shard_index}/{len(tri_meta_pairs)}: "
                    f"samples={shard_size}, batches={num_batches}, world_size={world_size}, output_mode=rank_part"
                )
            for round_start in range(0, num_batches, world_size):
                batch_index = round_start + rank

                if batch_index < num_batches:
                    start = batch_index * args.batch_size
                    end = min(start + args.batch_size, shard_size)
                    tri_batch = [np.asarray(sample, dtype=np.float32) for sample in tri_samples[start:end]]
                    meta_batch = meta_samples[start:end]
                    local_payload = _process_one_batch(
                        pipeline=pipeline,
                        helpers=helpers,
                        start_index=int(start),
                        tri_batch=tri_batch,
                        meta_batch=meta_batch,
                    )
                    records = list(local_payload["records"])
                    if len(records) != int(local_payload["sample_count"]):
                        raise RuntimeError("Local tri2ind batch result length is inconsistent.")
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

            output_path = _build_rank_output_path(args.output_dir, pair.tri_path, rank)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"[RANK {rank}] Saving part shard {shard_index}/{len(tri_meta_pairs)}: "
                f"{output_path.name} ({rank_sample_count} samples)",
                flush=True,
            )
            torch.save(rank_outputs, output_path)

            dist.barrier(device_ids=[torch.device(device).index])

            if is_rank0:
                for part_rank in range(world_size):
                    part_path = _build_rank_output_path(args.output_dir, pair.tri_path, part_rank)
                    if not part_path.is_file():
                        raise FileNotFoundError(f"Missing rank output part: {part_path}")
                    part_samples = 0
                    for batch_index in range(part_rank, num_batches, world_size):
                        start = batch_index * args.batch_size
                        end = min(start + args.batch_size, shard_size)
                        part_samples += end - start
                    shard_records.append(
                        {
                            "path": str(part_path.resolve()),
                            "rank": int(part_rank),
                            "sample_count": int(part_samples),
                            "size_bytes": int(part_path.stat().st_size),
                            "input_tri_path": str(pair.tri_path),
                            "input_meta_path": str(pair.meta_path),
                        }
                    )
                print(
                    f"[INFO] Encoded shard {shard_index}/{len(tri_meta_pairs)}: {pair.tri_path.name} "
                    f"-> {world_size} rank part files"
                )
                shard_progress.close()

            dist.barrier(device_ids=[torch.device(device).index])

        if is_rank0:
            total_progress.close()
            manifest_path = helpers.write_task_manifest(
                output_dir=args.output_dir,
                manifest_name="tri2ind.manifest.json",
                metadata={
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "tri_dir": str(Path(args.tri_dir).expanduser().resolve()),
                    "model_dir": str(Path(args.model_dir).expanduser().resolve()),
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                    "batch_size": int(args.batch_size),
                    "device": str(args.device),
                    "world_size": int(world_size),
                    "output_mode": "rank_part",
                    "project_root": str(project_root),
                },
                shard_records=shard_records,
            )
            print(f"[INFO] Saved tri2ind shards to: {Path(args.output_dir).expanduser().resolve()}")
            print(f"[INFO] Manifest: {manifest_path}")
            print(f"[INFO] Exported samples: {sum(int(item['sample_count']) for item in shard_records)}")
    finally:
        if total_progress is not None:
            total_progress.close()

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
