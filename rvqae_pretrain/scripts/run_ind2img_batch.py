"""Batch decode RVQ index shards to real/imag frequency maps (and optional ICFT)."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

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


def _broadcast_object(value, enabled: bool, rank: int):
    import torch.distributed as dist

    if not enabled:
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _setup_distributed(args, helpers):
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if not distributed:
        if str(args.gpus).strip():
            worker_devices = helpers.parse_gpu_devices(args.gpus, fallback_device=args.device)
            device = worker_devices[0]
        else:
            device = helpers.resolve_runtime_device(args.device)
        return rank, local_rank, world_size, distributed, device

    requested_device = str(args.device).strip().lower()
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Torchrun mode requested CUDA, but `torch.cuda.is_available()` is False.")
        if str(args.gpus).strip():
            worker_devices = helpers.parse_gpu_devices(args.gpus, fallback_device=args.device)
            if local_rank >= len(worker_devices):
                raise ValueError(
                    f"`--gpus` does not cover LOCAL_RANK={local_rank}: "
                    f"parsed devices={worker_devices}"
                )
            device = worker_devices[local_rank]
        else:
            device = f"cuda:{local_rank}"
        torch.cuda.set_device(torch.device(device))
        backend = "nccl"
    else:
        device = "cpu"
        backend = "gloo"

    if backend == "nccl":
        dist.init_process_group(backend=backend, device_id=torch.device(device))
    else:
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size, distributed, device


def _barrier(enabled: bool, device: str) -> None:
    if not enabled:
        return
    import torch
    import torch.distributed as dist

    if str(device).startswith("cuda"):
        dist.barrier(device_ids=[torch.device(device).index])
    else:
        dist.barrier()


def _build_rank_output_path(output_dir: str, input_shard_path: Path, rank: int) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    in_path = Path(input_shard_path).expanduser().resolve()
    stem_path = output_root / f"ind2img_{in_path.stem}.pt"
    return stem_path.with_name(f"{stem_path.stem}_rank{rank:03d}{stem_path.suffix}")


def _process_one_batch(pipeline, helpers, start_index: int, batch_samples, nicft: int):
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
    indices_batch_u16 = helpers.to_uint16_indices(indices_batch, context="ind2img")
    real_batch, imag_batch = pipeline.decode_indices(indices_batch)

    icft_batch = None
    if int(nicft) > 0:
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
            real_for_icft,
            f_uv_imag=imag_for_icft,
            spatial_size=int(nicft),
        ).float().cpu()

    records = []
    for sample_offset in range(len(batch_samples)):
        record = {
            "sample_index": int(batch_sample_indices[sample_offset]),
            "indices": indices_batch_u16[sample_offset].cpu(),
            "meta": torch.as_tensor(batch_meta[sample_offset], dtype=torch.float32).cpu(),
            "real": real_batch[sample_offset].float().cpu(),
            "imag": imag_batch[sample_offset].float().cpu(),
        }
        if icft_batch is not None:
            record["icft"] = icft_batch[sample_offset].float().cpu()
        records.append(record)

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

    parser = argparse.ArgumentParser(description="Decode index shards to real/imag maps.")
    parser.add_argument("--ind_dir", type=str, required=True, help="Directory containing tri2ind shard files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for ind2img shards.")
    parser.add_argument("--nicft", type=int, default=0, help="ICFT output size. <=0 disables ICFT export.")
    parser.add_argument("--batch_size", type=int, default=64, help="Per-rank inference micro-batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Runtime device type, usually `cuda`.")
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="GPU id CSV for this node.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.nicft < 0:
        raise ValueError(f"`nicft` must be >= 0, got {args.nicft}")

    rank = 0
    world_size = 1
    distributed = False
    shard_records = []
    total_progress = None

    try:
        rank, local_rank, world_size, distributed, device = _setup_distributed(args, helpers)
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
            helpers.clear_task_outputs(
                args.output_dir,
                task_prefix="ind2img",
                manifest_name="ind2img.manifest.json",
            )
        total_samples = int(_broadcast_object(total_samples, distributed, rank))
        _barrier(distributed, device)

        pipeline = pipeline_module.PolyRvqDecodePipeline(
            decoder_path=str(decoder_path),
            quantizer_path=str(quantizer_path),
            config_path=str(config_path),
            device=str(device),
            precision="fp32",
        )

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
                        helpers=helpers,
                        start_index=int(start),
                        batch_samples=batch_samples,
                        nicft=int(args.nicft),
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

            _barrier(distributed, device)

            if is_rank0:
                for part_rank in range(world_size):
                    part_path = _build_rank_output_path(args.output_dir, ind_path, part_rank)
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
                            "input_ind_path": str(Path(ind_path).expanduser().resolve()),
                        }
                    )
                print(
                    f"[INFO] Decoded shard {shard_index}/{len(ind_shards)}: {Path(ind_path).name} "
                    f"-> {world_size} rank part files"
                )
                shard_progress.close()

            _barrier(distributed, device)

        if is_rank0:
            total_progress.close()
            manifest_path = helpers.write_task_manifest(
                output_dir=args.output_dir,
                manifest_name="ind2img.manifest.json",
                metadata={
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "ind_dir": str(Path(args.ind_dir).expanduser().resolve()),
                    "model_dir": str(Path(args.model_dir).expanduser().resolve()),
                    "decoder_path": str(decoder_path),
                    "quantizer_path": str(quantizer_path),
                    "config_path": str(config_path),
                    "batch_size": int(args.batch_size),
                    "device": str(args.device),
                    "gpus": str(args.gpus),
                    "world_size": int(world_size),
                    "output_mode": "rank_part",
                    "nicft": int(args.nicft),
                    "project_root": str(project_root),
                },
                shard_records=shard_records,
            )
            print(f"[INFO] Saved ind2img shards to: {Path(args.output_dir).expanduser().resolve()}")
            print(f"[INFO] Manifest: {manifest_path}")
            print(f"[INFO] Exported samples: {sum(int(item['sample_count']) for item in shard_records)}")
    finally:
        if total_progress is not None:
            total_progress.close()
        if distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
