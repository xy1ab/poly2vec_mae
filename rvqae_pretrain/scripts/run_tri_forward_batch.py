"""Batch full-forward RVQAE export from triangles to indices/real/imag (+optional ICFT)."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

import matplotlib.path as mpltPath
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


def _str2bool(value):
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y"}:
        return True
    if token in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


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


def rasterize_triangles(tris_batch, sample_nicfts) -> list[np.ndarray]:
    results = []
    for tris, nicft in zip(tris_batch, sample_nicfts):
        x = np.linspace(-1.0, 1.0, nicft)
        y = np.linspace(1.0, -1.0, nicft)
        x_grid, y_grid = np.meshgrid(x, y)
        points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        mask = np.zeros(nicft * nicft, dtype=bool)
        for tri in tris:
            path = mpltPath.Path(tri)
            mask = mask | path.contains_points(points)
        results.append(mask.reshape(nicft, nicft))
    return results


def _process_one_batch(
    pipeline,
    helpers,
    start_index: int,
    tri_batch,
    meta_batch,
    nicft: int,
    resolution: int,
    rec_flag: bool,
):
    """Process one triangle batch and return output records with global start index."""
    import torch

    imgs = pipeline.triangles_to_images(tri_batch)

    with torch.no_grad():
        outputs = pipeline.model(imgs, use_vq=True)

    indices_batch = outputs.indices.long()
    if indices_batch.ndim == 3:
        indices_batch = indices_batch.unsqueeze(1)
    indices_batch_u16 = helpers.to_uint16_indices(indices_batch, context="tri_forward")

    recon = outputs.recon_imgs.float()
    mag_valid = recon[:, 0, : pipeline.valid_h, : pipeline.valid_w]
    cos_valid = recon[:, 1, : pipeline.valid_h, : pipeline.valid_w]
    sin_valid = recon[:, 2, : pipeline.valid_h, : pipeline.valid_w]
    phase_valid = torch.atan2(sin_valid, cos_valid)
    raw_mag_valid = torch.expm1(mag_valid)
    real_batch = (raw_mag_valid * torch.cos(phase_valid)).float().cpu()
    imag_batch = (raw_mag_valid * torch.sin(phase_valid)).float().cpu()

    n_samples = len(tri_batch)
    sample_nicfts = []
    if int(resolution) > 0:
        for metadata in meta_batch:
            dL = float(metadata[2])
            sample_nicfts.append(int(np.ceil(dL * 118000.0 / float(resolution))))
    else:
        sample_nicfts = [int(nicft)] * n_samples

    rec_label = rasterize_triangles(tri_batch, sample_nicfts) if rec_flag else None

    icft_batch = [None] * n_samples
    target_h = int(pipeline.codec.converter.U.shape[0])
    target_w = int(pipeline.codec.converter.U.shape[1])
    device = pipeline.device

    for i in range(n_samples):
        if sample_nicfts[i] <= 0:
            continue
        real_for_icft = real_batch[i].to(device)
        imag_for_icft = imag_batch[i].to(device)

        if real_for_icft.shape[0] > target_h or real_for_icft.shape[1] > target_w:
            raise ValueError(f"Decoded grid {real_for_icft.shape} exceeds full grid ({target_h}, {target_w})")

        if real_for_icft.shape[0] != target_h or real_for_icft.shape[1] != target_w:
            padded_real = torch.zeros((target_h, target_w), dtype=real_for_icft.dtype, device=device)
            padded_imag = torch.zeros((target_h, target_w), dtype=imag_for_icft.dtype, device=device)
            padded_real[: real_for_icft.shape[0], : real_for_icft.shape[1]] = real_for_icft
            padded_imag[: imag_for_icft.shape[0], : imag_for_icft.shape[1]] = imag_for_icft
            real_for_icft, imag_for_icft = padded_real, padded_imag

        icft_res = pipeline.codec.icft_2d(
            f_uv_real=real_for_icft.unsqueeze(0),
            f_uv_imag=imag_for_icft.unsqueeze(0),
            spatial_size=sample_nicfts[i],
        ).float().cpu().squeeze(0)
        icft_batch[i] = icft_res

    records = []
    for sample_offset in range(len(tri_batch)):
        record = {
            "sample_index": int(start_index + sample_offset),
            "indices": indices_batch_u16[sample_offset].cpu(),
            "meta": torch.as_tensor(meta_batch[sample_offset], dtype=torch.float32).cpu().numpy(),
            "real": real_batch[sample_offset].float().cpu().numpy(),
            "imag": imag_batch[sample_offset].float().cpu().numpy(),
        }
        if icft_batch[sample_offset] is not None:
            record["icft"] = icft_batch[sample_offset].float().cpu().numpy()
        if rec_label is not None:
            record["rec_label"] = rec_label[sample_offset]
        records.append(record)

    return {
        "start_index": int(start_index),
        "sample_count": int(len(records)),
        "records": records,
    }


def _build_rank_output_path(output_dir: str, input_shard_path: Path, rank: int) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    in_path = Path(input_shard_path).expanduser().resolve()
    stem_path = output_root / f"tri_forward_{in_path.stem}.pt"
    return stem_path.with_name(f"{stem_path.stem}_rank{rank:03d}{stem_path.suffix}")


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

    num_gpus = torch.cuda.device_count()
    default_gpus = ",".join(map(str, range(num_gpus)))
    parser = argparse.ArgumentParser(description="Full-forward RVQAE export for triangle shards.")
    parser.add_argument("--tri_dir", type=str, required=True, help="Directory containing triangle+meta shard files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tri_forward shards.")
    parser.add_argument("--nicft", type=int, default=256, help="ICFT output size. <=0 disables ICFT export.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=0,
        help="Spatial resolution (meter). If >0, per-sample nicft is derived from metadata.",
    )
    parser.add_argument("--rec_flag", type=_str2bool, default=False, help="Save rasterized triangle mask.")
    parser.add_argument("--batch_size", type=int, default=512, help="Per-rank inference micro-batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Runtime device type, usually `cuda`.")
    parser.add_argument(
        "--gpus",
        type=str,
        default=default_gpus,
        help=f"GPU id CSV for this node. Default: all visible ({default_gpus}).",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.nicft < 0:
        raise ValueError(f"`nicft` must be >= 0, got {args.nicft}")
    if args.resolution < 0:
        raise ValueError(f"`resolution` must be >= 0, got {args.resolution}")

    rank = 0
    world_size = 1
    distributed = False
    shard_records = []
    total_progress = None

    try:
        rank, local_rank, world_size, distributed, device = _setup_distributed(args, helpers)
        is_rank0 = rank == 0

        checkpoint_path, config_path = helpers.resolve_model_paths(args.model_dir)
        tri_meta_pairs = helpers.resolve_tri_meta_pairs(args.tri_dir)

        total_samples = 0
        if is_rank0:
            print("[INFO] Preflight started.", flush=True)
            total_samples = helpers.preflight_validate_tri_meta_pairs(tri_meta_pairs)
            print(
                f"[INFO] Tri/meta preflight passed: {len(tri_meta_pairs)} shard pairs, "
                f"total_samples={total_samples}, world_size={world_size}, device={device}",
                flush=True,
            )
            helpers.clear_task_outputs(
                args.output_dir,
                task_prefix="tri_forward",
                manifest_name="tri_forward.manifest.json",
            )
        total_samples = int(_broadcast_object(total_samples, distributed, rank))
        _barrier(distributed, device)

        if is_rank0:
            print("[INFO] Pipeline loading started.", flush=True)
        pipeline = pipeline_module.PolyRvqAePipeline(
            weight_path=str(checkpoint_path),
            config_path=str(config_path),
            device=str(device),
            precision="fp32",
        )
        if is_rank0:
            print("[INFO] Pipeline loading finished.", flush=True)

        if is_rank0:
            total_progress = tqdm(total=total_samples, desc="tri_forward total", unit="sample", position=0)

        for shard_index, pair in enumerate(tri_meta_pairs, start=1):
            if is_rank0:
                print(f"[INFO] Loading shard {shard_index}/{len(tri_meta_pairs)}: {pair.tri_path.name}", flush=True)
            tri_samples = helpers.load_torch_list(pair.tri_path)
            meta_samples = helpers.load_torch_list(pair.meta_path)
            if len(tri_samples) != len(meta_samples):
                raise ValueError(
                    f"Triangle/meta sample count mismatch for {pair.tri_path.name} and {pair.meta_path.name}: "
                    f"{len(tri_samples)} vs {len(meta_samples)}"
                )

            shard_size = len(tri_samples)
            shard_progress = (
                tqdm(
                    total=shard_size,
                    desc=f"tri_forward shard {shard_index}/{len(tri_meta_pairs)}",
                    unit="sample",
                    leave=False,
                    position=1,
                )
                if is_rank0
                else None
            )

            num_batches = (shard_size + args.batch_size - 1) // args.batch_size
            rank_outputs = []
            rank_sample_count = 0
            if is_rank0:
                print(
                    f"[INFO] Processing shard {shard_index}/{len(tri_meta_pairs)}: "
                    f"samples={shard_size}, batches={num_batches}, world_size={world_size}, "
                    f"nicft={int(args.nicft)}, resolution={int(args.resolution)}, "
                    f"rec_flag={bool(args.rec_flag)}, output_mode=rank_part",
                    flush=True,
                )
            for round_start in range(0, num_batches, world_size):
                batch_index = round_start + rank

                if batch_index < num_batches:
                    start = batch_index * args.batch_size
                    end = min(start + args.batch_size, shard_size)
                    if round_start == 0:
                        print(
                            f"[RANK {rank}] first batch started: batch_index={batch_index}, "
                            f"samples={end - start}, triangles={sum(len(sample) for sample in tri_samples[start:end])}",
                            flush=True,
                        )
                    tri_batch = [np.asarray(sample, dtype=np.float32) for sample in tri_samples[start:end]]
                    meta_batch = meta_samples[start:end]
                    local_payload = _process_one_batch(
                        pipeline=pipeline,
                        helpers=helpers,
                        start_index=int(start),
                        tri_batch=tri_batch,
                        meta_batch=meta_batch,
                        nicft=int(args.nicft),
                        resolution=int(args.resolution),
                        rec_flag=bool(args.rec_flag),
                    )
                    records = list(local_payload["records"])
                    if len(records) != int(local_payload["sample_count"]):
                        raise RuntimeError("Local tri_forward batch result length is inconsistent.")
                    rank_outputs.extend(records)
                    rank_sample_count += len(records)
                    if round_start == 0:
                        print(f"[RANK {rank}] first batch finished: batch_index={batch_index}", flush=True)

                if is_rank0:
                    round_sample_count = 0
                    for round_rank in range(world_size):
                        round_batch_index = round_start + round_rank
                        if round_batch_index >= num_batches:
                            continue
                        round_start_index = round_batch_index * args.batch_size
                        round_end_index = min(round_start_index + args.batch_size, shard_size)
                        round_sample_count += round_end_index - round_start_index
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

            if distributed:
                _barrier(distributed, device)

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
                    f"[INFO] Forwarded shard {shard_index}/{len(tri_meta_pairs)}: {pair.tri_path.name} "
                    f"-> {world_size} rank part files"
                )
                shard_progress.close()

            if distributed:
                _barrier(distributed, device)

        if is_rank0:
            total_progress.close()
            manifest_path = helpers.write_task_manifest(
                output_dir=args.output_dir,
                manifest_name="tri_forward.manifest.json",
                metadata={
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "tri_dir": str(Path(args.tri_dir).expanduser().resolve()),
                    "model_dir": str(Path(args.model_dir).expanduser().resolve()),
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                    "batch_size": int(args.batch_size),
                    "device": str(args.device),
                    "gpus": str(args.gpus),
                    "world_size": int(world_size),
                    "output_mode": "rank_part",
                    "nicft": int(args.nicft),
                    "resolution": int(args.resolution),
                    "project_root": str(project_root),
                    "rec_flag": bool(args.rec_flag),
                },
                shard_records=shard_records,
            )
            print(f"[INFO] Saved tri_forward shards to: {Path(args.output_dir).expanduser().resolve()}")
            print(f"[INFO] Manifest: {manifest_path}")
            print(f"[INFO] Exported samples: {sum(int(item['sample_count']) for item in shard_records)}")
    finally:
        if total_progress is not None:
            total_progress.close()
        if distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
