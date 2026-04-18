"""Batch full-forward RVQAE export from triangles to indices/real/imag (+optional ICFT)."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import multiprocessing as mp
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


_WORKER_PIPELINE = None
_WORKER_HELPERS = None


def _inject_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def _init_worker(repo_root: str, checkpoint_path: str, config_path: str, device: str) -> None:
    """Initialize one per-process full-forward pipeline bound to one device."""
    global _WORKER_PIPELINE, _WORKER_HELPERS

    repo_root = str(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import importlib

    pipeline_module = importlib.import_module("rvqae_pretrain.src.engine.pipeline")
    _WORKER_HELPERS = importlib.import_module("rvqae_pretrain.scripts.batch_infer_common")
    _WORKER_PIPELINE = pipeline_module.PolyRvqAePipeline(
        weight_path=str(checkpoint_path),
        config_path=str(config_path),
        device=str(device),
        precision="fp32",
    )


def _process_one_batch(
    start_index: int,
    tri_batch,
    meta_batch,
    nicft: int,
):
    """Process one triangle batch and return output records with global start index."""
    global _WORKER_PIPELINE, _WORKER_HELPERS
    if _WORKER_PIPELINE is None or _WORKER_HELPERS is None:
        raise RuntimeError("Worker pipeline is not initialized.")

    import torch

    imgs = _WORKER_PIPELINE.triangles_to_images(tri_batch)
    with torch.no_grad():
        outputs = _WORKER_PIPELINE.model(imgs, use_vq=True)

    indices_batch = outputs.indices.long()
    if indices_batch.ndim == 3:
        indices_batch = indices_batch.unsqueeze(1)
    indices_batch_u16 = _WORKER_HELPERS.to_uint16_indices(indices_batch, context="tri_forward")

    recon = outputs.recon_imgs.float()
    mag_valid = recon[:, 0, : _WORKER_PIPELINE.valid_h, : _WORKER_PIPELINE.valid_w]
    cos_valid = recon[:, 1, : _WORKER_PIPELINE.valid_h, : _WORKER_PIPELINE.valid_w]
    sin_valid = recon[:, 2, : _WORKER_PIPELINE.valid_h, : _WORKER_PIPELINE.valid_w]
    phase_valid = torch.atan2(sin_valid, cos_valid)
    raw_mag_valid = torch.expm1(mag_valid)
    real_batch = (raw_mag_valid * torch.cos(phase_valid)).float().cpu()
    imag_batch = (raw_mag_valid * torch.sin(phase_valid)).float().cpu()

    icft_batch = None
    if int(nicft) > 0:
        real_for_icft = real_batch.to(_WORKER_PIPELINE.device)
        imag_for_icft = imag_batch.to(_WORKER_PIPELINE.device)
        target_h = int(_WORKER_PIPELINE.codec.converter.U.shape[0])
        target_w = int(_WORKER_PIPELINE.codec.converter.U.shape[1])
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
        icft_batch = _WORKER_PIPELINE.codec.icft_2d(
            real_for_icft,
            f_uv_imag=imag_for_icft,
            spatial_size=int(nicft),
        ).float().cpu()

    records = []
    for sample_offset in range(len(tri_batch)):
        record = {
            "indices": indices_batch_u16[sample_offset].cpu(),
            "meta": torch.as_tensor(meta_batch[sample_offset], dtype=torch.float32).cpu(),
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

    if __package__ in {None, ""}:
        import importlib

        helpers = importlib.import_module("rvqae_pretrain.scripts.batch_infer_common")
    else:
        from . import batch_infer_common as helpers

    parser = argparse.ArgumentParser(description="Full-forward RVQAE export for triangle shards.")
    parser.add_argument("--tri_dir", type=str, required=True, help="Directory containing triangle+meta shard files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tri_forward shards.")
    parser.add_argument("--nicft", type=int, default=0, help="ICFT output size. <=0 disables ICFT export.")
    parser.add_argument("--batch_size", type=int, default=64, help="Per-GPU inference micro-batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Fallback single runtime device.")
    parser.add_argument("--gpus", type=str, default="", help="GPU id CSV for multi-GPU workers, e.g. `0,1,2,3`.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")
    if args.nicft < 0:
        raise ValueError(f"`nicft` must be >= 0, got {args.nicft}")

    worker_devices = helpers.parse_gpu_devices(args.gpus, fallback_device=args.device)
    checkpoint_path, config_path = helpers.resolve_model_paths(args.model_dir)
    tri_meta_pairs = helpers.resolve_tri_meta_pairs(args.tri_dir)
    total_samples = helpers.preflight_validate_tri_meta_pairs(tri_meta_pairs)
    print(
        f"[INFO] Tri/meta preflight passed: {len(tri_meta_pairs)} shard pairs, "
        f"total_samples={total_samples}, workers={worker_devices}"
    )

    helpers.clear_task_outputs(args.output_dir, task_prefix="tri_forward", manifest_name="tri_forward.manifest.json")

    executors: list[ProcessPoolExecutor] = []
    mp_context = mp.get_context("spawn")
    shard_records = []
    import torch

    try:
        for device in worker_devices:
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp_context,
                initializer=_init_worker,
                initargs=(str(project_root.parent), str(checkpoint_path), str(config_path), str(device)),
            )
            executors.append(executor)

        with tqdm(total=total_samples, desc="tri_forward total", unit="sample", position=0) as total_progress:
            for shard_index, pair in enumerate(tri_meta_pairs, start=1):
                tri_samples = helpers.load_torch_list(pair.tri_path)
                meta_samples = helpers.load_torch_list(pair.meta_path)
                if len(tri_samples) != len(meta_samples):
                    raise ValueError(
                        f"Triangle/meta sample count mismatch for {pair.tri_path.name} and {pair.meta_path.name}: "
                        f"{len(tri_samples)} vs {len(meta_samples)}"
                    )

                shard_size = len(tri_samples)
                shard_outputs: list[dict | None] = [None] * shard_size
                futures = []
                batch_index = 0
                for start in range(0, shard_size, args.batch_size):
                    end = min(start + args.batch_size, shard_size)
                    tri_batch = [np.asarray(sample, dtype=np.float32) for sample in tri_samples[start:end]]
                    meta_batch = meta_samples[start:end]
                    executor = executors[batch_index % len(executors)]
                    futures.append(executor.submit(_process_one_batch, int(start), tri_batch, meta_batch, int(args.nicft)))
                    batch_index += 1

                with tqdm(
                    total=shard_size,
                    desc=f"tri_forward shard {shard_index}/{len(tri_meta_pairs)}",
                    unit="sample",
                    leave=False,
                    position=1,
                ) as shard_progress:
                    for future in as_completed(futures):
                        record = future.result()
                        start_index = int(record["start_index"])
                        sample_count = int(record["sample_count"])
                        records = list(record["records"])
                        if len(records) != sample_count:
                            raise RuntimeError("Worker returned inconsistent tri_forward batch result length.")
                        shard_outputs[start_index : start_index + sample_count] = records
                        shard_progress.update(sample_count)
                        total_progress.update(sample_count)

                if any(item is None for item in shard_outputs):
                    raise RuntimeError(f"tri_forward shard has missing outputs: {pair.tri_path}")
                output_samples = [item for item in shard_outputs if item is not None]
                output_path = helpers.build_task_output_path(args.output_dir, "tri_forward", pair.tri_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(output_samples, output_path)
                shard_records.append(
                    {
                        "path": str(output_path.resolve()),
                        "sample_count": int(len(output_samples)),
                        "size_bytes": int(output_path.stat().st_size),
                        "input_tri_path": str(pair.tri_path),
                        "input_meta_path": str(pair.meta_path),
                    }
                )
                print(
                    f"[INFO] Forwarded shard {shard_index}/{len(tri_meta_pairs)}: {pair.tri_path.name} "
                    f"-> {output_path.name} ({len(output_samples)} samples)"
                )
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=False)

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
            "worker_devices": worker_devices,
            "nicft": int(args.nicft),
            "project_root": str(project_root),
        },
        shard_records=shard_records,
    )

    print(f"[INFO] Saved tri_forward shards to: {Path(args.output_dir).expanduser().resolve()}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Exported samples: {sum(int(item['sample_count']) for item in shard_records)}")


if __name__ == "__main__":
    main()
