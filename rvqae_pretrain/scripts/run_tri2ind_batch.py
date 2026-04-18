"""Batch encode triangle shards to RVQ indices with paired meta export."""

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
    """Initialize one per-process inference pipeline bound to one device."""
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
):
    """Process one triangle batch and return output records with global start index."""
    global _WORKER_PIPELINE, _WORKER_HELPERS
    if _WORKER_PIPELINE is None or _WORKER_HELPERS is None:
        raise RuntimeError("Worker pipeline is not initialized.")

    import torch

    indices_batch = _WORKER_PIPELINE.quantize_triangles(tri_batch)
    if indices_batch.ndim == 3:
        indices_batch = indices_batch.unsqueeze(1)
    indices_batch = _WORKER_HELPERS.to_uint16_indices(indices_batch, context="tri2ind")

    records = []
    for sample_offset in range(len(tri_batch)):
        records.append(
            {
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

    if __package__ in {None, ""}:
        import importlib

        helpers = importlib.import_module("rvqae_pretrain.scripts.batch_infer_common")
    else:
        from . import batch_infer_common as helpers

    parser = argparse.ArgumentParser(description="Encode triangle shards to RVQ indices (with meta).")
    parser.add_argument("--tri_dir", type=str, required=True, help="Directory containing triangle+meta shard files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tri2ind shards.")
    parser.add_argument("--batch_size", type=int, default=64, help="Per-GPU inference micro-batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Fallback single runtime device.")
    parser.add_argument("--gpus", type=str, default="", help="GPU id CSV for multi-GPU workers, e.g. `0,1,2,3`.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"`batch_size` must be > 0, got {args.batch_size}")

    worker_devices = helpers.parse_gpu_devices(args.gpus, fallback_device=args.device)
    checkpoint_path, config_path = helpers.resolve_model_paths(args.model_dir)
    tri_meta_pairs = helpers.resolve_tri_meta_pairs(args.tri_dir)
    total_samples = helpers.preflight_validate_tri_meta_pairs(tri_meta_pairs)
    print(
        f"[INFO] Tri/meta preflight passed: {len(tri_meta_pairs)} shard pairs, "
        f"total_samples={total_samples}, workers={worker_devices}"
    )

    helpers.clear_task_outputs(args.output_dir, task_prefix="tri2ind", manifest_name="tri2ind.manifest.json")

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

        with tqdm(total=total_samples, desc="tri2ind total", unit="sample", position=0) as total_progress:
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
                    futures.append(executor.submit(_process_one_batch, int(start), tri_batch, meta_batch))
                    batch_index += 1

                with tqdm(
                    total=shard_size,
                    desc=f"tri2ind shard {shard_index}/{len(tri_meta_pairs)}",
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
                            raise RuntimeError("Worker returned inconsistent tri2ind batch result length.")
                        shard_outputs[start_index : start_index + sample_count] = records
                        shard_progress.update(sample_count)
                        total_progress.update(sample_count)

                if any(item is None for item in shard_outputs):
                    raise RuntimeError(f"tri2ind shard has missing outputs: {pair.tri_path}")
                output_samples = [item for item in shard_outputs if item is not None]
                output_path = helpers.build_task_output_path(args.output_dir, "tri2ind", pair.tri_path)
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
                    f"[INFO] Encoded shard {shard_index}/{len(tri_meta_pairs)}: {pair.tri_path.name} "
                    f"-> {output_path.name} ({len(output_samples)} samples)"
                )
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=False)

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
            "gpus": str(args.gpus),
            "worker_devices": worker_devices,
            "project_root": str(project_root),
        },
        shard_records=shard_records,
    )

    print(f"[INFO] Saved tri2ind shards to: {Path(args.output_dir).expanduser().resolve()}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Exported samples: {sum(int(item['sample_count']) for item in shard_records)}")


if __name__ == "__main__":
    main()
