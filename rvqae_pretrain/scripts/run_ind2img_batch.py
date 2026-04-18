"""Batch decode RVQ index shards to real/imag frequency maps (and optional ICFT)."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import multiprocessing as mp
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


_WORKER_PIPELINE = None
_WORKER_HELPERS = None


def _inject_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def _preflight_validate_index_shards(ind_shards: list[Path], helpers_module) -> int:
    """Validate index shard payload schema before launching worker pool."""
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


def _init_worker(
    repo_root: str,
    decoder_path: str,
    quantizer_path: str,
    config_path: str,
    device: str,
) -> None:
    """Initialize one per-process decode pipeline bound to one device."""
    global _WORKER_PIPELINE, _WORKER_HELPERS

    repo_root = str(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import importlib

    pipeline_module = importlib.import_module("rvqae_pretrain.src.engine.pipeline")
    _WORKER_HELPERS = importlib.import_module("rvqae_pretrain.scripts.batch_infer_common")
    _WORKER_PIPELINE = pipeline_module.PolyRvqDecodePipeline(
        decoder_path=str(decoder_path),
        quantizer_path=str(quantizer_path),
        config_path=str(config_path),
        device=str(device),
        precision="fp32",
    )


def _process_one_batch(
    start_index: int,
    batch_samples,
    nicft: int,
):
    """Process one index batch and return output records with global start index."""
    global _WORKER_PIPELINE, _WORKER_HELPERS
    if _WORKER_PIPELINE is None or _WORKER_HELPERS is None:
        raise RuntimeError("Worker pipeline is not initialized.")

    import torch

    batch_indices = []
    batch_meta = []
    for local_index, sample in enumerate(batch_samples):
        if not isinstance(sample, dict):
            raise TypeError(f"Index batch sample must be dict: local#{local_index}")
        if "indices" not in sample or "meta" not in sample:
            raise KeyError(f"Index batch sample missing `indices`/`meta`: local#{local_index}")
        batch_indices.append(_WORKER_HELPERS.normalize_indices_grid(sample["indices"]))
        batch_meta.append(sample["meta"])

    indices_batch = torch.stack(batch_indices, dim=0)
    indices_batch_u16 = _WORKER_HELPERS.to_uint16_indices(indices_batch, context="ind2img")
    real_batch, imag_batch = _WORKER_PIPELINE.decode_indices(indices_batch)

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
    for sample_offset in range(len(batch_samples)):
        record = {
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

    if __package__ in {None, ""}:
        import importlib

        helpers = importlib.import_module("rvqae_pretrain.scripts.batch_infer_common")
    else:
        from . import batch_infer_common as helpers

    parser = argparse.ArgumentParser(description="Decode index shards to real/imag maps.")
    parser.add_argument("--ind_dir", type=str, required=True, help="Directory containing tri2ind shard files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Training `best` directory or its parent.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for ind2img shards.")
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
    decoder_path, quantizer_path, config_path = helpers.resolve_decode_paths(args.model_dir)
    ind_shards = helpers.resolve_ind_shards(args.ind_dir)
    total_samples = _preflight_validate_index_shards(ind_shards, helpers)
    print(
        f"[INFO] Index preflight passed: {len(ind_shards)} shards, "
        f"total_samples={total_samples}, workers={worker_devices}"
    )

    helpers.clear_task_outputs(args.output_dir, task_prefix="ind2img", manifest_name="ind2img.manifest.json")

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
                initargs=(
                    str(project_root.parent),
                    str(decoder_path),
                    str(quantizer_path),
                    str(config_path),
                    str(device),
                ),
            )
            executors.append(executor)

        with tqdm(total=total_samples, desc="ind2img total", unit="sample", position=0) as total_progress:
            for shard_index, ind_path in enumerate(ind_shards, start=1):
                samples = helpers.load_torch_list(ind_path)
                shard_size = len(samples)
                shard_outputs: list[dict | None] = [None] * shard_size

                futures = []
                batch_index = 0
                for start in range(0, shard_size, args.batch_size):
                    end = min(start + args.batch_size, shard_size)
                    batch_samples = samples[start:end]
                    executor = executors[batch_index % len(executors)]
                    futures.append(executor.submit(_process_one_batch, int(start), batch_samples, int(args.nicft)))
                    batch_index += 1

                with tqdm(
                    total=shard_size,
                    desc=f"ind2img shard {shard_index}/{len(ind_shards)}",
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
                            raise RuntimeError("Worker returned inconsistent ind2img batch result length.")
                        shard_outputs[start_index : start_index + sample_count] = records
                        shard_progress.update(sample_count)
                        total_progress.update(sample_count)

                if any(item is None for item in shard_outputs):
                    raise RuntimeError(f"ind2img shard has missing outputs: {ind_path}")
                output_samples = [item for item in shard_outputs if item is not None]
                output_path = helpers.build_task_output_path(args.output_dir, "ind2img", ind_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(output_samples, output_path)
                shard_records.append(
                    {
                        "path": str(output_path.resolve()),
                        "sample_count": int(len(output_samples)),
                        "size_bytes": int(output_path.stat().st_size),
                        "input_ind_path": str(Path(ind_path).expanduser().resolve()),
                    }
                )
                print(
                    f"[INFO] Decoded shard {shard_index}/{len(ind_shards)}: {Path(ind_path).name} "
                    f"-> {output_path.name} ({len(output_samples)} samples)"
                )
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=False)

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
            "worker_devices": worker_devices,
            "nicft": int(args.nicft),
            "project_root": str(project_root),
        },
        shard_records=shard_records,
    )

    print(f"[INFO] Saved ind2img shards to: {Path(args.output_dir).expanduser().resolve()}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Exported samples: {sum(int(item['sample_count']) for item in shard_records)}")


if __name__ == "__main__":
    main()
