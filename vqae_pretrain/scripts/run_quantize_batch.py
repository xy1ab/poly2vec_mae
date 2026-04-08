"""Batch-quantize triangulated polygon samples into VQ indices."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CURRENT_DIR.parent
    _REPO_ROOT = _PROJECT_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib

    ensure_cuda_runtime_libs = importlib.import_module(
        "vqae_pretrain.scripts.runtime_bootstrap"
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


def _resolve_model_paths(model_dir: str) -> tuple[Path, Path]:
    base = Path(model_dir).expanduser().resolve()
    search_roots = [base] + [p for p in (base / "best", base / "ckpt") if p.is_dir()]
    checkpoint_path = None
    config_path = None
    for root in search_roots:
        if checkpoint_path is None and (root / "vqae_best.pth").is_file():
            checkpoint_path = root / "vqae_best.pth"
        if config_path is None:
            for candidate in (root / "config.yaml", root / "config.yml", root / "poly_vqae_config.json"):
                if candidate.is_file():
                    config_path = candidate
                    break
    if checkpoint_path is None or config_path is None:
        raise FileNotFoundError(f"Failed to resolve `vqae_best.pth` + config under: {base}")
    return checkpoint_path, config_path


def _default_output_path(project_root: Path, data_dir: str) -> Path:
    data_name = Path(data_dir).expanduser().resolve().name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "outputs" / "quantized" / f"{data_name}_codes_{timestamp}.pt"


def main() -> None:
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    if __package__ in {None, ""}:
        import importlib

        shard_io = importlib.import_module("vqae_pretrain.src.datasets.shard_io")
        resolve_triangle_shard_paths = shard_io.resolve_triangle_shard_paths
        load_triangle_shard = shard_io.load_triangle_shard
        PolyVqAePipeline = importlib.import_module("vqae_pretrain.src.engine.pipeline").PolyVqAePipeline
    else:
        from ..src.datasets.shard_io import load_triangle_shard, resolve_triangle_shard_paths
        from ..src.engine.pipeline import PolyVqAePipeline

    parser = argparse.ArgumentParser(description="Batch-quantize triangle shard samples into VQ indices")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing VQAE checkpoint and config.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing triangle shard `.pt` files.")
    parser.add_argument("--output_path", type=str, default="", help="Output `.pt` file path.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples.")
    args = parser.parse_args()

    checkpoint_path, config_path = _resolve_model_paths(args.model_dir)
    pipeline = PolyVqAePipeline(
        weight_path=str(checkpoint_path),
        config_path=str(config_path),
        device=args.device,
        precision=args.precision,
    )

    shard_paths = resolve_triangle_shard_paths(Path(args.input_dir).expanduser().resolve())
    all_samples: list[dict[str, torch.Tensor]] = []
    current_batch: list = []
    sample_count = 0

    for shard_path in shard_paths:
        shard_samples = load_triangle_shard(shard_path)
        for sample in tqdm(shard_samples, desc=f"Quantizing {shard_path.name}", leave=False):
            current_batch.append(sample)
            if args.max_samples > 0 and (sample_count + len(current_batch)) > args.max_samples:
                current_batch = current_batch[: args.max_samples - sample_count]
            if len(current_batch) >= args.batch_size or (args.max_samples > 0 and sample_count + len(current_batch) >= args.max_samples):
                indices_batch = pipeline.quantize_triangles(current_batch)
                for triangles, indices in zip(current_batch, indices_batch):
                    all_samples.append(
                        {
                            "triangles": torch.tensor(triangles, dtype=torch.float32),
                            "indices": indices.to(dtype=torch.long).cpu(),
                        }
                    )
                sample_count += len(current_batch)
                current_batch = []
                if args.max_samples > 0 and sample_count >= args.max_samples:
                    break
        if args.max_samples > 0 and sample_count >= args.max_samples:
            break

    if current_batch:
        indices_batch = pipeline.quantize_triangles(current_batch)
        for triangles, indices in zip(current_batch, indices_batch):
            all_samples.append(
                {
                    "triangles": torch.tensor(triangles, dtype=torch.float32),
                    "indices": indices.to(dtype=torch.long).cpu(),
                }
            )

    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else _default_output_path(project_root, args.input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "model_dir": str(Path(args.model_dir).expanduser().resolve()),
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
                "input_dir": str(Path(args.input_dir).expanduser().resolve()),
                "batch_size": int(args.batch_size),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            "samples": all_samples,
        },
        output_path,
    )
    print(f"[INFO] Saved quantized batch to: {output_path}")
    print(f"[INFO] Exported samples: {len(all_samples)}")


if __name__ == "__main__":
    main()
