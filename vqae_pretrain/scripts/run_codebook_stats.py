"""Collect basic codebook-usage statistics from a VQAE model over triangle shards."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
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

    parser = argparse.ArgumentParser(description="Collect VQ codebook usage statistics")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing VQAE checkpoint and config.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing triangle shard `.pt` files.")
    parser.add_argument("--output_path", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path, config_path = _resolve_model_paths(args.model_dir)
    pipeline = PolyVqAePipeline(
        weight_path=str(checkpoint_path),
        config_path=str(config_path),
        device=args.device,
        precision=args.precision,
    )

    code_usage = torch.zeros(pipeline.model.quantizer.num_embeddings, dtype=torch.long)
    total_perplexity = 0.0
    batch_count = 0
    sample_count = 0
    current_batch: list = []

    shard_paths = resolve_triangle_shard_paths(Path(args.input_dir).expanduser().resolve())
    for shard_path in shard_paths:
        shard_samples = load_triangle_shard(shard_path)
        for sample in tqdm(shard_samples, desc=f"Stats {shard_path.name}", leave=False):
            current_batch.append(sample)
            if args.max_samples > 0 and (sample_count + len(current_batch)) > args.max_samples:
                current_batch = current_batch[: args.max_samples - sample_count]
            if len(current_batch) >= args.batch_size or (args.max_samples > 0 and sample_count + len(current_batch) >= args.max_samples):
                imgs = pipeline.triangles_to_images(current_batch)
                with torch.no_grad():
                    with torch.autocast(device_type=pipeline.device.type, enabled=False):
                        outputs = pipeline.model(imgs, use_vq=True)
                code_usage += torch.bincount(
                    outputs.indices.reshape(-1).cpu(),
                    minlength=pipeline.model.quantizer.num_embeddings,
                )
                total_perplexity += float(outputs.perplexity.item())
                batch_count += 1
                sample_count += len(current_batch)
                current_batch = []
                if args.max_samples > 0 and sample_count >= args.max_samples:
                    break
        if args.max_samples > 0 and sample_count >= args.max_samples:
            break

    if current_batch:
        imgs = pipeline.triangles_to_images(current_batch)
        with torch.no_grad():
            with torch.autocast(device_type=pipeline.device.type, enabled=False):
                outputs = pipeline.model(imgs, use_vq=True)
        code_usage += torch.bincount(
            outputs.indices.reshape(-1).cpu(),
            minlength=pipeline.model.quantizer.num_embeddings,
        )
        total_perplexity += float(outputs.perplexity.item())
        batch_count += 1
        sample_count += len(current_batch)

    active_codes = int((code_usage > 0).sum().item())
    total_codes = int(code_usage.numel())
    usage_probs = code_usage.float() / code_usage.sum().clamp_min(1)
    global_perplexity = float(torch.exp(-(usage_probs * torch.log(usage_probs.clamp_min(1e-10))).sum()).item())
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "input_dir": str(Path(args.input_dir).expanduser().resolve()),
        "total_samples": int(sample_count),
        "total_codes": total_codes,
        "active_codes": active_codes,
        "code_usage_ratio": float(active_codes / max(1, total_codes)),
        "avg_batch_perplexity": float(total_perplexity / max(1, batch_count)),
        "global_perplexity": global_perplexity,
        "top10_code_usage": [int(v) for v in torch.topk(code_usage, k=min(10, total_codes)).values.tolist()],
    }

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
    else:
        output_path = project_root / "outputs" / "codebook_stats" / f"codebook_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[INFO] Saved codebook stats to: {output_path}")


if __name__ == "__main__":
    main()
