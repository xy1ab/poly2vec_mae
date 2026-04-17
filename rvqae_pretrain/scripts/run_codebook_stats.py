"""Collect basic codebook-usage statistics from a RVQAE model over triangle shards."""

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


def _resolve_model_paths(model_dir: str) -> tuple[Path, Path]:
    base = Path(model_dir).expanduser().resolve()
    search_roots = [base] + [p for p in (base / "best", base / "ckpt") if p.is_dir()]
    checkpoint_path = None
    config_path = None
    for root in search_roots:
        if checkpoint_path is None and (root / "rvqae_best.pth").is_file():
            checkpoint_path = root / "rvqae_best.pth"
        if config_path is None:
            for candidate in (root / "config.yaml", root / "config.yml", root / "poly_rvqae_config.json"):
                if candidate.is_file():
                    config_path = candidate
                    break
    if checkpoint_path is None or config_path is None:
        raise FileNotFoundError(f"Failed to resolve `rvqae_best.pth` + config under: {base}")
    return checkpoint_path, config_path


def main() -> None:
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    if __package__ in {None, ""}:
        import importlib

        shard_io = importlib.import_module("rvqae_pretrain.src.datasets.shard_io")
        resolve_triangle_shard_paths = shard_io.resolve_triangle_shard_paths
        load_triangle_shard = shard_io.load_triangle_shard
        PolyRvqAePipeline = importlib.import_module("rvqae_pretrain.src.engine.pipeline").PolyRvqAePipeline
    else:
        from ..src.datasets.shard_io import load_triangle_shard, resolve_triangle_shard_paths
        from ..src.engine.pipeline import PolyRvqAePipeline

    parser = argparse.ArgumentParser(description="Collect RVQ codebook usage statistics")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing RVQAE checkpoint and config.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing triangle shard `.pt` files.")
    parser.add_argument("--output_path", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path, config_path = _resolve_model_paths(args.model_dir)
    pipeline = PolyRvqAePipeline(
        weight_path=str(checkpoint_path),
        config_path=str(config_path),
        device=args.device,
        precision=args.precision,
    )

    num_quantizers = int(getattr(pipeline.model.quantizer, "num_quantizers", 1))
    code_usage = torch.zeros(num_quantizers, pipeline.model.quantizer.num_embeddings, dtype=torch.long)
    total_perplexity = torch.zeros(num_quantizers, dtype=torch.float32)
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
                indices = outputs.indices.detach().cpu()
                if indices.ndim == 3:
                    indices = indices.unsqueeze(1)
                for level in range(num_quantizers):
                    code_usage[level] += torch.bincount(
                        indices[:, level].reshape(-1),
                        minlength=pipeline.model.quantizer.num_embeddings,
                    )
                total_perplexity += outputs.perplexity.detach().float().cpu()
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
        indices = outputs.indices.detach().cpu()
        if indices.ndim == 3:
            indices = indices.unsqueeze(1)
        for level in range(num_quantizers):
            code_usage[level] += torch.bincount(
                indices[:, level].reshape(-1),
                minlength=pipeline.model.quantizer.num_embeddings,
            )
        total_perplexity += outputs.perplexity.detach().float().cpu()
        batch_count += 1
        sample_count += len(current_batch)

    active_codes = (code_usage > 0).sum(dim=1)
    total_codes = int(code_usage.shape[1])
    usage_probs = code_usage.float() / code_usage.sum(dim=1, keepdim=True).clamp_min(1)
    global_perplexity = torch.exp(-(usage_probs * torch.log(usage_probs.clamp_min(1e-10))).sum(dim=1))
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "input_dir": str(Path(args.input_dir).expanduser().resolve()),
        "total_samples": int(sample_count),
        "num_quantizers": num_quantizers,
        "total_codes_per_quantizer": total_codes,
        "active_codes": [int(v) for v in active_codes.tolist()],
        "code_usage_ratio": [float(v / max(1, total_codes)) for v in active_codes.tolist()],
        "avg_batch_perplexity": [float(v) for v in (total_perplexity / max(1, batch_count)).tolist()],
        "global_perplexity": [float(v) for v in global_perplexity.tolist()],
        "top10_code_usage": [
            [int(v) for v in torch.topk(level_usage, k=min(10, total_codes)).values.tolist()]
            for level_usage in code_usage
        ],
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
