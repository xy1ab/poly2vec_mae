"""Decode VQ index batches back into reconstructed frequency maps."""

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


def _resolve_decode_paths(model_dir: str) -> tuple[Path, Path, Path]:
    base = Path(model_dir).expanduser().resolve()
    search_roots = [base] + [p for p in (base / "best", base / "ckpt") if p.is_dir()]
    decoder_path = quantizer_path = config_path = None
    for root in search_roots:
        if decoder_path is None and (root / "decoder.pth").is_file():
            decoder_path = root / "decoder.pth"
        if quantizer_path is None and (root / "quantizer.pth").is_file():
            quantizer_path = root / "quantizer.pth"
        if config_path is None:
            for candidate in (root / "config.yaml", root / "config.yml", root / "poly_vqae_config.json"):
                if candidate.is_file():
                    config_path = candidate
                    break
    if decoder_path is None or quantizer_path is None or config_path is None:
        raise FileNotFoundError(f"Failed to resolve decoder/quantizer/config under: {base}")
    return decoder_path, quantizer_path, config_path


def main() -> None:
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    if __package__ in {None, ""}:
        import importlib

        PolyVqDecodePipeline = importlib.import_module("vqae_pretrain.src.engine.pipeline").PolyVqDecodePipeline
    else:
        from ..src.engine.pipeline import PolyVqDecodePipeline

    parser = argparse.ArgumentParser(description="Decode quantized VQ indices into frequency maps")
    parser.add_argument("--indices_path", type=str, required=True, help="Path produced by `run_quantize_batch.py`.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing decoder/quantizer/config.")
    parser.add_argument("--output_path", type=str, default="", help="Output `.pt` path.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=str, default="bf16")
    args = parser.parse_args()

    decoder_path, quantizer_path, config_path = _resolve_decode_paths(args.model_dir)
    pipeline = PolyVqDecodePipeline(
        decoder_path=str(decoder_path),
        quantizer_path=str(quantizer_path),
        config_path=str(config_path),
        device=args.device,
        precision=args.precision,
    )

    payload = torch.load(Path(args.indices_path).expanduser().resolve(), map_location="cpu", weights_only=False)
    samples = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    output_samples: list[dict[str, torch.Tensor]] = []

    for start in tqdm(range(0, len(samples), args.batch_size), desc="Decoding VQ indices"):
        batch_samples = samples[start : start + args.batch_size]
        batch_indices = torch.stack([sample["indices"].long() for sample in batch_samples], dim=0)
        real_part, imag_part = pipeline.decode_indices(batch_indices)
        for idx, sample in enumerate(batch_samples):
            output_samples.append(
                {
                    "triangles": sample["triangles"].float().cpu(),
                    "indices": sample["indices"].long().cpu(),
                    "freq_real": real_part[idx].float().cpu(),
                    "freq_imag": imag_part[idx].float().cpu(),
                }
            )

    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else project_root / "outputs" / "decoded" / f"decoded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "indices_path": str(Path(args.indices_path).expanduser().resolve()),
                "model_dir": str(Path(args.model_dir).expanduser().resolve()),
                "decoder_path": str(decoder_path),
                "quantizer_path": str(quantizer_path),
                "config_path": str(config_path),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            "samples": output_samples,
        },
        output_path,
    )
    print(f"[INFO] Saved decoded batch to: {output_path}")


if __name__ == "__main__":
    main()
