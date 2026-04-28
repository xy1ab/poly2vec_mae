"""Convert RVQ codebook indices into summed quantized embeddings."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from pathlib import Path
import sys
import warnings

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


def _load_codebooks(quantizer_path: str | Path):
    """Load RVQ codebooks from one `quantizer.pth` state dict."""
    import torch

    path = Path(quantizer_path).expanduser().resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Quantizer checkpoint must be a state dict: {path}")

    level_items = []
    for key, value in state.items():
        if not str(key).startswith("quantizers.") or not str(key).endswith(".codebook"):
            continue
        parts = str(key).split(".")
        if len(parts) >= 3 and parts[1].isdigit():
            level_items.append((int(parts[1]), value))

    if level_items:
        level_items.sort(key=lambda item: item[0])
        codebooks = [value.detach().float().cpu().contiguous() for _, value in level_items]
    elif "codebook" in state:
        codebooks = [state["codebook"].detach().float().cpu().contiguous()]
    else:
        raise KeyError(f"Failed to find codebook tensors in quantizer checkpoint: {path}")

    if not codebooks:
        raise ValueError(f"No codebooks found in quantizer checkpoint: {path}")
    first_shape = tuple(codebooks[0].shape)
    if len(first_shape) != 2:
        raise ValueError(f"Codebook tensor must have shape [K,C], got {first_shape}")
    for level, codebook in enumerate(codebooks):
        if codebook.ndim != 2:
            raise ValueError(f"Codebook level {level} must have shape [K,C], got {tuple(codebook.shape)}")
        if int(codebook.shape[1]) != int(codebooks[0].shape[1]):
            raise ValueError("All RVQ codebooks must share the same embedding dimension")
    return codebooks


def _normalize_indices(indices, num_levels: int):
    """Normalize one index grid to `[Q,H,W]` and validate RVQ level count."""
    import torch

    tensor = torch.as_tensor(indices)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"`indices` must have shape [H,W] or [Q,H,W], got {tuple(tensor.shape)}")
    if int(tensor.shape[0]) != int(num_levels):
        raise ValueError(f"Expected {num_levels} RVQ levels, got {int(tensor.shape[0])}")
    return tensor


def indices_to_embedding(indices, codebooks):
    """Map `[Q,H,W]` RVQ indices to summed decoder input embedding `[C,H,W]`."""
    import torch

    indices_tensor = _normalize_indices(indices, len(codebooks))
    levels, height, width = indices_tensor.shape
    code_dim = int(codebooks[0].shape[1])
    embedding = torch.zeros((code_dim, height, width), dtype=torch.float32)
    for level in range(int(levels)):
        codebook = codebooks[level]
        flat_indices = indices_tensor[level].reshape(-1).long()
        if flat_indices.numel() > 0:
            min_value = int(flat_indices.min().item())
            max_value = int(flat_indices.max().item())
            if min_value < 0 or max_value >= int(codebook.shape[0]):
                raise ValueError(
                    f"Index out of range at RVQ level {level}: "
                    f"valid=[0,{int(codebook.shape[0]) - 1}], got min={min_value}, max={max_value}"
                )
        quantized = codebook.index_select(0, flat_indices)
        quantized = quantized.reshape(height, width, code_dim).permute(2, 0, 1).contiguous()
        embedding.add_(quantized)
    return embedding


def _warn_missing_optional_keys(missing_keys: set[str], context: str | None = None) -> None:
    """Warn once when optional provenance fields are absent in an input file."""
    if not missing_keys:
        return
    location = f" in {context}" if context else ""
    keys = ", ".join(sorted(missing_keys))
    warnings.warn(
        f"Optional field(s) missing{location}: {keys}. Output will store None for missing values.",
        RuntimeWarning,
        stacklevel=2,
    )


def _convert_record(record: dict, codebooks, missing_optional_keys: set[str] | None = None) -> dict:
    """Convert one tri2ind record while preserving original fields."""
    if not isinstance(record, dict):
        raise TypeError(f"Each input sample must be a dict, got {type(record).__name__}")
    if "indices" not in record:
        raise KeyError("Input sample is missing required key `indices`")
    output = dict(record)
    for key in ("gid", "meta"):
        if key not in output:
            output[key] = None
            if missing_optional_keys is not None:
                missing_optional_keys.add(key)
    output["embedding"] = indices_to_embedding(record["indices"], codebooks).float().cpu()
    return output


def _convert_chunk(start_index: int, records: list[dict], codebooks) -> tuple[int, list[dict], set[str]]:
    """Worker task for converting one contiguous record chunk."""
    missing_optional_keys: set[str] = set()
    converted = [_convert_record(record, codebooks, missing_optional_keys) for record in records]
    return int(start_index), converted, missing_optional_keys


def load_index_records(ind_path: str | Path) -> list[dict]:
    """Load one torch-saved tri2ind record list."""
    import torch

    path = Path(ind_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    if not isinstance(records, list):
        raise TypeError(f"Input index file must contain a list of records: {path}")
    return records


def convert_records_parallel(
    records: list[dict],
    quantizer_path: str | Path,
    num_workers: int,
    *,
    show_progress: bool = True,
    context: str | None = None,
) -> list[dict]:
    """Convert one loaded record list with optional CPU multiprocessing."""
    if not records:
        return []

    worker_count = max(1, int(num_workers))
    worker_count = min(worker_count, len(records))
    if worker_count == 1:
        codebooks = _load_codebooks(quantizer_path)
        iterator = tqdm(records, desc="ind2emd", unit="sample") if show_progress else records
        missing_optional_keys: set[str] = set()
        converted = [_convert_record(record, codebooks, missing_optional_keys) for record in iterator]
        _warn_missing_optional_keys(missing_optional_keys, context)
        return converted

    codebooks = _load_codebooks(quantizer_path)
    chunk_size = max(1, math.ceil(len(records) / (worker_count * 4)))
    output: list[dict | None] = [None] * len(records)
    missing_optional_keys: set[str] = set()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for start in range(0, len(records), chunk_size):
            end = min(start + chunk_size, len(records))
            futures.append(executor.submit(_convert_chunk, int(start), records[start:end], codebooks))

        futures_iter = as_completed(futures)
        if show_progress:
            with tqdm(total=len(records), desc="ind2emd", unit="sample") as progress:
                for future in futures_iter:
                    start_index, converted, chunk_missing_keys = future.result()
                    missing_optional_keys.update(chunk_missing_keys)
                    output[start_index : start_index + len(converted)] = converted
                    progress.update(len(converted))
        else:
            for future in futures_iter:
                start_index, converted, chunk_missing_keys = future.result()
                missing_optional_keys.update(chunk_missing_keys)
                output[start_index : start_index + len(converted)] = converted

    if any(item is None for item in output):
        raise RuntimeError("Converted output has missing records.")
    _warn_missing_optional_keys(missing_optional_keys, context)
    return [item for item in output if item is not None]


def convert_file(
    ind_path: str | Path,
    quantizer_path: str | Path,
    output_path: str | Path,
    num_workers: int,
    *,
    show_progress: bool = True,
) -> dict:
    """Convert one tri2ind `.pt` file and write one ind2emd `.pt` file."""
    import torch

    in_path = Path(ind_path).expanduser().resolve()
    out_path = Path(output_path).expanduser().resolve()
    records = load_index_records(in_path)
    converted = convert_records_parallel(
        records,
        quantizer_path,
        num_workers,
        show_progress=show_progress,
        context=str(in_path),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, out_path)
    return {
        "path": str(out_path),
        "sample_count": int(len(converted)),
        "size_bytes": int(out_path.stat().st_size),
        "input_ind_path": str(in_path),
    }


def main() -> None:
    ensure_cuda_runtime_libs()

    parser = argparse.ArgumentParser(description="Convert one tri2ind shard to quantized embeddings.")
    parser.add_argument("--ind_path", type=str, required=True, help="Input `.pt` file from `run_tri2ind_batch.py`.")
    parser.add_argument("--quantizer_path", type=str, required=True, help="Path to `quantizer.pth`.")
    parser.add_argument("--output_path", type=str, required=True, help="Output `.pt` path.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 1, help="CPU worker count.")
    args = parser.parse_args()

    if args.num_workers <= 0:
        raise ValueError(f"`num_workers` must be > 0, got {args.num_workers}")
    record = convert_file(
        ind_path=args.ind_path,
        quantizer_path=args.quantizer_path,
        output_path=args.output_path,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Saved embeddings to: {record['path']}")
    print(f"[INFO] Converted samples: {record['sample_count']}")


if __name__ == "__main__":
    main()
