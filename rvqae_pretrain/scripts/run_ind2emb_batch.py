"""Batch convert tri2ind shards into summed quantized embeddings."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
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
    ind2emd = importlib.import_module("rvqae_pretrain.scripts.run_ind2emd")
else:
    from .runtime_bootstrap import ensure_cuda_runtime_libs
    from . import run_ind2emb as ind2emd


def _resolve_manifest_entry_path(raw_path: str | Path, manifest_path: Path) -> Path:
    value = Path(str(raw_path)).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (manifest_path.parent / value).resolve()


def _resolve_ind_paths(ind_dir: str | Path) -> list[Path]:
    """Resolve tri2ind input shards, preferring `tri2ind.manifest.json`."""
    base = Path(ind_dir).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"`ind_dir` is not a directory: {base}")

    manifest_path = base / "tri2ind.manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        entries = payload.get("shards")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"`tri2ind.manifest.json` has no non-empty `shards` list: {manifest_path}")
        paths = []
        for entry in entries:
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError(f"Invalid shard entry in manifest: {manifest_path}")
            path = _resolve_manifest_entry_path(entry["path"], manifest_path)
            if not path.is_file():
                raise FileNotFoundError(f"Manifest-listed tri2ind shard does not exist: {path}")
            paths.append(path)
        return paths

    paths = sorted(path.resolve() for path in base.glob("tri2ind*.pt") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No `tri2ind*.pt` files found under: {base}")
    return paths


def _build_output_path(output_dir: str | Path, ind_path: str | Path) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    stem = Path(ind_path).stem
    suffix = stem[len("tri2ind_") :] if stem.startswith("tri2ind_") else stem
    return output_root / f"ind2emd_{suffix}.pt"


def _convert_one_file_task(task: dict) -> dict:
    return ind2emd.convert_file(
        ind_path=task["ind_path"],
        quantizer_path=task["quantizer_path"],
        output_path=task["output_path"],
        num_workers=1,
        show_progress=False,
    )


def _write_manifest(
    *,
    output_dir: str | Path,
    ind_dir: str | Path,
    quantizer_path: str | Path,
    num_workers: int,
    shard_records: list[dict],
) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "ind2emd.manifest.json"
    manifest = {
        "serialization": "torch_save_list_of_dicts",
        "num_shards": int(len(shard_records)),
        "total_samples": int(sum(int(record["sample_count"]) for record in shard_records)),
        "shards": shard_records,
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "ind_dir": str(Path(ind_dir).expanduser().resolve()),
            "quantizer_path": str(Path(quantizer_path).expanduser().resolve()),
            "num_workers": int(num_workers),
            "embedding_key": "embedding",
            "embedding_dtype": "float32",
            "embedding_layout": "[C,H,W]",
            "indices_dtype": "preserved",
            "missing_optional_fields_policy": "store_none_for_gid_and_meta",
        },
    }
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    ensure_cuda_runtime_libs()

    parser = argparse.ArgumentParser(description="Convert all tri2ind shards in a directory to embeddings.")
    parser.add_argument("--ind_dir", type=str, required=True, help="Directory containing `tri2ind*.pt` shards.")
    parser.add_argument("--quantizer_path", type=str, required=True, help="Path to `quantizer.pth`.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for `ind2emd*.pt` shards.")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 1, help="CPU worker count.")
    args = parser.parse_args()

    if args.num_workers <= 0:
        raise ValueError(f"`num_workers` must be > 0, got {args.num_workers}")

    ind_paths = _resolve_ind_paths(args.ind_dir)
    worker_count = min(max(1, int(args.num_workers)), len(ind_paths))
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "ind_path": str(path),
            "quantizer_path": str(Path(args.quantizer_path).expanduser().resolve()),
            "output_path": str(_build_output_path(output_root, path)),
        }
        for path in ind_paths
    ]

    shard_records = []
    if worker_count == 1:
        iterator = tqdm(tasks, desc="ind2emd files", unit="file")
        for task in iterator:
            shard_records.append(_convert_one_file_task(task))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_convert_one_file_task, task) for task in tasks]
            with tqdm(total=len(futures), desc="ind2emd files", unit="file") as progress:
                for future in as_completed(futures):
                    shard_records.append(future.result())
                    progress.update(1)

    shard_records.sort(key=lambda record: record["input_ind_path"])
    manifest_path = _write_manifest(
        output_dir=output_root,
        ind_dir=args.ind_dir,
        quantizer_path=args.quantizer_path,
        num_workers=worker_count,
        shard_records=shard_records,
    )
    print(f"[INFO] Saved ind2emd shards to: {output_root}")
    print(f"[INFO] Manifest: {manifest_path}")
    print(f"[INFO] Exported samples: {sum(int(item['sample_count']) for item in shard_records)}")


if __name__ == "__main__":
    main()
