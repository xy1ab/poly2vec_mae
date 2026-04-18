"""Shared helpers for RVQAE batch export scripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UINT16_MAX = 65535
OUTPUT_SERIALIZATION = "torch_save_list_of_dicts"


@dataclass(frozen=True)
class TriMetaShardPair:
    """One paired triangle/meta shard descriptor."""

    tri_path: Path
    meta_path: Path
    sample_count: int | None


def resolve_runtime_device(requested_device: str) -> str:
    """Resolve runtime device with CUDA-unavailable fallback."""
    import torch

    value = str(requested_device).strip() or "cuda"
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; falling back to CPU.")
        return "cpu"
    return value


def parse_gpu_devices(gpus_csv: str, fallback_device: str = "cuda") -> list[str]:
    """Parse GPU CSV into runtime device list with CUDA-unavailable fallback."""
    import torch

    gpus_csv = str(gpus_csv).strip()
    if not gpus_csv:
        return [resolve_runtime_device(fallback_device)]

    if not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; ignoring `--gpus` and using CPU.")
        return ["cpu"]

    tokens = [item.strip() for item in gpus_csv.split(",") if item.strip()]
    if not tokens:
        return [resolve_runtime_device(fallback_device)]

    device_count = int(torch.cuda.device_count())
    resolved: list[str] = []
    for token in tokens:
        gpu_id = int(token)
        if gpu_id < 0 or gpu_id >= device_count:
            raise ValueError(
                f"Invalid GPU id {gpu_id}; visible CUDA device count is {device_count}."
            )
        resolved.append(f"cuda:{gpu_id}")
    return resolved


def resolve_model_paths(model_dir: str | Path) -> tuple[Path, Path]:
    """Resolve `rvqae_best.pth` and one config file from model directory."""
    base = Path(model_dir).expanduser().resolve()
    search_roots = [base] + [path for path in (base / "best", base / "ckpt") if path.is_dir()]

    checkpoint_path: Path | None = None
    config_path: Path | None = None
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


def resolve_decode_paths(model_dir: str | Path) -> tuple[Path, Path, Path]:
    """Resolve decoder/quantizer/config paths from model directory."""
    base = Path(model_dir).expanduser().resolve()
    search_roots = [base] + [path for path in (base / "best", base / "ckpt") if path.is_dir()]

    decoder_path: Path | None = None
    quantizer_path: Path | None = None
    config_path: Path | None = None
    for root in search_roots:
        if decoder_path is None and (root / "decoder.pth").is_file():
            decoder_path = root / "decoder.pth"
        if quantizer_path is None and (root / "quantizer.pth").is_file():
            quantizer_path = root / "quantizer.pth"
        if config_path is None:
            for candidate in (root / "config.yaml", root / "config.yml", root / "poly_rvqae_config.json"):
                if candidate.is_file():
                    config_path = candidate
                    break

    if decoder_path is None or quantizer_path is None or config_path is None:
        raise FileNotFoundError(f"Failed to resolve decoder/quantizer/config under: {base}")
    return decoder_path, quantizer_path, config_path


def _resolve_manifest_entry_path(raw_path: str | Path, manifest_path: Path) -> Path:
    """Resolve one manifest-listed path to an absolute existing path."""
    value = Path(str(raw_path)).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (manifest_path.parent / value).resolve()


def _extract_sample_count(entry: dict[str, Any]) -> int | None:
    """Extract one non-negative integer sample count from manifest entry."""
    if "sample_count" not in entry:
        return None
    try:
        value = int(entry["sample_count"])
    except Exception:
        return None
    if value < 0:
        return None
    return value


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load one JSON manifest file as dictionary."""
    with manifest_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be one JSON object: {manifest_path}")
    return payload


def _expected_meta_path_from_tri(tri_path: Path) -> Path:
    """Infer paired meta shard path from one triangle shard path."""
    stem = tri_path.stem
    if "_tri" in stem:
        meta_stem = stem.replace("_tri", "_meta", 1)
    else:
        meta_stem = f"{stem}_meta"
    return tri_path.with_name(f"{meta_stem}{tri_path.suffix}")


def resolve_tri_meta_pairs(tri_dir: str | Path) -> list[TriMetaShardPair]:
    """Resolve paired triangle/meta shards from input directory."""
    base = Path(tri_dir).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"`tri_dir` is not a directory: {base}")

    manifest_files = sorted(path.resolve() for path in base.glob("*.manifest.json") if path.is_file())
    if len(manifest_files) == 1:
        manifest_path = manifest_files[0]
        payload = _load_manifest(manifest_path)
        tri_entries = payload.get("shards")
        meta_entries = payload.get("meta_shards")
        if isinstance(tri_entries, list) and isinstance(meta_entries, list):
            if len(tri_entries) != len(meta_entries):
                raise ValueError(
                    f"Manifest triangle/meta shard count mismatch: tri={len(tri_entries)}, meta={len(meta_entries)}"
                )
            if not tri_entries:
                raise ValueError(f"Manifest shard lists are empty: {manifest_path}")

            pairs: list[TriMetaShardPair] = []
            for tri_entry, meta_entry in zip(tri_entries, meta_entries):
                if not isinstance(tri_entry, dict) or not isinstance(meta_entry, dict):
                    raise ValueError(f"Manifest shard entries must be objects: {manifest_path}")
                tri_raw = tri_entry.get("path")
                meta_raw = meta_entry.get("path")
                if not tri_raw or not meta_raw:
                    raise ValueError(f"Manifest shard entry missing `path`: {manifest_path}")

                tri_path = _resolve_manifest_entry_path(tri_raw, manifest_path)
                meta_path = _resolve_manifest_entry_path(meta_raw, manifest_path)
                if not tri_path.is_file():
                    raise FileNotFoundError(f"Triangle shard listed in manifest does not exist: {tri_path}")
                if not meta_path.is_file():
                    raise FileNotFoundError(f"Meta shard listed in manifest does not exist: {meta_path}")

                tri_count = _extract_sample_count(tri_entry)
                meta_count = _extract_sample_count(meta_entry)
                if tri_count is not None and meta_count is not None and tri_count != meta_count:
                    raise ValueError(
                        f"Manifest sample_count mismatch for pair tri={tri_path.name}, meta={meta_path.name}: "
                        f"{tri_count} vs {meta_count}"
                    )
                sample_count = tri_count if tri_count is not None else meta_count
                pairs.append(TriMetaShardPair(tri_path=tri_path, meta_path=meta_path, sample_count=sample_count))

            return pairs

    tri_files = sorted(path.resolve() for path in base.glob("*.pt") if path.is_file() and "_meta" not in path.stem)
    meta_files = sorted(path.resolve() for path in base.glob("*_meta*.pt") if path.is_file())
    if not tri_files:
        raise FileNotFoundError(f"No triangle shard `.pt` files found under: {base}")
    if not meta_files:
        raise FileNotFoundError(f"No meta shard `.pt` files found under: {base}")

    matched_meta: set[Path] = set()
    pairs = []
    for tri_path in tri_files:
        meta_path = _expected_meta_path_from_tri(tri_path).resolve()
        if not meta_path.is_file():
            raise FileNotFoundError(f"Failed to locate paired meta shard for {tri_path.name}: expected {meta_path.name}")
        matched_meta.add(meta_path)
        pairs.append(TriMetaShardPair(tri_path=tri_path, meta_path=meta_path, sample_count=None))

    extra_meta = [path for path in meta_files if path not in matched_meta]
    if extra_meta:
        raise ValueError(
            f"Found meta shard(s) without paired triangle shards: {[path.name for path in extra_meta[:5]]}"
        )

    return pairs


def preflight_validate_tri_meta_pairs(pairs: list[TriMetaShardPair]) -> int:
    """Validate triangle/meta sample-count consistency before inference starts."""
    total_samples = 0
    for pair in pairs:
        if pair.sample_count is not None:
            total_samples += int(pair.sample_count)
            continue

        tri_samples = load_torch_list(pair.tri_path)
        meta_samples = load_torch_list(pair.meta_path)
        if len(tri_samples) != len(meta_samples):
            raise ValueError(
                f"Triangle/meta sample count mismatch for {pair.tri_path.name} and {pair.meta_path.name}: "
                f"{len(tri_samples)} vs {len(meta_samples)}"
            )
        total_samples += len(tri_samples)
    return total_samples


def resolve_ind_shards(ind_dir: str | Path) -> list[Path]:
    """Resolve index shard files from one index directory."""
    base = Path(ind_dir).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"`ind_dir` is not a directory: {base}")

    preferred_manifest = base / "tri2ind.manifest.json"
    manifest_candidates = []
    if preferred_manifest.is_file():
        manifest_candidates = [preferred_manifest]
    else:
        manifest_candidates = sorted(path.resolve() for path in base.glob("*.manifest.json") if path.is_file())

    if len(manifest_candidates) == 1:
        payload = _load_manifest(manifest_candidates[0])
        entries = payload.get("shards")
        if isinstance(entries, list) and entries:
            shard_paths = []
            for entry in entries:
                if not isinstance(entry, dict) or "path" not in entry:
                    raise ValueError(f"Invalid shard entry in manifest: {manifest_candidates[0]}")
                shard_path = _resolve_manifest_entry_path(entry["path"], manifest_candidates[0])
                if not shard_path.is_file():
                    raise FileNotFoundError(f"Index shard listed in manifest does not exist: {shard_path}")
                shard_paths.append(shard_path)
            return shard_paths

    shard_paths = sorted(path.resolve() for path in base.glob("tri2ind_part_*.pt") if path.is_file())
    if not shard_paths:
        shard_paths = sorted(path.resolve() for path in base.glob("*.pt") if path.is_file())
    if not shard_paths:
        raise FileNotFoundError(f"No index shard `.pt` files found under: {base}")
    return shard_paths


def load_torch_list(path: str | Path) -> list[Any]:
    """Load one torch-saved Python list."""
    import torch

    shard_path = Path(path).expanduser().resolve()
    payload = torch.load(shard_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, list):
        raise TypeError(f"Shard payload must be a list: {shard_path}")
    return payload


def build_task_output_path(
    output_dir: str | Path,
    task_prefix: str,
    input_shard_path: str | Path,
) -> Path:
    """Build one output shard path with task prefix and input-stem suffix."""
    output_root = Path(output_dir).expanduser().resolve()
    in_path = Path(input_shard_path).expanduser().resolve()
    return output_root / f"{task_prefix}_{in_path.stem}.pt"


def clear_task_outputs(output_dir: str | Path, task_prefix: str, manifest_name: str) -> None:
    """Remove old outputs for one task prefix and manifest file."""
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(task_prefix)}_.+\.pt$")
    for path in output_root.glob("*.pt"):
        if path.is_file() and pattern.match(path.name):
            path.unlink()
    manifest_path = output_root / manifest_name
    if manifest_path.exists():
        manifest_path.unlink()


def write_task_manifest(
    output_dir: str | Path,
    manifest_name: str,
    metadata: dict[str, Any],
    shard_records: list[dict[str, Any]],
) -> Path:
    """Write one task manifest with shard-level metadata."""
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / manifest_name
    total_samples = int(sum(int(record.get("sample_count", 0)) for record in shard_records))
    manifest = {
        "serialization": OUTPUT_SERIALIZATION,
        "num_shards": int(len(shard_records)),
        "total_samples": total_samples,
        "shards": shard_records,
        "metadata": dict(metadata),
    }
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    return manifest_path


def normalize_indices_grid(indices: Any):
    """Normalize one indices grid to long tensor shape `[Q,H,W]`."""
    import torch

    tensor = torch.as_tensor(indices, dtype=torch.long)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Indices grid must have shape [Q,H,W] or [H,W], got {tuple(tensor.shape)}")
    return tensor


def to_uint16_indices(indices: Any, context: str = ""):
    """Convert indices to `uint16` with range validation."""
    import torch

    tensor = torch.as_tensor(indices, dtype=torch.long)
    if tensor.numel() > 0:
        min_value = int(tensor.min().item())
        max_value = int(tensor.max().item())
        if min_value < 0 or max_value > UINT16_MAX:
            prefix = f"{context}: " if context else ""
            raise ValueError(
                f"{prefix}indices out of uint16 range [0, {UINT16_MAX}], got min={min_value}, max={max_value}"
            )
    return tensor.to(dtype=torch.uint16, device="cpu")


class SampleShardWriter:
    """Buffered shard writer for list-of-dict output payloads."""

    def __init__(
        self,
        output_dir: str | Path,
        shard_prefix: str,
        manifest_name: str,
        shard_size: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.shard_prefix = str(shard_prefix)
        self.manifest_path = self.output_dir / str(manifest_name)
        self.shard_size = int(shard_size)
        if self.shard_size <= 0:
            raise ValueError(f"`shard_size` must be > 0, got {self.shard_size}")
        self.metadata = dict(metadata or {})

        self._buffer: list[dict[str, Any]] = []
        self.shard_paths: list[Path] = []
        self.shard_counts: list[int] = []
        self.total_samples = 0

        self._prepare_output_dir()

    def _prepare_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for path in self.output_dir.glob(f"{self.shard_prefix}_part_*.pt"):
            path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    def _flush(self) -> None:
        if not self._buffer:
            return
        import torch

        part_index = len(self.shard_paths) + 1
        shard_path = self.output_dir / f"{self.shard_prefix}_part_{part_index:04d}.pt"
        torch.save(list(self._buffer), shard_path)

        self.shard_paths.append(shard_path)
        self.shard_counts.append(len(self._buffer))
        self.total_samples += len(self._buffer)
        self._buffer = []

    def add(self, sample: dict[str, Any]) -> None:
        """Append one sample and flush when current shard is full."""
        self._buffer.append(sample)
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def finalize(self) -> Path:
        """Flush remaining buffered samples and write one JSON manifest."""
        self._flush()
        manifest = {
            "serialization": OUTPUT_SERIALIZATION,
            "shard_size": self.shard_size,
            "num_shards": len(self.shard_paths),
            "total_samples": self.total_samples,
            "shards": [
                {
                    "path": str(path),
                    "sample_count": int(count),
                    "size_bytes": int(path.stat().st_size) if path.exists() else -1,
                }
                for path, count in zip(self.shard_paths, self.shard_counts)
            ],
            "metadata": self.metadata,
        }
        with self.manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(manifest, fp, ensure_ascii=False, indent=2)
        return self.manifest_path
