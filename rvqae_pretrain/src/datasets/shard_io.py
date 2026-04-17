"""Unified torch-only shard IO helpers for triangle datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence


TORCH_SHARD_SERIALIZATION = "torch_save_numpy_list"


def _emit_warning(warn_fn: Callable[[str], None] | None, message: str) -> None:
    """Emit one warning message when a callback is provided."""
    if warn_fn is not None:
        warn_fn(f"[WARN] {message}")


def save_triangle_shard(path: str | Path, samples: Sequence[Any]) -> Path:
    """Serialize one triangle shard with torch format only.

    Args:
        path: Target shard file path.
        samples: Sequence of per-sample payloads.

    Returns:
        Resolved output path.
    """
    import torch

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(samples), output_path)
    return output_path


def load_triangle_shard(path: str | Path) -> list[Any]:
    """Load one triangle shard saved by the torch-only pipeline.

    Args:
        path: Shard file path.

    Returns:
        Python list stored in the shard.

    Raises:
        RuntimeError: When the shard is not readable by `torch.load`.
        TypeError: When the shard payload is not a Python list.
    """
    import torch

    shard_path = Path(path).expanduser().resolve()
    if not shard_path.is_file():
        raise FileNotFoundError(f"Triangle shard file does not exist: {shard_path}")

    try:
        shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load triangle shard with torch.load: "
            f"{shard_path}. Legacy pickle shards are no longer supported. "
            "Please rebuild the dataset with the current triangulation pipeline."
        ) from exc

    if not isinstance(shard_data, list):
        raise TypeError(f"Triangle shard must store a Python list: {shard_path}")
    return shard_data


def _resolve_manifest_shard_paths(
    manifest_path: Path,
    data_dir: Path,
    warn_fn: Callable[[str], None] | None,
) -> list[Path] | None:
    """Resolve shard paths from one manifest JSON file."""
    try:
        with manifest_path.open("r", encoding="utf-8") as fp:
            manifest_payload = json.load(fp)
    except Exception as exc:
        _emit_warning(
            warn_fn,
            f"Ignoring manifest {manifest_path.name}: failed to parse JSON ({type(exc).__name__}: {exc}).",
        )
        return None

    if not isinstance(manifest_payload, dict):
        _emit_warning(warn_fn, f"Ignoring manifest {manifest_path.name}: top-level JSON must be an object.")
        return None

    serialization = manifest_payload.get("serialization")
    if serialization != TORCH_SHARD_SERIALIZATION:
        _emit_warning(
            warn_fn,
            f"Ignoring manifest {manifest_path.name}: serialization must be "
            f"{TORCH_SHARD_SERIALIZATION!r}, got {serialization!r}.",
        )
        return None

    shard_entries = manifest_payload.get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        _emit_warning(warn_fn, f"Ignoring manifest {manifest_path.name}: `shards` must be a non-empty list.")
        return None

    resolved_paths: list[Path] = []
    for entry in shard_entries:
        if not isinstance(entry, dict):
            _emit_warning(
                warn_fn,
                f"Ignoring manifest {manifest_path.name}: every shard entry must be an object.",
            )
            return None
        raw_path = entry.get("path")
        if not raw_path:
            _emit_warning(
                warn_fn,
                f"Ignoring manifest {manifest_path.name}: shard entry is missing `path`.",
            )
            return None

        shard_path = Path(str(raw_path)).expanduser()
        if not shard_path.is_absolute():
            manifest_relative_path = (manifest_path.parent / shard_path).resolve()
            if manifest_relative_path.is_file():
                shard_path = manifest_relative_path
            else:
                ancestor_relative_path = None
                for ancestor in manifest_path.parents:
                    candidate = (ancestor / shard_path).resolve()
                    if candidate.is_file():
                        ancestor_relative_path = candidate
                        break
                if ancestor_relative_path is not None:
                    shard_path = ancestor_relative_path
                else:
                    shard_path = shard_path.resolve()
        else:
            shard_path = shard_path.resolve()

        if not shard_path.is_file():
            _emit_warning(
                warn_fn,
                f"Ignoring manifest {manifest_path.name}: listed shard does not exist ({shard_path}).",
            )
            return None
        resolved_paths.append(shard_path)

    manifest_path_set = {path.resolve() for path in resolved_paths}
    extra_pt_files = sorted(path.resolve() for path in data_dir.glob("*.pt") if path.is_file() and path.resolve() not in manifest_path_set)
    if extra_pt_files:
        _emit_warning(
            warn_fn,
            f"Found {len(extra_pt_files)} `.pt` file(s) in {data_dir} that are not listed in {manifest_path.name}; they will be ignored.",
        )

    return resolved_paths


def resolve_triangle_shard_paths(
    data_dir: str | Path,
    warn_fn: Callable[[str], None] | None = None,
) -> list[Path]:
    """Resolve shard files from one data directory using manifest-first logic.

    Resolution policy:
    1) If exactly one `.manifest.json` exists and is valid, use its shard list.
    2) If the manifest is missing, invalid, or ambiguous, fall back to direct
       non-recursive `*.pt` discovery under `data_dir`.

    Args:
        data_dir: Directory containing dataset shards.
        warn_fn: Optional warning sink.

    Returns:
        Ordered shard path list.
    """
    base = Path(data_dir).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"Data directory does not exist: {base}")

    manifest_files = sorted(path.resolve() for path in base.glob("*.manifest.json") if path.is_file())
    if len(manifest_files) == 1:
        resolved_manifest_paths = _resolve_manifest_shard_paths(manifest_files[0], base, warn_fn)
        if resolved_manifest_paths is not None:
            return resolved_manifest_paths
    elif len(manifest_files) > 1:
        _emit_warning(
            warn_fn,
            f"Found multiple manifest files under {base}; ignoring all manifests and falling back to direct `.pt` discovery.",
        )

    pt_files = sorted(path.resolve() for path in base.glob("*.pt") if path.is_file())
    if not pt_files:
        raise FileNotFoundError(f"No torch shard `.pt` files found directly under data directory: {base}")
    return pt_files
