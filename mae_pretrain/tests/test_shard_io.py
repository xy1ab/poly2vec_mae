"""Tests for torch-only shard IO and manifest-first shard discovery."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from ..src.datasets.shard_io import (
    TORCH_SHARD_SERIALIZATION,
    load_triangle_shard,
    resolve_triangle_shard_paths,
    save_triangle_shard,
)


def _sample(value: float) -> np.ndarray:
    """Create one tiny triangle sample."""
    return np.full((1, 3, 2), value, dtype=np.float32)


def _write_manifest(manifest_path: Path, shards: list[Path], serialization: str = TORCH_SHARD_SERIALIZATION) -> None:
    """Write one minimal shard manifest JSON file."""
    payload = {
        "serialization": serialization,
        "shards": [{"path": path.name, "sample_count": 1, "size_bytes": int(path.stat().st_size)} for path in shards],
    }
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def test_load_triangle_shard_rejects_legacy_pickle(tmp_path: Path) -> None:
    """Torch-only loader should fail fast on legacy pickle shards."""
    shard_path = tmp_path / "legacy.pt"
    with shard_path.open("wb") as fp:
        pickle.dump([_sample(1.0)], fp)

    with pytest.raises(RuntimeError, match="Legacy pickle shards are no longer supported"):
        load_triangle_shard(shard_path)


def test_resolve_triangle_shard_paths_prefers_valid_manifest_and_ignores_extra_pt(tmp_path: Path) -> None:
    """A valid manifest should control shard order and ignore unmanaged `.pt` files."""
    shard_b = save_triangle_shard(tmp_path / "b.pt", [_sample(2.0)])
    shard_a = save_triangle_shard(tmp_path / "a.pt", [_sample(1.0)])
    save_triangle_shard(tmp_path / "extra.pt", [_sample(9.0)])
    _write_manifest(tmp_path / "dataset.manifest.json", [shard_b, shard_a])

    warnings: list[str] = []
    resolved = resolve_triangle_shard_paths(tmp_path, warn_fn=warnings.append)

    assert resolved == [shard_b, shard_a]
    assert any("not listed" in message for message in warnings)


def test_resolve_triangle_shard_paths_falls_back_when_manifest_is_invalid(tmp_path: Path) -> None:
    """Invalid manifest metadata should trigger warning + sorted glob fallback."""
    shard_b = save_triangle_shard(tmp_path / "b.pt", [_sample(2.0)])
    shard_a = save_triangle_shard(tmp_path / "a.pt", [_sample(1.0)])
    _write_manifest(tmp_path / "dataset.manifest.json", [shard_b, shard_a], serialization="legacy_pickle")

    warnings: list[str] = []
    resolved = resolve_triangle_shard_paths(tmp_path, warn_fn=warnings.append)

    assert resolved == [shard_a, shard_b]
    assert any("Ignoring manifest" in message for message in warnings)


def test_resolve_triangle_shard_paths_ignores_multiple_manifests(tmp_path: Path) -> None:
    """Multiple manifests should be ignored and direct shard discovery should take over."""
    shard_b = save_triangle_shard(tmp_path / "b.pt", [_sample(2.0)])
    shard_a = save_triangle_shard(tmp_path / "a.pt", [_sample(1.0)])
    _write_manifest(tmp_path / "first.manifest.json", [shard_b, shard_a])
    _write_manifest(tmp_path / "second.manifest.json", [shard_a, shard_b])

    warnings: list[str] = []
    resolved = resolve_triangle_shard_paths(tmp_path, warn_fn=warnings.append)

    assert resolved == [shard_a, shard_b]
    assert any("multiple manifest files" in message for message in warnings)
