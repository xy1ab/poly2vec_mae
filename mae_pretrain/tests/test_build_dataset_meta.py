"""Meta-output tests for dataset triangulation builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("triangle")

from shapely.geometry import Polygon

from ..src.datasets import build_dataset_triangle as bdt


def test_build_meta_output_path_suffix_swap() -> None:
    """`*_tri.pt` should map to paired `*_meta.pt`."""
    tri_path = Path("/tmp/example_tri.pt")
    meta_path = bdt._build_meta_output_path(tri_path)
    assert meta_path.name == "example_meta.pt"


def test_build_row_meta4_from_polygon() -> None:
    """Meta vector should follow `[cx, cy, L, Lx, Ly, N]` definition."""
    poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)])
    meta = bdt._build_row_meta4(poly)

    assert meta.dtype == np.float32
    assert meta.shape == (6,)
    assert np.isclose(meta[0], 2.0)  # cx
    assert np.isclose(meta[1], 1.0)  # cy
    assert np.isclose(meta[2], 4.0)  # longest bbox side length
    assert np.isclose(meta[3], 4.0)  # bbox width
    assert np.isclose(meta[4], 2.0)  # bbox height
    assert np.isclose(meta[5], 4.0)  # cleaned shell node count


def test_shard_writer_finalize_writes_manifest_without_samples(tmp_path) -> None:
    """Manifest should always be written even when no sample is emitted."""
    tri_path = tmp_path / "demo_tri.pt"
    writer = bdt._ShardWriter(output_path=tri_path, shard_size_mb=0.0, with_meta=True)
    shard_paths, manifest_path = writer.finalize()

    assert shard_paths == []
    assert manifest_path is not None
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["num_shards"] == 0
    assert payload["total_samples"] == 0
    assert payload["shards"] == []
    assert payload["meta_base_output_path"].endswith("demo_meta.pt")
    assert payload["meta_shards"] == []
