"""Layout tests for ae_pretrain package."""

from __future__ import annotations

from pathlib import Path


def test_classic_layout_exists() -> None:
    """Ensure classic deep-learning folder layout is present."""
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "configs",
        root / "scripts",
        root / "data",
        root / "outputs",
        root / "src",
        root / "tests",
    ]
    for path in required:
        assert path.is_dir(), f"missing directory: {path}"


def test_src_subpackages_exist() -> None:
    """Ensure source subpackages match expected architecture."""
    root = Path(__file__).resolve().parents[1] / "src"
    required = ["models", "datasets", "losses", "engine", "utils"]
    for name in required:
        assert (root / name).is_dir(), f"missing src package: {name}"


def test_configs_are_flat_and_complete() -> None:
    """Ensure configs folder is flat and contains required config files."""
    configs_root = Path(__file__).resolve().parents[1] / "configs"
    required_files = {
        "pretrain_base.yaml",
        "eval_default.yaml",
        "export_default.yaml",
    }

    discovered_files = {path.name for path in configs_root.glob("*.yaml")}
    assert required_files.issubset(discovered_files), f"missing config files: {required_files - discovered_files}"

    nested_dirs = [path for path in configs_root.iterdir() if path.is_dir()]
    assert not nested_dirs, f"configs should be flat, found subdirectories: {nested_dirs}"
