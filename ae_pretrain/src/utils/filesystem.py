"""Filesystem helper utilities for AE pretraining.

This module keeps file and directory path operations in one place so training
and scripts can stay focused on algorithmic logic.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory recursively if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        Path object for the created/existing directory.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def make_timestamped_dir(base_dir: str | Path, fmt: str = "%Y%m%d_%H%M") -> tuple[Path, str]:
    """Create and return a timestamped run directory.

    Args:
        base_dir: Parent directory in which the run directory will be created.
        fmt: Datetime format for the run folder name.

    Returns:
        A tuple of `(run_dir, run_timestamp)`.
    """
    run_timestamp = datetime.now().strftime(fmt)
    run_dir = ensure_dir(Path(base_dir) / run_timestamp)
    return run_dir, run_timestamp


def copy_if_exists(src: str | Path, dst: str | Path) -> bool:
    """Copy a file only when the source exists.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        True if file was copied, False if source does not exist.
    """
    src_path = Path(src)
    if not src_path.exists() or not src_path.is_file():
        return False

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(src_path.read_bytes())
    return True
