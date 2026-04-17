"""Filesystem helper utilities for VQAE pretraining.

This module keeps file and directory path operations in one place so training
and scripts can stay focused on algorithmic logic.
"""

from __future__ import annotations

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
