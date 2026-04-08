"""Runtime bootstrap helpers for script entrypoints.

This module provides lightweight environment-compatibility hooks used by
launcher scripts before importing heavy modules (for example, torch).

Why this exists:
1) Some environments have CUDA runtime libraries installed in user-level
   site-packages (for example, `~/.local/.../nvidia/*/lib`) but not in
   `LD_LIBRARY_PATH`.
2) CUDA-enabled torch import can fail early with errors like
   `libcudnn.so.9: cannot open shared object file`.

To keep launcher commands simple, we attempt to detect this situation and
re-exec the current process once with a patched `LD_LIBRARY_PATH`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


_BOOTSTRAP_FLAG = "POLY2VEC_CUDA_LIB_BOOTSTRAPPED"
_REQUIRED_CUDA_SOS = ("libcudnn.so.9", "libcupti.so.12")


def _can_load_shared_object(so_name: str) -> bool:
    """Check whether a shared object can be loaded by dynamic linker.

    Args:
        so_name: Shared library filename (for example, `libcudnn.so.9`).

    Returns:
        True if loading succeeds, otherwise False.
    """
    try:
        ctypes.CDLL(so_name)
        return True
    except OSError:
        return False


def _find_candidate_cuda_lib_dirs() -> list[str]:
    """Discover candidate CUDA runtime library directories.

    Returns:
        Ordered list of existing directory paths.
    """
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    home = Path.home()
    prefixes = [
        home / ".local" / "lib" / py_ver / "site-packages" / "nvidia",
        Path(sys.prefix) / "lib" / py_ver / "site-packages" / "nvidia",
    ]

    dirs: list[str] = []
    seen: set[str] = set()
    for prefix in prefixes:
        if not prefix.exists():
            continue
        for child in sorted(prefix.glob("*/lib")):
            if child.is_dir():
                path_str = str(child)
                if path_str not in seen:
                    seen.add(path_str)
                    dirs.append(path_str)
    return dirs


def ensure_cuda_runtime_libs() -> None:
    """Ensure CUDA runtime library search path is usable for torch import.

    Behavior:
    1) If required CUDA shared objects are already loadable, do nothing.
    2) Otherwise, discover candidate `nvidia/*/lib` directories.
    3) Re-exec current process once with updated `LD_LIBRARY_PATH`.

    Notes:
        This function is intentionally called very early in launcher scripts,
        before importing modules that transitively import torch.
    """
    if os.environ.get(_BOOTSTRAP_FLAG) == "1":
        return

    missing = [name for name in _REQUIRED_CUDA_SOS if not _can_load_shared_object(name)]
    if not missing:
        return

    candidate_dirs = _find_candidate_cuda_lib_dirs()
    if not candidate_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    merged = ":".join(candidate_dirs + ([current] if current else []))

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = merged
    env[_BOOTSTRAP_FLAG] = "1"

    # Re-exec process so dynamic linker takes the new library path into account.
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
