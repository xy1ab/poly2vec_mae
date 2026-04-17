"""Random seed helpers for reproducible training.

This module provides one entrypoint to initialize Python, NumPy, and PyTorch
random states in a consistent way across scripts and training processes.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set global random seed for Python, NumPy, and PyTorch.

    Args:
        seed: Global integer seed.
        deterministic: Whether to enable deterministic CUDA backend flags.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    """Capture Python, NumPy, and Torch RNG states for later restoration.

    Returns:
        Serializable RNG state dictionary.
    """
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    """Restore RNG states captured by :func:`capture_rng_state`.

    Args:
        state: RNG state dictionary or `None`.
    """
    if not state:
        return

    python_state = state.get("python_random")
    numpy_state = state.get("numpy_random")
    torch_state = state.get("torch_random")
    cuda_state = state.get("cuda_random_all")

    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
