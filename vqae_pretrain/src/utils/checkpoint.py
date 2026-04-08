"""Checkpoint save/export helpers for VQAE pretraining.

This module centralizes:
1) Precision-cast model checkpoint export.
2) Atomic training-state checkpoint save/load for resume.
3) Simple `a/b` rotation of the latest epoch checkpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch

from .config import dump_yaml_config
from .filesystem import copy_if_exists, ensure_dir
from .precision import normalize_precision, precision_to_torch_dtype


def _move_tensors_to_cpu(data: Any) -> Any:
    """Recursively move tensors in a nested object to CPU.

    Args:
        data: Arbitrary nested Python object.

    Returns:
        Object of the same structure with tensors detached onto CPU.
    """
    if torch.is_tensor(data):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {key: _move_tensors_to_cpu(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_move_tensors_to_cpu(value) for value in data]
    if isinstance(data, tuple):
        return tuple(_move_tensors_to_cpu(value) for value in data)
    return data


def _atomic_torch_save(obj: Any, path: str | Path) -> Path:
    """Atomically save a PyTorch object to disk.

    Args:
        obj: Serializable Python object.
        path: Target output path.

    Returns:
        Final saved path.
    """
    out_path = Path(path)
    ensure_dir(out_path.parent)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")

    with tmp_path.open("wb") as fp:
        torch.save(obj, fp)
        fp.flush()
        os.fsync(fp.fileno())

    os.replace(tmp_path, out_path)
    return out_path


def cast_state_dict_floats(state_dict: Mapping[str, Any], precision: str) -> dict[str, Any]:
    """Cast floating tensors in a state dict to target precision.

    Args:
        state_dict: Input state dict to convert.
        precision: Target precision string.

    Returns:
        Converted state dict (CPU tensors).
    """
    precision = normalize_precision(precision)
    target_dtype = precision_to_torch_dtype(precision)

    converted: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=target_dtype)
            converted[key] = tensor
        else:
            converted[key] = value
    return converted


def save_checkpoint(path: str | Path, state_dict: Mapping[str, Any], precision: str = "fp32") -> Path:
    """Save a checkpoint file with optional float cast.

    Args:
        path: Output checkpoint path.
        state_dict: Model state dict.
        precision: Output float precision.

    Returns:
        Saved checkpoint path.
    """
    return _atomic_torch_save(cast_state_dict_floats(state_dict, precision), path)


def save_training_state(path: str | Path, state: Mapping[str, Any]) -> Path:
    """Save one full training-state checkpoint without lossy dtype casting.

    Args:
        path: Output checkpoint path.
        state: Training-state dictionary.

    Returns:
        Saved checkpoint path.
    """
    return _atomic_torch_save(_move_tensors_to_cpu(dict(state)), path)


def save_latest_training_state_pair(
    ckpt_dir: str | Path,
    state: Mapping[str, Any],
    latest_name: str = "train_state_a.pth",
    previous_name: str = "train_state_b.pth",
) -> tuple[Path, Path]:
    """Save newest training checkpoint and keep exactly one previous backup.

    Rotation policy:
    1) Existing `a` is copied to `b`.
    2) New state is atomically written to `a`.

    Args:
        ckpt_dir: Checkpoint directory.
        state: Training-state dictionary.
        latest_name: Filename of the newest checkpoint.
        previous_name: Filename of the previous checkpoint.

    Returns:
        Tuple `(latest_path, previous_path)`.
    """
    ckpt_root = ensure_dir(ckpt_dir)
    latest_path = ckpt_root / latest_name
    previous_path = ckpt_root / previous_name

    if latest_path.exists():
        tmp_previous = previous_path.with_name(f"{previous_path.name}.tmp")
        shutil.copy2(latest_path, tmp_previous)
        os.replace(tmp_previous, previous_path)

    save_training_state(latest_path, state)
    return latest_path, previous_path


def load_training_state(path: str | Path) -> dict[str, Any]:
    """Load one training-state checkpoint.

    Args:
        path: Checkpoint path.

    Returns:
        Loaded checkpoint dictionary on CPU.
    """
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_latest_training_state(
    resume_dir: str | Path,
    latest_name: str = "train_state_a.pth",
    previous_name: str = "train_state_b.pth",
) -> tuple[dict[str, Any], Path]:
    """Load the newest valid training-state checkpoint from one run directory.

    The function tries `a` first and falls back to `b` when necessary.

    Args:
        resume_dir: Run directory that contains `ckpt/`.
        latest_name: Filename of the newest checkpoint.
        previous_name: Filename of the previous checkpoint.

    Returns:
        Tuple `(state_dict, loaded_path)`.
    """
    ckpt_dir = Path(resume_dir).expanduser().resolve() / "ckpt"
    candidate_paths = [ckpt_dir / latest_name, ckpt_dir / previous_name]

    last_error: Exception | None = None
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            return load_training_state(path), path
        except Exception as exc:  # pragma: no cover - exercised in real resume fallback paths
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(f"Failed to load any resume checkpoint under: {ckpt_dir}") from last_error
    raise FileNotFoundError(f"No resume checkpoint found under: {ckpt_dir}")

