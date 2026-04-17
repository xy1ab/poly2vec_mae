"""Precision policy helpers for training and inference.

This module standardizes precision naming and autocast behavior so that
trainer, pipeline, and scripts share one consistent precision policy.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


_SUPPORTED_PRECISIONS = {"fp32", "fp16", "bf16"}
_ALIAS_MAP = {
    "float32": "fp32",
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
}


def normalize_precision(precision: str) -> str:
    """Normalize precision aliases to canonical values.

    Args:
        precision: User-provided precision string.

    Returns:
        Canonical precision in {"fp32", "fp16", "bf16"}.

    Raises:
        ValueError: If precision is unsupported.
    """
    if precision is None:
        return "fp32"
    key = str(precision).strip().lower()
    key = _ALIAS_MAP.get(key, key)
    if key not in _SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}. Supported: {_SUPPORTED_PRECISIONS}")
    return key


def precision_to_torch_dtype(precision: str) -> torch.dtype:
    """Convert canonical precision name to torch dtype.

    Args:
        precision: Canonical precision string.

    Returns:
        Matching torch floating dtype.
    """
    precision = normalize_precision(precision)
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_precision_for_device(device: torch.device | str, precision: str) -> str:
    """Resolve precision against hardware capability.

    Args:
        device: Runtime device.
        precision: Requested precision.

    Returns:
        Runtime precision that is safe for the current device.
    """
    precision = normalize_precision(precision)
    dev = torch.device(device)

    if dev.type != "musa" and precision in {"fp16", "bf16"}:
        return "fp32"

    if dev.type == "musa" and precision == "bf16" and not torch.musa.is_bf16_supported():
        return "fp16"

    return precision


def should_enable_grad_scaler(device: torch.device | str, precision: str) -> bool:
    """Check whether GradScaler should be enabled.

    Args:
        device: Runtime device.
        precision: Requested precision.

    Returns:
        True only when running fp16 on MUSA.
    """
    dev = torch.device(device)
    resolved = resolve_precision_for_device(dev, precision)
    return dev.type == "musa" and resolved == "fp16"


def build_grad_scaler(device: torch.device | str, precision: str) -> torch.amp.GradScaler | torch.musa.amp.GradScaler:
    """Create a GradScaler instance with broad PyTorch-version compatibility.

    Args:
        device: Runtime device.
        precision: Requested precision string.

    Returns:
        A GradScaler configured for current device/precision policy.

    Notes:
        Newer PyTorch recommends `torch.amp.GradScaler(...)`, while older
        versions only provide `torch.musa.amp.GradScaler(...)`. This helper
        prefers the new API and gracefully falls back to the legacy one.
    """
    dev = torch.device(device)
    enabled = should_enable_grad_scaler(dev, precision)

    amp_module = getattr(torch, "amp", None)
    amp_grad_scaler = getattr(amp_module, "GradScaler", None) if amp_module is not None else None
    if amp_grad_scaler is not None:
        try:
            return amp_grad_scaler(device=dev.type, enabled=enabled)
        except TypeError:
            # Older torch.amp API variants may require positional `device_type`.
            try:
                return amp_grad_scaler(dev.type, enabled=enabled)
            except TypeError:
                pass

    return torch.musa.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device | str, precision: str) -> ContextManager:
    """Build an autocast context manager for current runtime precision.

    Args:
        device: Runtime device.
        precision: Requested precision.

    Returns:
        A context manager. Returns nullcontext when autocast is unnecessary.
    """
    dev = torch.device(device)
    resolved = resolve_precision_for_device(dev, precision)
    if dev.type != "musa" or resolved == "fp32":
        return nullcontext()
    return torch.autocast(device_type="musa", dtype=precision_to_torch_dtype(resolved))
