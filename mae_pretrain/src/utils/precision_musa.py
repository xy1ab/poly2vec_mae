"""Precision policy helpers for MUSA-oriented training and inference.

This module is a MUSA-aware counterpart of `utils/precision.py`. It keeps the
same public function contracts so callers can switch imports with minimal code
changes while preserving behavior consistency.

Design goals:
1) Keep precision naming and alias handling identical to the default module.
2) Resolve precision against runtime device capability for cpu/cuda/musa.
3) Provide autocast and GradScaler creation with broad API compatibility.
4) Fail gracefully when optional backend-specific APIs are unavailable.
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
    "fp13": "fp16",  # compatibility alias requested by user
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


<<<<<<< HEAD
def resolve_precision_for_device(device: torch.device | str, precision: str) -> str:
    """Resolve precision against hardware capability.
=======
def _is_bf16_supported_on_cuda() -> bool:
    """Check bf16 support on CUDA devices.

    Returns:
        True when CUDA bf16 is supported, otherwise False.
    """
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def _is_bf16_supported_on_musa() -> bool:
    """Check bf16 support on MUSA devices.

    Returns:
        True when MUSA bf16 is supported, otherwise False.
    """
    musa_mod = getattr(torch, "musa", None)
    if musa_mod is None:
        return False

    # Prefer explicit runtime capability API when available.
    checker = getattr(musa_mod, "is_bf16_supported", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            pass

    # Fallback: if MUSA is not available, bf16 cannot be used.
    is_avail = getattr(musa_mod, "is_available", None)
    if callable(is_avail):
        try:
            return bool(is_avail())
        except Exception:
            return False

    return False


def resolve_precision_for_device(device: torch.device | str, precision: str) -> str:
    """Resolve requested precision against runtime device capability.

    Rules:
    1) CPU runs in fp32.
    2) CUDA and MUSA keep fp16 by default.
    3) CUDA/MUSA bf16 downgrades to fp16 when bf16 support is unavailable.
>>>>>>> bb87083 (precision_musa.py was added.)

    Args:
        device: Runtime device.
        precision: Requested precision.

    Returns:
<<<<<<< HEAD
        Runtime precision that is safe for the current device.
=======
        Runtime-safe precision string in {"fp32", "fp16", "bf16"}.
>>>>>>> bb87083 (precision_musa.py was added.)
    """
    precision = normalize_precision(precision)
    dev = torch.device(device)

<<<<<<< HEAD
    if dev.type != "musa" and precision in {"fp16", "bf16"}:
        return "fp32"

    if dev.type == "musa" and precision == "bf16" and not torch.musa.is_bf16_supported():
=======
    if dev.type not in {"cuda", "musa"} and precision in {"fp16", "bf16"}:
        return "fp32"

    if dev.type == "cuda" and precision == "bf16" and not _is_bf16_supported_on_cuda():
        return "fp16"

    if dev.type == "musa" and precision == "bf16" and not _is_bf16_supported_on_musa():
>>>>>>> bb87083 (precision_musa.py was added.)
        return "fp16"

    return precision


def should_enable_grad_scaler(device: torch.device | str, precision: str) -> bool:
    """Check whether GradScaler should be enabled.

    Args:
        device: Runtime device.
        precision: Requested precision.

    Returns:
<<<<<<< HEAD
        True only when running fp16 on MUSA.
    """
    dev = torch.device(device)
    resolved = resolve_precision_for_device(dev, precision)
    return dev.type == "musa" and resolved == "fp16"


def build_grad_scaler(device: torch.device | str, precision: str) -> torch.amp.GradScaler | torch.musa.amp.GradScaler:
    """Create a GradScaler instance with broad PyTorch-version compatibility.
=======
        True when running fp16 on CUDA/MUSA.
    """
    dev = torch.device(device)
    resolved = resolve_precision_for_device(dev, precision)
    return dev.type in {"cuda", "musa"} and resolved == "fp16"


def build_grad_scaler(device: torch.device | str, precision: str):
    """Create a backend-aware GradScaler with API compatibility fallbacks.
>>>>>>> bb87083 (precision_musa.py was added.)

    Args:
        device: Runtime device.
        precision: Requested precision string.

    Returns:
<<<<<<< HEAD
        A GradScaler configured for current device/precision policy.

    Notes:
        Newer PyTorch recommends `torch.amp.GradScaler(...)`, while older
        versions only provide `torch.musa.amp.GradScaler(...)`. This helper
        prefers the new API and gracefully falls back to the legacy one.
=======
        A GradScaler instance compatible with current runtime.
>>>>>>> bb87083 (precision_musa.py was added.)
    """
    dev = torch.device(device)
    enabled = should_enable_grad_scaler(dev, precision)

<<<<<<< HEAD
=======
    # Newer API path.
>>>>>>> bb87083 (precision_musa.py was added.)
    amp_module = getattr(torch, "amp", None)
    amp_grad_scaler = getattr(amp_module, "GradScaler", None) if amp_module is not None else None
    if amp_grad_scaler is not None:
        try:
            return amp_grad_scaler(device=dev.type, enabled=enabled)
        except TypeError:
<<<<<<< HEAD
            # Older torch.amp API variants may require positional `device_type`.
=======
>>>>>>> bb87083 (precision_musa.py was added.)
            try:
                return amp_grad_scaler(dev.type, enabled=enabled)
            except TypeError:
                pass

<<<<<<< HEAD
    return torch.musa.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device | str, precision: str) -> ContextManager:
    """Build an autocast context manager for current runtime precision.
=======
    # MUSA legacy API path.
    if dev.type == "musa":
        musa_mod = getattr(torch, "musa", None)
        musa_amp = getattr(musa_mod, "amp", None) if musa_mod is not None else None
        musa_grad_scaler = getattr(musa_amp, "GradScaler", None) if musa_amp is not None else None
        if musa_grad_scaler is not None:
            return musa_grad_scaler(enabled=enabled)

    # CUDA legacy API path (kept for backward compatibility).
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device | str, precision: str) -> ContextManager:
    """Build autocast context manager for cpu/cuda/musa runtime.
>>>>>>> bb87083 (precision_musa.py was added.)

    Args:
        device: Runtime device.
        precision: Requested precision.

    Returns:
        A context manager. Returns nullcontext when autocast is unnecessary.
    """
    dev = torch.device(device)
    resolved = resolve_precision_for_device(dev, precision)
<<<<<<< HEAD
    if dev.type != "musa" or resolved == "fp32":
        return nullcontext()
    return torch.autocast(device_type="musa", dtype=precision_to_torch_dtype(resolved))
=======
    if dev.type not in {"cuda", "musa"} or resolved == "fp32":
        return nullcontext()

    return torch.autocast(
        device_type=dev.type,
        dtype=precision_to_torch_dtype(resolved),
    )

>>>>>>> bb87083 (precision_musa.py was added.)
