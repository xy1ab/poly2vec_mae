"""Engine package exports for VQAE pretraining."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["PolyEncoderPipeline", "PolyAeReconstructionPipeline"]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "PolyEncoderPipeline": (".pipeline", "PolyEncoderPipeline"),
    "PolyAeReconstructionPipeline": (".pipeline", "PolyAeReconstructionPipeline"),
    "PolyMaeReconstructionPipeline": (".pipeline", "PolyMaeReconstructionPipeline"),
}


def __getattr__(name: str) -> Any:
    """Resolve engine exports lazily on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
