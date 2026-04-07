"""Engine package exports for downstream pipeline APIs.

This package keeps its top-level import surface lazy so importing
`mae_pretrain.src.engine` does not eagerly pull in deep model/runtime
dependencies unless a pipeline symbol is actually requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["PolyEncoderPipeline", "PolyMaeReconstructionPipeline"]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "PolyEncoderPipeline": (".pipeline", "PolyEncoderPipeline"),
    "PolyMaeReconstructionPipeline": (".pipeline", "PolyMaeReconstructionPipeline"),
}


def __getattr__(name: str) -> Any:
    """Resolve engine exports lazily on first access.

    Args:
        name: Exported symbol name.

    Returns:
        Requested symbol object.

    Raises:
        AttributeError: If the symbol is not a declared package export.
    """
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
