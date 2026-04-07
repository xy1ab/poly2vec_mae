"""Model package exports for MAE pretraining.

This package keeps top-level imports lazy so lightweight callers can access
factory symbols without eagerly importing every model submodule.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PolyEncoder",
    "MaskedAutoencoderViTPoly",
    "build_mae_model_from_config",
    "load_mae_model",
    "load_pretrained_encoder",
    "export_encoder_from_mae_checkpoint",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "PolyEncoder": (".encoder", "PolyEncoder"),
    "MaskedAutoencoderViTPoly": (".mae", "MaskedAutoencoderViTPoly"),
    "build_mae_model_from_config": (".factory", "build_mae_model_from_config"),
    "load_mae_model": (".factory", "load_mae_model"),
    "load_pretrained_encoder": (".factory", "load_pretrained_encoder"),
    "export_encoder_from_mae_checkpoint": (".factory", "export_encoder_from_mae_checkpoint"),
}


def __getattr__(name: str) -> Any:
    """Resolve model exports lazily on first access.

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
