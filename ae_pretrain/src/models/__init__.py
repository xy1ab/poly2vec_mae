"""Model package exports for AE pretraining."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PolyEncoder",
    "PolyDecoder",
    "PolyAutoencoder",
    "build_ae_model_from_config",
    "load_ae_model",
    "load_pretrained_encoder",
    "export_encoder_from_ae_checkpoint",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "PolyEncoder": (".encoder", "PolyEncoder"),
    "PolyDecoder": (".decoder", "PolyDecoder"),
    "PolyAutoencoder": (".mae", "PolyAutoencoder"),
    "MaskedAutoencoderViTPoly": (".mae", "MaskedAutoencoderViTPoly"),
    "build_ae_model_from_config": (".factory", "build_ae_model_from_config"),
    "load_ae_model": (".factory", "load_ae_model"),
    "load_pretrained_encoder": (".factory", "load_pretrained_encoder"),
    "export_encoder_from_ae_checkpoint": (".factory", "export_encoder_from_ae_checkpoint"),
    "build_mae_model_from_config": (".factory", "build_mae_model_from_config"),
    "load_mae_model": (".factory", "load_mae_model"),
    "export_encoder_from_mae_checkpoint": (".factory", "export_encoder_from_mae_checkpoint"),
}


def __getattr__(name: str) -> Any:
    """Resolve model exports lazily on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
