"""Model package exports for VQAE pretraining."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PolyEncoder",
    "PolyDecoder",
    "PolyVqAutoencoder",
    "EMAVectorQuantizer",
    "build_vqae_model_from_config",
    "load_vqae_model",
    "load_decoder_from_components",
    "load_pretrained_encoder",
    "export_vqae_components",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "PolyEncoder": (".encoder", "PolyEncoder"),
    "PolyDecoder": (".decoder", "PolyDecoder"),
    "PolyVqAutoencoder": (".vqae", "PolyVqAutoencoder"),
    "EMAVectorQuantizer": (".quantizer", "EMAVectorQuantizer"),
    "build_vqae_model_from_config": (".factory", "build_vqae_model_from_config"),
    "load_vqae_model": (".factory", "load_vqae_model"),
    "load_decoder_from_components": (".factory", "load_decoder_from_components"),
    "load_pretrained_encoder": (".factory", "load_pretrained_encoder"),
    "export_vqae_components": (".factory", "export_vqae_components"),
}


def __getattr__(name: str) -> Any:
    """Resolve model exports lazily on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
