"""Model package exports for RVQAE pretraining."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PolyEncoder",
    "PolyDecoder",
    "PolyRvqAutoencoder",
    "EMAVectorQuantizer",
    "ResidualEMAVectorQuantizer",
    "build_rvqae_model_from_config",
    "load_rvqae_model",
    "load_decoder_from_components",
    "load_pretrained_encoder",
    "export_rvqae_components",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "PolyEncoder": (".encoder", "PolyEncoder"),
    "PolyDecoder": (".decoder", "PolyDecoder"),
    "PolyRvqAutoencoder": (".rvqae", "PolyRvqAutoencoder"),
    "EMAVectorQuantizer": (".quantizer", "EMAVectorQuantizer"),
    "ResidualEMAVectorQuantizer": (".quantizer", "ResidualEMAVectorQuantizer"),
    "build_rvqae_model_from_config": (".factory", "build_rvqae_model_from_config"),
    "load_rvqae_model": (".factory", "load_rvqae_model"),
    "load_decoder_from_components": (".factory", "load_decoder_from_components"),
    "load_pretrained_encoder": (".factory", "load_pretrained_encoder"),
    "export_rvqae_components": (".factory", "export_rvqae_components"),
}


def __getattr__(name: str) -> Any:
    """Resolve model exports lazily on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
