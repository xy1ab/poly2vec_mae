"""Utility package for AE pretraining.

This package contains infrastructure helpers that are intentionally generic and
low-level. Modules here should avoid training/business logic and instead provide
small reusable capabilities such as configuration loading, precision handling,
checkpoint export, filesystem helpers, logging, and distributed initialization.
"""

from .config import load_config_any, load_json_config, load_yaml_config
from .filesystem import ensure_dir, make_timestamped_dir
from .precision import autocast_context, normalize_precision, precision_to_torch_dtype

__all__ = [
    "load_config_any",
    "load_json_config",
    "load_yaml_config",
    "ensure_dir",
    "make_timestamped_dir",
    "autocast_context",
    "normalize_precision",
    "precision_to_torch_dtype",
]
