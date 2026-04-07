"""Configuration loading and merging helpers for AE pretraining.

This module centralizes config parsing so that scripts and engines do not
duplicate YAML/JSON handling logic. It also provides a small merge utility that
applies CLI overrides on top of file-based defaults in a predictable way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: YAML file path.

    Returns:
        Parsed configuration dictionary. Empty dict if file is empty.
    """
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def load_json_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON configuration file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON dictionary.
    """
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_config_any(path: str | Path) -> Dict[str, Any]:
    """Load YAML or JSON config based on file extension.

    Args:
        path: Config path ending in .yaml/.yml or .json.

    Returns:
        Parsed config dictionary.

    Raises:
        ValueError: If extension is unsupported.
    """
    path_str = str(path).lower()
    if path_str.endswith((".yaml", ".yml")):
        return load_yaml_config(path)
    if path_str.endswith(".json"):
        return load_json_config(path)
    raise ValueError(f"Unsupported config file extension: {path}")


def merge_cli_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    skip_none: bool = True,
) -> Dict[str, Any]:
    """Merge flat CLI overrides into a flat base config.

    Args:
        base: Base configuration dictionary from file.
        overrides: Parsed CLI args dictionary.
        skip_none: Whether `None` values in overrides should be ignored.

    Returns:
        Merged dictionary where overrides take precedence.
    """
    merged: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if key == "config":
            continue
        if skip_none and value is None:
            continue
        merged[key] = value
    return merged


def dump_yaml_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Serialize a config dictionary into a YAML file.

    Args:
        config: Configuration dictionary to dump.
        path: Output YAML path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(dict(config), fp, allow_unicode=True, sort_keys=False)
