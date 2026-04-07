"""Precision policy tests for ae_pretrain utils."""

from __future__ import annotations

import pytest

from ..src.utils.precision import normalize_precision


def test_precision_fp13_is_rejected() -> None:
    """Ensure unsupported typo precision `fp13` is rejected."""
    with pytest.raises(ValueError):
        normalize_precision("fp13")


def test_precision_keeps_bf16() -> None:
    """Ensure bf16 remains unchanged."""
    assert normalize_precision("bf16") == "bf16"
