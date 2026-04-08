"""MUSA trainer skeleton for polygon VQAE pretraining.

This file is intentionally kept minimal in the first CUDA-focused delivery.
It preserves the CLI boundary and makes later MUSA adaptation straightforward
without duplicating the full CUDA trainer implementation here.
"""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a minimal MUSA trainer parser aligned with the CUDA entrypoint."""
    parser = argparse.ArgumentParser(description="VQAE pretraining trainer (MUSA skeleton)")
    parser.add_argument("--config", type=str, default=None, help="Reserved for future MUSA implementation.")
    return parser


def train_main(_args) -> None:
    """Placeholder MUSA entrypoint."""
    raise NotImplementedError(
        "vqae_pretrain MUSA support is intentionally left as a skeleton in this first version. "
        "Please adapt `src/engine/trainer_musa.py` based on the completed CUDA trainer."
    )


def run_cli(argv=None) -> None:
    """CLI wrapper for the MUSA skeleton trainer."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train_main(args)
