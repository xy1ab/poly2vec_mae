"""MUSA launcher placeholder for polygon VQAE pretraining."""

from __future__ import annotations

import sys
from pathlib import Path


def _inject_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def main() -> None:
    _inject_repo_root()
    if __package__ in {None, ""}:
        import importlib

        run_cli = importlib.import_module("vqae_pretrain.src.engine.trainer_musa").run_cli
    else:
        from ..src.engine.trainer_musa import run_cli

    run_cli()


if __name__ == "__main__":
    main()
