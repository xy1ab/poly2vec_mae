"""Checkpoint export launcher script.

This script exports an encoder-only checkpoint from a full AE checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _CURRENT_DIR.parent
    _REPO_ROOT = _PROJECT_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib

    ensure_cuda_runtime_libs = importlib.import_module(
        "ae_pretrain.scripts.runtime_bootstrap"
    ).ensure_cuda_runtime_libs
else:
    from .runtime_bootstrap import ensure_cuda_runtime_libs


def _inject_repo_root() -> Path:
    """Inject repository root into `sys.path` for direct script execution.

    Returns:
        `ae_pretrain` project root path.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def main() -> None:
    """CLI main function for encoder export."""
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    if __package__ in {None, ""}:
        import importlib

        export_encoder_from_ae_checkpoint = importlib.import_module(
            "ae_pretrain.src.models.factory"
        ).export_encoder_from_ae_checkpoint
        load_yaml_config = importlib.import_module(
            "ae_pretrain.src.utils.config"
        ).load_yaml_config
    else:
        from ..src.models.factory import export_encoder_from_ae_checkpoint
        from ..src.utils.config import load_yaml_config

    parser = argparse.ArgumentParser(description="Export encoder from AE checkpoint")
    parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "export_default.yaml"),
        type=str,
    )
    parser.add_argument("--ae_ckpt_path", type=str, default=None)
    parser.add_argument("--mae_ckpt_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    ae_ckpt_path = args.ae_ckpt_path or args.mae_ckpt_path or cfg.get("ae_ckpt_path") or cfg.get("mae_ckpt_path")
    config_path = args.config_path or cfg.get("config_path")
    output_path = args.output_path or cfg.get("output_path")
    precision = args.precision or cfg.get("precision", "bf16")

    if not ae_ckpt_path or not config_path or not output_path:
        raise ValueError("ae_ckpt_path, config_path, output_path must be provided via config or CLI")

    saved_path = export_encoder_from_ae_checkpoint(
        ae_ckpt_path=ae_ckpt_path,
        config_path=config_path,
        output_path=output_path,
        precision=precision,
    )
    print(f"[INFO] Exported encoder checkpoint to: {saved_path}")


if __name__ == "__main__":
    main()
