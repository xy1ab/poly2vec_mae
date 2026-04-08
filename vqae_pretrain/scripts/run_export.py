"""Export VQAE component checkpoints from one full checkpoint directory."""

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
        "vqae_pretrain.scripts.runtime_bootstrap"
    ).ensure_cuda_runtime_libs
else:
    from .runtime_bootstrap import ensure_cuda_runtime_libs


def _inject_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    repo_root = project_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return project_root


def main() -> None:
    ensure_cuda_runtime_libs()
    project_root = _inject_repo_root()

    if __package__ in {None, ""}:
        import importlib

        export_vqae_components = importlib.import_module(
            "vqae_pretrain.src.models.factory"
        ).export_vqae_components
    else:
        from ..src.models.factory import export_vqae_components

    parser = argparse.ArgumentParser(description="Export VQAE components from a full checkpoint")
    parser.add_argument("--vqae_ckpt_path", type=str, required=True, help="Full VQAE checkpoint path.")
    parser.add_argument("--config_path", type=str, required=True, help="Config path associated with the checkpoint.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "outputs" / "exports" / "vqae_bundle"),
        help="Export directory.",
    )
    parser.add_argument("--precision", type=str, default="fp32", help="Float precision for exported tensor checkpoints.")
    args = parser.parse_args()

    export_dir = export_vqae_components(
        vqae_ckpt_path=args.vqae_ckpt_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        precision=args.precision,
    )
    print(f"[INFO] Exported VQAE bundle to: {export_dir}")


if __name__ == "__main__":
    main()
