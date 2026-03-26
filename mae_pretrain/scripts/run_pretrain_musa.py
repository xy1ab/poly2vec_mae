"""Pretraining launcher script.

This script reads YAML config defaults, applies CLI overrides, and launches the
trainer engine. It also supports optional auto-spawn for multi-GPU DDP runs.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from runtime_bootstrap import ensure_cuda_runtime_libs


def _inject_src_path() -> Path:
    """Inject local `src` directory into `sys.path`.

    Returns:
        Project root path.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return project_root


def _build_cli_args_from_config(config_dict: dict) -> list[str]:
    """Convert config dictionary into CLI argument list.

    Args:
        config_dict: Parsed YAML config dictionary.

    Returns:
        Flat CLI argument list.
    """
    cli_args: list[str] = []
    for key, value in config_dict.items():
        arg_name = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli_args.append(arg_name)
        else:
            cli_args.extend([arg_name, str(value)])
    return cli_args


def _split_gpu_list(gpu_value: str) -> list[str]:
    """Split comma-separated GPU string into normalized ID list.

    Args:
        gpu_value: Comma-separated GPU string.

    Returns:
        Non-empty GPU id list.
    """
    return [item.strip() for item in str(gpu_value).split(",") if item.strip()]


def _normalize_gpu_csv(gpu_list: list[str]) -> str:
    """Serialize normalized GPU id list to CSV string.

    Args:
        gpu_list: Parsed GPU id list.

    Returns:
        Comma-separated GPU id string.
    """
    return ",".join(gpu_list)


def _is_cuda_available() -> bool:
    """Check whether CUDA runtime is available for current Python process.

    Returns:
        True when torch can be imported and CUDA is available, else False.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _terminate_process_group(proc: subprocess.Popen, timeout_sec: float = 8.0) -> int | None:
    """Terminate a spawned process group with graceful fallback to SIGKILL.

    Args:
        proc: Managed subprocess object created by `subprocess.Popen`.
        timeout_sec: Grace period in seconds before force kill.

    Returns:
        Final process return code when available.
    """
    if proc.poll() is not None:
        return proc.returncode

    try:
        proc_group = os.getpgid(proc.pid)
    except ProcessLookupError:
        return proc.poll()

    try:
        os.killpg(proc_group, signal.SIGTERM)
    except ProcessLookupError:
        return proc.poll()

    deadline = time.time() + float(max(0.1, timeout_sec))
    while time.time() < deadline:
        ret = proc.poll()
        if ret is not None:
            return ret
        time.sleep(0.2)

    try:
        os.killpg(proc_group, signal.SIGKILL)
    except ProcessLookupError:
        return proc.poll()

    try:
        return proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        return proc.poll()


def _run_torchrun_with_signal_guard(cmd: list[str], env: dict[str, str]) -> int:
    """Run torchrun command with signal-safe cleanup for spawned child group.

    Args:
        cmd: Launch command list.
        env: Child environment variables.

    Returns:
        Child process return code.
    """
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    interrupted = {"code": 130}

    def _handle_interrupt(signum, _frame) -> None:
        interrupted["code"] = 128 + int(signum)
        print(
            f"[WARN] Received signal {signum}, terminating DDP child process group...",
            file=sys.stderr,
        )
        _terminate_process_group(proc)
        raise KeyboardInterrupt()

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        _terminate_process_group(proc)
        return int(interrupted["code"])
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def main() -> None:
    """CLI main function for MAE pretraining launch."""
    ensure_cuda_runtime_libs()
    project_root = _inject_src_path()

    from engine.trainer import run_cli
    from utils.config import load_yaml_config

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "pretrain_base.yaml"),
        type=str,
    )
    pre_parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Optional GPU override used for DDP auto-spawn sizing and trainer runtime.",
    )
    pre_parser.add_argument(
        "--master_port",
        type=int,
        default=None,
        help="Optional torchrun master port override for DDP rendezvous.",
    )
    pre_parser.add_argument("--no_auto_spawn", action="store_true")
    pre_args, remaining = pre_parser.parse_known_args()

    config = load_yaml_config(pre_args.config)

    # Apply early GPU override so DDP process-count follows CLI user intent.
    if pre_args.gpu is not None:
        config["gpu"] = str(pre_args.gpu)

    gpu_from_config = str(config.get("gpu", "0"))
    gpu_list = _split_gpu_list(gpu_from_config)
    cuda_available = _is_cuda_available()

    if len(gpu_list) > 1 and "LOCAL_RANK" not in os.environ and not pre_args.no_auto_spawn and cuda_available:
        visible_gpu_csv = _normalize_gpu_csv(gpu_list)
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(len(gpu_list)),
        ]
        if pre_args.master_port is not None:
            cmd.extend(["--master_port", str(pre_args.master_port)])
        cmd.extend(
            [
            str(Path(__file__).resolve()),
            "--config",
            str(pre_args.config),
            "--no_auto_spawn",
            ]
        )
        cmd.extend(remaining)
        env = dict(os.environ)
        # Ensure torchrun local ranks map strictly to requested GPU subset.
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu_csv
        return_code = _run_torchrun_with_signal_guard(cmd=cmd, env=env)
        if return_code == 0:
            return
        if return_code >= 128:
            raise SystemExit(return_code)
        print(
            "[WARN] Auto DDP spawn failed; fallback to single-process launch. "
            f"Exit code: {return_code}",
            file=sys.stderr,
        )

    if len(gpu_list) > 1 and not pre_args.no_auto_spawn and not cuda_available:
        print(
            "[WARN] Multiple GPUs configured but CUDA is unavailable in current runtime; "
            "fallback to single-process launch.",
            file=sys.stderr,
        )

    config_cli_args = _build_cli_args_from_config(config)
    run_cli(config_cli_args + remaining)


if __name__ == "__main__":
    main()
