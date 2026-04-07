"""Pretraining launcher script for MUSA runtime.

This script reads YAML config defaults, applies CLI overrides, and launches the
MUSA trainer engine. It also supports optional auto-spawn for multi-device DDP
runs.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


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
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli_args.append(arg_name)
        else:
            cli_args.extend([arg_name, str(value)])
    return cli_args


def _resolve_project_relative_config_paths(config_dict: dict, project_root: Path) -> dict:
    """Resolve path-like config entries relative to project root.

    Args:
        config_dict: Parsed YAML config dictionary.
        project_root: `ae_pretrain` project root.

    Returns:
        Config copy with normalized path-like values.
    """
    resolved = dict(config_dict)
    for key in ("data_dir", "data_path", "save_dir", "resume_dir"):
        value = resolved.get(key)
        if value is None:
            continue

        path_value = Path(str(value)).expanduser()
        if not path_value.is_absolute():
            resolved[key] = str((project_root / path_value).resolve())
    return resolved


def _load_resume_config(resume_dir: str | Path, project_root: Path) -> tuple[dict, Path]:
    """Load saved training config from one previous run directory.

    Args:
        resume_dir: Run directory path `<save_dir>/<run_timestamp>`.
        project_root: `ae_pretrain` project root.

    Returns:
        Tuple `(config_dict, resolved_resume_dir)`.
    """
    if __package__ in {None, ""}:
        import importlib

        load_yaml_config = importlib.import_module("ae_pretrain.src.utils.config").load_yaml_config
    else:
        from ..src.utils.config import load_yaml_config

    resolved_resume_dir = Path(resume_dir).expanduser().resolve()
    config_path = resolved_resume_dir / "ckpt" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Resume config not found: {config_path}")

    config = _resolve_project_relative_config_paths(
        load_yaml_config(config_path),
        project_root=project_root,
    )
    config["resume_dir"] = str(resolved_resume_dir)
    return config, resolved_resume_dir


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


def _is_musa_available() -> bool:
    """Check whether MUSA runtime is available for current Python process.

    Returns:
        True when torch can be imported and MUSA is available, else False.
    """
    try:
        import torch_musa  # noqa: F401
        import torch
        torch.backends.mudnn.allow_tf32 = True
        return bool(torch.musa.is_available())
    except Exception:
        return False


def _get_visible_musa_device_count() -> int:
    """Query the number of MUSA devices visible to current process.

    Returns:
        Visible device count, or `0` when unavailable.
    """
    try:
        import torch_musa  # noqa: F401
        import torch

        if not torch.musa.is_available():
            return 0
        device_count = getattr(torch.musa, "device_count", None)
        if device_count is None:
            return 0
        return int(device_count())
    except Exception:
        return 0


def _normalize_requested_gpu_list(gpu_list: list[str], visible_device_count: int) -> list[str]:
    """Trim requested device ids to a subset that fits current visibility.

    Args:
        gpu_list: Requested MUSA id list from config/CLI.
        visible_device_count: Device count visible to current process.

    Returns:
        Effective device list safe for current runtime.
    """
    if not gpu_list:
        return ["0"]

    if visible_device_count <= 0:
        return gpu_list

    if len(gpu_list) <= visible_device_count:
        return gpu_list

    return gpu_list[:visible_device_count]


def _normalize_runtime_config(config: dict) -> dict:
    """Normalize legacy config keys into the current training interface."""
    normalized = dict(config)
    if "eval_every" not in normalized:
        legacy_eval = normalized.get("save_every", normalized.get("viz_every", 20))
        normalized["eval_every"] = int(legacy_eval)

    normalized.pop("save_every", None)
    normalized.pop("viz_every", None)
    normalized.pop("export_dir", None)
    return normalized


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
    """CLI main function for AE pretraining launch."""
    project_root = _inject_repo_root()

    if __package__ in {None, ""}:
        import importlib

        run_cli = importlib.import_module("ae_pretrain.src.engine.trainer_musa").run_cli
        load_yaml_config = importlib.import_module(
            "ae_pretrain.src.utils.config"
        ).load_yaml_config
    else:
        from ..src.engine.trainer_musa import run_cli
        from ..src.utils.config import load_yaml_config

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
    pre_parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="Optional evaluation-epoch interval override.",
    )
    pre_parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Resume from an existing run directory `<save_dir>/<run_timestamp>`.",
    )
    pre_parser.add_argument("--no_auto_spawn", action="store_true")
    pre_args, remaining = pre_parser.parse_known_args()

    if pre_args.resume_dir is not None:
        config, resolved_resume_dir = _load_resume_config(pre_args.resume_dir, project_root=project_root)
    else:
        config = _resolve_project_relative_config_paths(
            load_yaml_config(pre_args.config),
            project_root=project_root,
        )
        resolved_resume_dir = None
    config = _normalize_runtime_config(config)

    if pre_args.gpu is not None:
        config["gpu"] = str(pre_args.gpu)
    if pre_args.eval_every is not None:
        config["eval_every"] = int(pre_args.eval_every)
    if resolved_resume_dir is not None:
        config["resume_dir"] = str(resolved_resume_dir)

    gpu_from_config = str(config.get("gpu", "0"))
    requested_gpu_list = _split_gpu_list(gpu_from_config)
    musa_available = _is_musa_available()
    visible_device_count = _get_visible_musa_device_count() if musa_available else 0
    gpu_list = _normalize_requested_gpu_list(requested_gpu_list, visible_device_count)
    if gpu_list != requested_gpu_list:
        print(
            "[WARN] Requested MUSA devices exceed visible runtime devices; "
            f"using {gpu_list} instead of {requested_gpu_list}.",
            file=sys.stderr,
        )
        config["gpu"] = _normalize_gpu_csv(gpu_list)

    if len(gpu_list) > 1 and "LOCAL_RANK" not in os.environ and not pre_args.no_auto_spawn and musa_available:
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
        cmd.append(str(Path(__file__).resolve()))
        if resolved_resume_dir is not None:
            cmd.extend(["--resume_dir", str(resolved_resume_dir)])
        else:
            cmd.extend(["--config", str(pre_args.config)])
        cmd.append("--no_auto_spawn")
        if pre_args.eval_every is not None:
            cmd.extend(["--eval_every", str(pre_args.eval_every)])
        cmd.extend(remaining)
        env = dict(os.environ)
        env["MUSA_VISIBLE_DEVICES"] = visible_gpu_csv
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

    if len(gpu_list) > 1 and not pre_args.no_auto_spawn and not musa_available:
        print(
            "[WARN] Multiple MUSA devices configured but MUSA is unavailable in current runtime; "
            "fallback to single-process launch.",
            file=sys.stderr,
        )

    config_cli_args = _build_cli_args_from_config(config)
    run_cli(config_cli_args + remaining)


if __name__ == "__main__":
    main()
