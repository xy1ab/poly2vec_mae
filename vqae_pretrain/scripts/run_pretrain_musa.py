"""Pretraining launcher script for polygon VQAE.

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


def _inject_repo_root() -> Path:
    """Inject repository root into `sys.path` for direct script execution.

    Returns:
        `vqae_pretrain` project root path.
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
        project_root: `vqae_pretrain` project root.

    Returns:
        Config copy with normalized path-like values.
    """
    resolved = dict(config_dict)
    for key in ("data_dir", "data_path", "save_dir"):
        value = resolved.get(key)
        if value is None:
            continue

        path_value = Path(str(value)).expanduser()
        if not path_value.is_absolute():
            resolved[key] = str((project_root / path_value).resolve())
    return resolved


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
        import torch_musa
        import torch
        torch.backends.mudnn.allow_tf32 = True
        return bool(torch.musa.is_available())
    except Exception:
        return False


def _get_visible_musa_device_count() -> int:
    """Query the number of MUSA devices visible to current process.

    Returns:
        Visible MUSA device count, or `0` when unavailable.
    """
    try:
        import torch_musa
        import torch

        if not torch.musa.is_available():
            return 0
        return int(torch.musa.device_count())
    except Exception:
        return 0


def _normalize_requested_gpu_list(gpu_list: list[str], visible_device_count: int) -> list[str]:
    """Trim requested GPU ids to a subset that fits current machine visibility.

    This keeps current behavior for valid multi-GPU requests while protecting
    single-GPU hosts from accidentally spawning ranks for non-existent devices.

    Args:
        gpu_list: Requested GPU id list from config/CLI.
        visible_device_count: MUSA device count visible to current process.

    Returns:
        Effective GPU list safe for current runtime.
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
    normalized.pop("latent_stride", None)
    normalized.pop("latent_grid_size", None)
    normalized.pop("num_latent_tokens", None)
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


def _build_torchrun_cmd(
    *,
    script_path: Path,
    nproc_per_node: int,
    master_port: int | None,
) -> list[str]:
    """Build one single-node torchrun command for local DDP auto-spawn.

    When `master_port` is omitted, the launcher uses `--standalone` so torchrun
    creates its local TCP store on a free port instead of the default static
    rendezvous port `29500`.
    """
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc_per_node),
    ]
    if master_port is not None:
        cmd.extend(["--master_port", str(master_port)])
    else:
        cmd.append("--standalone")
    cmd.append(str(script_path.resolve()))
    return cmd


def main() -> None:
    """CLI main function for VQAE pretraining launch."""

    project_root = _inject_repo_root()

    if __package__ in {None, ""}:
        import importlib

        run_cli = importlib.import_module("vqae_pretrain.src.engine.trainer_musa").run_cli
        load_yaml_config = importlib.import_module(
            "vqae_pretrain.src.utils.config"
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
        "--run_name",
        type=str,
        required=True,
        help="Stable run directory name under `save_dir`; existing checkpoints auto-resume.",
    )
    pre_parser.add_argument(
        "--vq_init_mode",
        type=str,
        default=None,
        choices=("rank0_local", "all_ranks_gather"),
        help="Optional VQ codebook initialization mode override for diagnosis.",
    )
    pre_parser.add_argument(
        "--vq_init_debug",
        action="store_true",
        help="Enable VQ initialization debug logging for diagnosis.",
    )
    pre_parser.add_argument("--no_auto_spawn", action="store_true")
    pre_args, remaining = pre_parser.parse_known_args()

    config = _resolve_project_relative_config_paths(
        load_yaml_config(pre_args.config),
        project_root=project_root,
    )
    config = _normalize_runtime_config(config)
    config["run_name"] = str(pre_args.run_name)

    # Apply early GPU override so DDP process-count follows CLI user intent.
    if pre_args.gpu is not None:
        config["gpu"] = str(pre_args.gpu)
    if pre_args.eval_every is not None:
        config["eval_every"] = int(pre_args.eval_every)
    if pre_args.vq_init_mode is not None:
        config["vq_init_mode"] = str(pre_args.vq_init_mode)
    if pre_args.vq_init_debug:
        config["vq_init_debug"] = True

    gpu_from_config = str(config.get("gpu", "0"))
    requested_gpu_list = _split_gpu_list(gpu_from_config)
    musa_available = _is_musa_available()
    visible_device_count = _get_visible_musa_device_count() if musa_available else 0
    gpu_list = _normalize_requested_gpu_list(requested_gpu_list, visible_device_count)
    if gpu_list != requested_gpu_list:
        print(
            "[WARN] Requested GPUs exceed visible MUSA devices; "
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
        cmd.extend(["--config", str(pre_args.config)])
        cmd.append("--no_auto_spawn")
        cmd.extend(["--run_name", str(pre_args.run_name)])
        if pre_args.gpu is not None:
            cmd.extend(["--gpu", str(pre_args.gpu)])
        if pre_args.eval_every is not None:
            cmd.extend(["--eval_every", str(pre_args.eval_every)])
        if pre_args.vq_init_mode is not None:
            cmd.extend(["--vq_init_mode", str(pre_args.vq_init_mode)])
        if pre_args.vq_init_debug:
            cmd.append("--vq_init_debug")
        cmd.extend(remaining)
        env = dict(os.environ)
        # Ensure torchrun local ranks map strictly to requested GPU subset.
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
            "[WARN] Multiple GPUs configured but MUSA is unavailable in current runtime; "
            "fallback to single-process launch.",
            file=sys.stderr,
        )

    config_cli_args = _build_cli_args_from_config(config)
    run_cli(config_cli_args + remaining)


if __name__ == "__main__":
    main()
