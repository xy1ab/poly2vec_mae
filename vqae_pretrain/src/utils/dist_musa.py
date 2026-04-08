"""Distributed training helpers.

This module hides low-level torch.distributed setup and teardown details from
the trainer, making the training loop easier to read and test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import torch_musa
import torch
import torch.distributed as dist


@dataclass
class DistContext:
    """Runtime distributed context.

    Attributes:
        enabled: Whether distributed training is enabled.
        rank: Global process rank.
        local_rank: Local process rank on the current node.
        world_size: Total number of processes.
        device: Device assigned to current process.
    """

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

def init_distributed_musa() -> DistContext:
    """Initialize distributed training when LOCAL_RANK is provided.

    Returns:
        DistContext describing runtime process/device state.
    """
    if "LOCAL_RANK" not in os.environ:
        device = torch.device("musa:0" if torch.musa.is_available() else "cpu")
        return DistContext(enabled=False, rank=0, local_rank=0, world_size=1, device=device)

    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "mccl" if torch.musa.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    if torch.musa.is_available():
        torch.musa.set_device(local_rank)
        device = torch.device(f"musa:{local_rank}")
    else:
        device = torch.device("cpu")

    return DistContext(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        device=device,
    )

def is_main_process(ctx: DistContext) -> bool:
    """Check whether current process is rank-0 process.

    Args:
        ctx: Distributed runtime context.

    Returns:
        True only on main process.
    """
    return ctx.rank == 0


def cleanup_distributed(ctx: DistContext) -> None:
    """Tear down process group when distributed is enabled.

    Args:
        ctx: Distributed runtime context.
    """
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(ctx: DistContext) -> None:
    """Synchronize distributed ranks with an explicit runtime device when needed.

    Args:
        ctx: Distributed runtime context.
    """
    if not ctx.enabled or not dist.is_initialized():
        return

    if ctx.device.type == "musa":
        dist.barrier(device_ids=[int(ctx.local_rank)])
        return

    dist.barrier()


def all_reduce_mean(value: torch.Tensor, ctx: DistContext) -> torch.Tensor:
    """All-reduce a scalar tensor and return world-size mean.

    Args:
        value: Scalar tensor on current device.
        ctx: Distributed runtime context.

    Returns:
        Mean tensor across processes.
    """
    if not ctx.enabled:
        return value

    out = value.clone()
    dist.all_reduce(out)
    out /= float(ctx.world_size)
    return out
