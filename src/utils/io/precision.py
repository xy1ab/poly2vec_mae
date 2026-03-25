from contextlib import nullcontext
import torch


def normalize_precision(precision):
    p = str(precision).lower()
    if p in ("float32", "fp32"):
        return "fp32"
    if p in ("float16", "fp16"):
        return "fp16"
    if p in ("bfloat16", "bf16"):
        return "bf16"
    raise ValueError(f"Unsupported precision: {precision}")


def precision_to_torch_dtype(precision):
    p = normalize_precision(precision)
    if p == "fp32":
        return torch.float32
    if p == "fp16":
        return torch.float16
    return torch.bfloat16


def autocast_context(device, precision):
    p = normalize_precision(precision)
    if device.type == "cuda" and p in ("fp16", "bf16"):
        return torch.autocast(device_type="cuda", dtype=precision_to_torch_dtype(p), enabled=True)
    return nullcontext()
