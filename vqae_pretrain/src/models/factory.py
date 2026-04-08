"""Model factory and checkpoint loading APIs for VQAE pretraining."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch

from ..utils.config import load_config_any
from ..utils.precision import normalize_precision, precision_to_torch_dtype, resolve_precision_for_device
from .encoder import PolyEncoder
from .vqae import PolyVqAutoencoder


def infer_img_size_from_config(config: dict[str, Any]) -> tuple[int, int]:
    """Infer runtime image size from Fourier-grid config when absent."""
    if "img_size" in config and config["img_size"] is not None:
        img_size = config["img_size"]
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            return int(img_size[0]), int(img_size[1])

    pos_freqs = int(config.get("pos_freqs", 31))
    stem_strides = _parse_int_sequence(config.get("stem_strides"), default=(2, 2, 2))
    align_stride = max(1, math.prod(int(value) for value in stem_strides))

    len_h = 2 * pos_freqs + 1
    len_w = pos_freqs + 1

    pad_h = (align_stride - (len_h % align_stride)) % align_stride
    pad_w = (align_stride - (len_w % align_stride)) % align_stride
    return len_h + pad_h, len_w + pad_w


def _parse_int_sequence(value: Any, default: Sequence[int]) -> tuple[int, ...]:
    """Parse one int sequence from config value or fall back to default."""
    if value is None:
        return tuple(int(v) for v in default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            return tuple(int(v) for v in default)
        return tuple(int(item) for item in items)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return (int(value),)


def _move_model_to_runtime_precision(
    model: torch.nn.Module,
    device: torch.device,
    precision: str,
) -> tuple[torch.nn.Module, str]:
    """Move model to runtime device and supported precision."""
    resolved = resolve_precision_for_device(device, precision)
    if resolved == "fp32":
        return model.to(device), resolved
    return model.to(device=device, dtype=precision_to_torch_dtype(resolved)), resolved


def build_vqae_model_from_config(
    config: dict[str, Any],
    device: str | torch.device = "cpu",
    precision: str = "fp32",
) -> PolyVqAutoencoder:
    """Construct one VQAE model from config dictionary."""
    img_size = infer_img_size_from_config(config)
    stem_channels = _parse_int_sequence(config.get("stem_channels"), default=(64, 128, 256))
    stem_strides = _parse_int_sequence(config.get("stem_strides"), default=(2, 2, 2))
    decoder_stage_channels = _parse_int_sequence(config.get("decoder_stage_channels"), default=(256, 192, 128))
    decoder_attention_heads = _parse_int_sequence(config.get("decoder_attention_heads"), default=(8, 4, 4))
    decoder_attention_depths = _parse_int_sequence(config.get("decoder_attention_depths"), default=(1, 1, 0))
    decoder_conv_depths = _parse_int_sequence(config.get("decoder_conv_depths"), default=(2, 2, 2))

    model = PolyVqAutoencoder(
        img_size=img_size,
        in_chans=int(config.get("in_chans", 3)),
        stem_channels=stem_channels,
        stem_strides=stem_strides,
        embed_dim=int(config.get("embed_dim", 256)),
        depth=int(config.get("depth", 8)),
        num_heads=int(config.get("num_heads", 8)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        drop_rate=float(config.get("drop_rate", 0.0)),
        decoder_stage_channels=decoder_stage_channels,
        decoder_attention_type=str(config.get("decoder_attention_type", "window")),
        decoder_attention_heads=decoder_attention_heads,
        decoder_attention_depths=decoder_attention_depths,
        decoder_conv_depths=decoder_conv_depths,
        decoder_window_size=int(config.get("decoder_window_size", 8)),
        decoder_upsample_mode=str(config.get("decoder_upsample_mode", "bilinear")),
        decoder_mlp_ratio=float(config.get("decoder_mlp_ratio", 4.0)),
        decoder_drop_rate=float(config.get("decoder_drop_rate", 0.0)),
        codebook_size=int(config.get("codebook_size", 8192)),
        code_dim=int(config.get("code_dim", 128)),
        vq_decay=float(config.get("vq_decay", 0.99)),
        vq_eps=float(config.get("vq_eps", 1.0e-5)),
        vq_dead_code_threshold=float(config.get("vq_dead_code_threshold", 1.0)),
    )
    model, _ = _move_model_to_runtime_precision(model, torch.device(device), normalize_precision(precision))
    return model


def load_vqae_model(
    weight_path: str | Path,
    config_path: str | Path,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
) -> tuple[PolyVqAutoencoder, dict[str, Any]]:
    """Load one full VQAE model from checkpoint and config."""
    config = load_config_any(config_path)
    model = build_vqae_model_from_config(config, device="cpu", precision="fp32")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    if "model_state" in state_dict and isinstance(state_dict["model_state"], dict):
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict, strict=True)

    model, runtime_precision = _move_model_to_runtime_precision(model, torch.device(device), precision)
    model.eval()

    runtime_config = dict(config)
    runtime_config["img_size"] = infer_img_size_from_config(runtime_config)
    runtime_config["runtime_precision"] = runtime_precision
    runtime_config["latent_stride"] = int(model.encoder.latent_stride)
    runtime_config["latent_grid_size"] = tuple(int(v) for v in model.encoder.latent_grid_size)
    runtime_config["num_latent_tokens"] = int(model.encoder.num_latent_tokens)
    return model, runtime_config


def _extract_encoder_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract encoder-only state dict from a full VQAE checkpoint."""
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if "model_state" in state_dict and isinstance(state_dict["model_state"], dict):
        state_dict = state_dict["model_state"]

    if any(str(k).startswith("encoder.") for k in state_dict.keys()):
        encoder_state = {k[len("encoder.") :]: v for k, v in state_dict.items() if str(k).startswith("encoder.")}
        if encoder_state:
            return encoder_state
    return state_dict


def load_pretrained_encoder(
    weight_path: str | Path,
    config_path: str | Path,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
) -> PolyEncoder:
    """Load frozen encoder weights for downstream tasks."""
    config = load_config_any(config_path)
    img_size = infer_img_size_from_config(config)

    encoder = PolyEncoder(
        img_size=img_size,
        in_chans=int(config.get("in_chans", 3)),
        stem_channels=_parse_int_sequence(config.get("stem_channels"), default=(64, 128, 256)),
        stem_strides=_parse_int_sequence(config.get("stem_strides"), default=(2, 2, 2)),
        embed_dim=int(config.get("embed_dim", 256)),
        depth=int(config.get("depth", 8)),
        num_heads=int(config.get("num_heads", 8)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        drop_rate=float(config.get("drop_rate", 0.0)),
    )

    raw_state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = _extract_encoder_state_dict(raw_state_dict)
    encoder.load_state_dict(state_dict, strict=True)

    for param in encoder.parameters():
        param.requires_grad = False

    encoder, _ = _move_model_to_runtime_precision(encoder, torch.device(device), precision)
    encoder.eval()
    return encoder


def export_vqae_components(
    vqae_ckpt_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    precision: str = "fp32",
) -> Path:
    """Export full-model and component checkpoints from one VQAE checkpoint."""
    from ..utils.checkpoint import save_checkpoint
    from ..utils.config import dump_yaml_config
    from ..utils.filesystem import ensure_dir

    model, runtime_config = load_vqae_model(
        weight_path=vqae_ckpt_path,
        config_path=config_path,
        device="cpu",
        precision="fp32",
    )
    out_dir = ensure_dir(Path(output_dir))
    dump_yaml_config(runtime_config, out_dir / "config.yaml")
    save_checkpoint(out_dir / "vqae_best.pth", model.state_dict(), precision=precision)
    save_checkpoint(out_dir / "encoder.pth", model.encoder.state_dict(), precision=precision)
    save_checkpoint(
        out_dir / "decoder.pth",
        {
            "post_vq_proj": model.post_vq_proj.state_dict(),
            "decoder": model.decoder.state_dict(),
        },
        precision=precision,
    )
    torch.save(model.quantizer.state_dict(), out_dir / "quantizer.pth")
    return out_dir


def load_decoder_from_components(
    decoder_path: str | Path,
    quantizer_path: str | Path,
    config_path: str | Path,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
) -> tuple[PolyVqAutoencoder, dict[str, Any]]:
    """Load one VQAE model with decoder+quantizer components for code decoding."""
    config = load_config_any(config_path)
    model = build_vqae_model_from_config(config, device="cpu", precision="fp32")

    decoder_state = torch.load(decoder_path, map_location="cpu", weights_only=False)
    quantizer_state = torch.load(quantizer_path, map_location="cpu", weights_only=False)
    model.post_vq_proj.load_state_dict(decoder_state["post_vq_proj"], strict=True)
    model.decoder.load_state_dict(decoder_state["decoder"], strict=True)
    model.quantizer.load_state_dict(quantizer_state, strict=True)

    model, runtime_precision = _move_model_to_runtime_precision(model, torch.device(device), precision)
    model.eval()

    runtime_config = dict(config)
    runtime_config["img_size"] = infer_img_size_from_config(runtime_config)
    runtime_config["runtime_precision"] = runtime_precision
    runtime_config["latent_stride"] = int(model.encoder.latent_stride)
    runtime_config["latent_grid_size"] = tuple(int(v) for v in model.encoder.latent_grid_size)
    runtime_config["num_latent_tokens"] = int(model.encoder.num_latent_tokens)
    return model, runtime_config
