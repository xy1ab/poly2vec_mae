import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
from utils.config.loader import load_json_config, load_yaml_config
from utils.fourier.engine import build_poly_fourier_converter_from_config
from utils.io.precision import normalize_precision, precision_to_torch_dtype

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(31, 31), patch_size=2, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# ==============================================================================
# 模块化拆分: 纯净编码器 (可独立供给下游任务使用)
# ==============================================================================
class PolyEncoder(nn.Module):
    def __init__(self, img_size=(31, 31), patch_size=2, in_chans=3, embed_dim=256, depth=12, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, 4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, ids_keep=None):
        # 1. 图像转 Token
        x = self.patch_embed(x)
        
        # 2. 加上位置编码 (此时不加 cls token)
        x = x + self.pos_embed[:, 1:, :]
        
        # 3. 掩码操作 (如果是下游任务，ids_keep 为 None，全量进入)
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            
        # 4. 拼接 Cls Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 5. 通过 Transformer Blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x

    def forward(self, x):
        """下游任务的极简标准接口，无需传递任何 mask 相关的参数"""
        return self.forward_features(x, ids_keep=None)

# ==============================================================================
# 模块化拆分: MAE 外壳 (仅用于自监督训练)
# ==============================================================================
class MaskedAutoencoderViTPoly(nn.Module):
    def __init__(self, img_size=(31, 31), patch_size=2, in_chans=3,
                 embed_dim=256, depth=12, num_heads=8,
                 decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4):
        super().__init__()
        
        # 1. 注入纯净的编码器
        self.encoder = PolyEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
        
        # 2. 独立的解码器组件
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.encoder.patch_embed.num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, 4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.encoder.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking_ids(self, N, L, device, mask_ratio):
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        N = x.shape[0]
        L = self.encoder.patch_embed.num_patches
        
        # 生成掩码索引
        ids_keep, mask, ids_restore = self.random_masking_ids(N, L, x.device, mask_ratio)
        
        # 调用分离出去的纯净 Encoder
        x = self.encoder.forward_features(x, ids_keep=ids_keep)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        
        loss = torch.tensor(0.0, device=imgs.device)
        return loss, pred, mask, pred, mask


def _load_model_config(config_path):
    path = str(config_path).lower()
    if path.endswith(".yaml") or path.endswith(".yml"):
        config = load_yaml_config(config_path)
    else:
        config = load_json_config(config_path)

    config = dict(config or {})
    img_size = config.get("img_size", None)
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        config["img_size"] = (int(img_size[0]), int(img_size[1]))
        return config

    # 兼容导出包 config.yaml（该文件通常不包含 img_size）
    # 通过 Fourier 参数自动推导训练时的有效频域网格大小。
    try:
        converter = build_poly_fourier_converter_from_config(config, device="cpu")
        config["img_size"] = (int(converter.U.shape[0]), int(converter.U.shape[1]))
    except Exception:
        config["img_size"] = tuple(config.get("img_size", (31, 31)))
    return config


def _resolve_precision_for_device(device, precision):
    resolved = normalize_precision(precision)
    dev = torch.device(device)
    if dev.type != "cuda" and resolved in ("fp16", "bf16"):
        return "fp32"
    if dev.type == "cuda" and resolved == "bf16" and not torch.cuda.is_bf16_supported():
        return "fp16"
    return resolved


def _move_model_to_precision(model, device, precision):
    resolved = _resolve_precision_for_device(device, precision)
    target_device = torch.device(device)
    if resolved == "fp32":
        return model.to(target_device), resolved
    target_dtype = precision_to_torch_dtype(resolved)
    return model.to(device=target_device, dtype=target_dtype), resolved


def _cast_state_dict_to_precision(state_dict, precision):
    target_dtype = precision_to_torch_dtype(precision)
    converted = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=target_dtype)
            converted[key] = tensor
        else:
            converted[key] = value
    return converted


def build_mae_model_from_config(config, device='cpu', precision='fp32'):
    model = MaskedAutoencoderViTPoly(
        img_size=tuple(config.get('img_size', (31, 31))),
        patch_size=config.get('patch_size', 2),
        in_chans=config.get('in_chans', 3),
        embed_dim=config.get('embed_dim', 256),
        depth=config.get('depth', 12),
        num_heads=config.get('num_heads', 8),
        decoder_embed_dim=config.get('dec_embed_dim', 128),
        decoder_depth=config.get('dec_depth', 4),
        decoder_num_heads=config.get('dec_num_heads', 4),
    )
    model, _ = _move_model_to_precision(model, device=device, precision=precision)
    return model

def load_mae_model(weight_path, config_path, device='cpu', precision='fp32'):
    """
    加载完整 MAE（Encoder + Decoder）并置为 eval，用于重建等下游任务。
    """
    config = _load_model_config(config_path)
    model = build_mae_model_from_config(config, device='cpu', precision='fp32')
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model, runtime_precision = _move_model_to_precision(model, device=device, precision=precision)
    model.eval()
    config = dict(config)
    config["runtime_precision"] = runtime_precision
    return model, config

# ==============================================================================
# 下游任务标准交付 API (结合 config.json 实现无缝载入)
# ==============================================================================
def load_pretrained_encoder(weight_path, config_path, device='cpu', precision='fp32'):
    """
    下游任务的开箱即用接口：自动通过 config.json (如 poly_mae_config.json) 读取结构参数，
    自动忽略无关的 Decoder 参数，加载剥离后的 Encoder 权重，并执行安全冻结。
    """
    config = _load_model_config(config_path)
        
    img_size = tuple(config.get("img_size", (31, 31)))
    patch_size = config.get("patch_size", 2)
    in_chans = config.get("in_chans", 3)
    embed_dim = config.get("embed_dim", 256)
    depth = config.get("depth", 12)
    num_heads = config.get("num_heads", 8)
    
    encoder = PolyEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
    state_dict = torch.load(weight_path, map_location='cpu')

    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and any(k.startswith("encoder.") for k in state_dict.keys()):
        encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
        if encoder_state:
            state_dict = encoder_state
    
    # 严格加载纯净的 encoder 权重
    try:
        encoder.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        if "size mismatch for pos_embed" in str(exc):
            ckpt_pos_embed = state_dict.get("pos_embed", None) if isinstance(state_dict, dict) else None
            ckpt_tokens = int(ckpt_pos_embed.shape[1]) if torch.is_tensor(ckpt_pos_embed) else "unknown"
            raise RuntimeError(
                "加载 encoder 权重失败：pos_embed 尺寸不匹配。\n"
                f"当前配置推导 img_size={img_size}, patch_size={patch_size}, 期望 token 数={encoder.pos_embed.shape[1]}；"
                f"checkpoint token 数={ckpt_tokens}。\n"
                "请确认 model_dir 下的配置文件和权重来自同一次训练导出。"
            ) from exc
        raise
    
    # 执行安全冻结
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    encoder, _ = _move_model_to_precision(encoder, device=device, precision=precision)
    return encoder

def export_encoder_from_mae_checkpoint(mae_ckpt_path, config_path, output_path, device='cpu', precision='fp32'):
    """
    从完整 MAE checkpoint 中提取 encoder 权重并导出。
    """
    config = _load_model_config(config_path)
    model = build_mae_model_from_config(config, device='cpu', precision='fp32')
    state_dict = torch.load(mae_ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    export_precision = _resolve_precision_for_device(device, precision)
    encoder_state = _cast_state_dict_to_precision(model.encoder.state_dict(), export_precision)
    torch.save(encoder_state, output_path)
    return output_path
