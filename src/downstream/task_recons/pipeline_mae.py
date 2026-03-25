import torch
import numpy as np

# 导入底层模型与引擎
from mae_core.model import load_mae_model
from utils.fourier.engine import build_poly_fourier_converter_from_config
from utils.io.precision import autocast_context, normalize_precision

class MaeReconstructionPipeline:
    """
    MAE 掩码重建流水线 (用于对接下游 ResNet 神经逆变换任务)
    输入：原始矢量几何坐标
    输出：去 Padding 后的 1/2 象限频域 实部(Real) 和 虚部(Imag) 张量
    """
    def __init__(self, weight_path, config_path, device='cuda' if torch.cuda.is_available() else 'cpu', precision='bf16'):
        self.device = torch.device(device)
        self.precision = normalize_precision(precision)
        print(f"[Reconstructor] 初始化 MAE 重建流水线，使用设备: {self.device}")
        
        # 1. 读取配置并加载 MAE 完整模型
        self.model, self.config = load_mae_model(
            weight_path,
            config_path,
            device=self.device,
            precision=self.precision,
        )
        self.precision = self.config.get("runtime_precision", self.precision)
            
        self.geom_type = self.config.get("geom_type", "polygon").lower() 
            
        # 2. 初始化底层物理引擎
        self.fourier_engine = build_poly_fourier_converter_from_config(self.config, device=self.device)
        
        # 获取有效频率区域尺寸 (去除自动 Padding 的部分)
        self.valid_h = self.fourier_engine.U.shape[0] - self.fourier_engine.pad_h
        self.valid_w = self.fourier_engine.U.shape[1] - self.fourier_engine.pad_w
        
        print(f"[Reconstructor] MAE 完整模型权重加载成功！(runtime precision: {self.precision})")

        # 注册预处理策略
        self._preprocess_strategies = {
            "polygon": self._preprocess_polygon,
        }
        self._engine_strategies = {
            "polygon": self.fourier_engine.cft_polygon_batch,
        }

    def _triangulate_polygon(self, poly_norm):
        return self.fourier_engine.triangulate_polygon(poly_norm)

    def _preprocess_polygon(self, poly):
        min_x, min_y = np.min(poly, axis=0)
        max_x, max_y = np.max(poly, axis=0)
        cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        L = max(max_x - min_x, max_y - min_y)
        if L == 0: L = 1e-6 
        poly_norm = (poly - np.array([cx, cy])) / (L / 2.0)
        processed_data = self._triangulate_polygon(poly_norm)
        return processed_data

    @torch.no_grad()
    def reconstruct_real_imag(self, geometries: list, mask_ratio=0.75):
        """
        核心 API：输入几何列表，输出掩码重建后的 1/2 象限纯净实部和虚部
        :param geometries: 几何实体坐标列表
        :param mask_ratio: MAE 掩码比例
        :return: (real_part, imag_part) 形状均为 [B, valid_h, valid_w]
        """
        B = len(geometries)
        parsed_data_list = []
        
        preprocess_fn = self._preprocess_strategies[self.geom_type]
        engine_fn = self._engine_strategies[self.geom_type]
        
        for geom in geometries:
            geom_np = np.array(geom, dtype=np.float32)
            parsed_data = preprocess_fn(geom_np)
            parsed_data_list.append(parsed_data)

        max_len = max(len(d) for d in parsed_data_list)
        data_shape = parsed_data_list[0].shape[1:] 
        
        batch_parsed_data = torch.zeros((B, max_len, *data_shape), dtype=torch.float32, device=self.device)
        lengths = torch.zeros(B, dtype=torch.long, device=self.device)
        
        for i, d in enumerate(parsed_data_list):
            l = len(d)
            batch_parsed_data[i, :l] = torch.tensor(d, dtype=torch.float32, device=self.device)
            lengths[i] = l

        # 1. 物理引擎生成 Ground Truth (带 Padding 的三通道图)
        mag, phase = engine_fn(batch_parsed_data, lengths)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        imgs = torch.cat([mag, cos_phase, sin_phase], dim=1) # [B, 3, H, W]

        # 2. MAE 掩码与重建
        with autocast_context(self.device, self.precision):
            _, pred, mask_seq, _, _ = self.model(imgs, mask_ratio=mask_ratio)
        pred = pred.float()
        mask_seq = mask_seq.float()
        
        # 3. 反序列化 (Unpatchify) 预测结果和掩码图
        p = self.config.get("patch_size", 2)
        H, W = imgs.shape[2], imgs.shape[3]
        h_p, w_p = H // p, W // p
        
        # 将 [B, L, p*p*3] 恢复为 [B, 3, H, W]
        pred_img = pred.reshape(B, h_p, w_p, 3, p, p)
        pred_img = torch.einsum('nhwcpq->nchpwq', pred_img).reshape(B, 3, H, W)
        
        # 将 [B, L] 恢复为二维网格 [B, 1, H, W] (修复了致命的 permute 乱序 BUG)
        mask_map = mask_seq.reshape(B, h_p, w_p, 1, 1).expand(-1, -1, -1, p, p)
        mask_map = mask_map.permute(0, 1, 3, 2, 4).reshape(B, 1, H, W)
        
        # 4. 融合：真实未掩码部分 + 预测的掩码部分
        recon_imgs = imgs * (1 - mask_map) + pred_img * mask_map
        
        # 5. 裁剪去除自动 Padding 区域，提取真正有效的 1/2 象限
        recon_valid = recon_imgs[:, :, :self.valid_h, :self.valid_w]
        
        mag_valid = recon_valid[:, 0, :, :]
        cos_valid = recon_valid[:, 1, :, :]
        sin_valid = recon_valid[:, 2, :, :]
        
        # 6. 计算物理实部与虚部
        # 由于预测的 cos 和 sin 可能不严格满足平方和为 1，使用 atan2 求解稳健相位
        phase_valid = torch.atan2(sin_valid, cos_valid)
        raw_mag_valid = torch.expm1(mag_valid) # 还原对数幅值
        
        real_part = raw_mag_valid * torch.cos(phase_valid)
        imag_part = raw_mag_valid * torch.sin(phase_valid)
        
        return real_part, imag_part
