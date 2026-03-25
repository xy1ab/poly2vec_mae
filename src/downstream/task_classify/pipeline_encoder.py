import os
import torch
import numpy as np

from mae_core.model import load_pretrained_encoder
from utils.config.loader import load_json_config
from utils.fourier.engine import build_poly_fourier_converter_from_config
from utils.io.precision import autocast_context, normalize_precision

class PolyFeaturePipeline:
    """
    通用矢量要素端到端特征提取流水线 (基于策略模式)。
    遵循 OCP 原则设计：对扩展线/点开放，对修改核心逻辑关闭。
    """
    def __init__(self, weight_path, config_path, device='cuda' if torch.cuda.is_available() else 'cpu', precision='bf16'):
        self.device = torch.device(device)
        self.precision = normalize_precision(precision)
        print(f"[Pipeline] 初始化特征流水线，使用设备: {self.device}")
        
        # 1. 读取 MAE 模型配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")
        self.config = load_json_config(config_path)
            
        # 2. 动态获取基础特征维度与身份
        self.embed_dim = self.config.get("embed_dim", 256)
        self.final_dim = self.embed_dim + 4
        # 默认兼容老版本，缺失时认为是 polygon
        self.geom_type = self.config.get("geom_type", "polygon").lower() 
            
        # 3. 初始化底层的物理引擎
        # (未来如果 Point 和 Line 有各自专属的 Converter，可以在策略字典里动态按需实例化)
        self.fourier_engine = build_poly_fourier_converter_from_config(self.config, device=self.device)
        
        # 4. 初始化并冻结 Encoder
        self.encoder = load_pretrained_encoder(
            weight_path,
            config_path,
            device=self.device,
            precision=self.precision,
        )
        print(f"[Pipeline] 预训练 Encoder ({self.embed_dim}维) 加载并冻结完成！")
        print(f"[Pipeline] 融合后最终特征向量维度为: {self.final_dim} 维")
        
        # =========================================================================
        # 核心 OCP 架构：注册策略字典 (Registry)
        # =========================================================================
        # 1. 几何预处理策略
        self._preprocess_strategies = {
            "polygon": self._preprocess_polygon,
            # "polyline": self._preprocess_polyline,  # 未来只需在此注册
            # "point": self._preprocess_point
        }
        
        # 2. 物理张量生成策略
        self._engine_strategies = {
            "polygon": self.fourier_engine.cft_polygon_batch,
            # "polyline": self.fourier_engine.cft_polyline_batch, # 未来扩展
        }
        
        if self.geom_type not in self._preprocess_strategies:
            raise NotImplementedError(
                f"[Pipeline Error] 安全拦截！您加载的模型是用 '{self.geom_type}' 训练的，"
                f"但流水线尚未注册该几何类型的预处理策略。请扩展对应的底层方法！"
            )

    # -------------------------------------------------------------------------
    # 策略 1: 针对 Polygon 的预处理实现
    # -------------------------------------------------------------------------
    def _triangulate_polygon(self, poly_norm):
        """统一复用 fourier engine 的 triangle 约束 Delaunay 剖分实现。"""
        return self.fourier_engine.triangulate_polygon(poly_norm)

    def _preprocess_polygon(self, poly):
        N = len(poly)
        min_x, min_y = np.min(poly, axis=0)
        max_x, max_y = np.max(poly, axis=0)
        cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        
        L = max(max_x - min_x, max_y - min_y)
        if L == 0: L = 1e-6 
        
        poly_norm = (poly - np.array([cx, cy])) / (L / 2.0)
        
        # 执行三角剖分 (多边形专属操作)
        processed_data = self._triangulate_polygon(poly_norm)
        
        return processed_data, cx, cy, L, N

    # -------------------------------------------------------------------------
    # 统一的主干特征提取循环 (封闭修改)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(self, geometries: list):
        """
        核心 API：输入几何坐标列表，输出融合特征张量
        :param geometries: list of np.ndarray, 每条是一段独立矢量
        :return: final_emb [B, embed_dim + 4], 包含基础特征 + [cx, cy, L, N]
        """
        B = len(geometries)
        parsed_data_list = []
        meta_list = []
        
        # 获取当前模型所匹配的专属策略
        preprocess_fn = self._preprocess_strategies[self.geom_type]
        engine_fn = self._engine_strategies[self.geom_type]
        
        # Step 1 & 2: 动态多态预处理 (可能是剖分，可能是线段切分)
        for geom in geometries:
            geom_np = np.array(geom, dtype=np.float32)
            parsed_data, cx, cy, L, N = preprocess_fn(geom_np)
            parsed_data_list.append(parsed_data)
            
            # 数值防冲垮缩放
            meta_list.append([cx, cy, L, N / 100.0]) 

        # Step 3: 根据返回的数据形态动态 Padding (例如三角形是 Nx3x2)
        max_len = max(len(d) for d in parsed_data_list)
        data_shape = parsed_data_list[0].shape[1:] # 抛去长度 N 剩下的维度
        
        batch_parsed_data = torch.zeros((B, max_len, *data_shape), dtype=torch.float32, device=self.device)
        lengths = torch.zeros(B, dtype=torch.long, device=self.device)
        
        for i, d in enumerate(parsed_data_list):
            l = len(d)
            batch_parsed_data[i, :l] = torch.tensor(d, dtype=torch.float32, device=self.device)
            lengths[i] = l

        # Step 4: 动态调用对应的物理引擎生成三通道张量
        mag, phase = engine_fn(batch_parsed_data, lengths)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        imgs = torch.cat([mag, cos_phase, sin_phase], dim=1) # [B, 3, H, W]

        # Step 5: 送入通用 Encoder 提取 emb0
        with autocast_context(self.device, self.precision):
            encoder_features = self.encoder(imgs)  # [B, L, embed_dim]
        emb0 = encoder_features[:, 0, :].float()   # 取 [CLS] Token，[B, embed_dim]

        # Step 6: 拼接全局元数据，生成最终嵌入
        meta_tensor = torch.tensor(meta_list, dtype=torch.float32, device=self.device) # [B, 4]
        final_emb = torch.cat([emb0, meta_tensor], dim=1) # [B, embed_dim + 4]

        return final_emb

# =====================================================================
# 下游开发者使用示例
# =====================================================================
if __name__ == "__main__":
    poly_1 = np.random.rand(8, 2) * 100 
    poly_2 = np.random.rand(15, 2) * 50  
    raw_polygons = [poly_1, poly_2]
    
    pipeline = PolyFeaturePipeline(
        weight_path='poly_encoder_epoch_100.pth', 
        config_path='poly_mae_config.json'
    )
    
    fused_embeddings = pipeline.extract_features(raw_polygons)
    
    print(f"\n提取成功！")
    print(f"输出特征维度: {fused_embeddings.shape}") 
    print(f"其中前 {pipeline.embed_dim} 维为频域高阶拓扑，后 4 维为 [cx, cy, L, N_scaled]")
