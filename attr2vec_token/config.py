import json
import os
import math

def get_optimal_dim(raw_dim, align=64):
    """向上取整到 align 的倍数，以获得最佳 Tensor Core 硬件加速性能"""
    return math.ceil(raw_dim / align) * align

class ModelConfig:
    def __init__(self):
        # ==========================================
        # 🌟 维度与架构设计参数
        # ==========================================
        self.truth_dim = 256         # 默认值，将被 data_builder 全局嗅探后动态覆盖
        self.semantic_dim = 256      # 语义轨道固定维度 (0.1B 模型的降维出口)
        
        # 🌟 0.1B FT-Transformer 参数
        self.embed_dim = 768         
        self.vocab_size = 65536      
        self.tf_layers = 12          
        self.tf_heads = 12           
        
        # 🌟 真值轨道 (INN) 参数
        self.inn_hidden_dim = 4096
        self.inn_layers = 12
        
        # ==========================================
        # 🌟 训练超参数 & 路径 (保持不变)
        # ==========================================
        self.batch_size = 2048       
        self.epochs = 10
        self.base_lr = 2e-4
        self.warmup_epochs = 2
        self.mask_ratio = 0.20
        self.data_dir = "./raw_data"
        self.output_dir = "./output"
        self.tokenizer_path = os.path.join(self.output_dir, "zrzy_tokenizer.json")
        
    @property
    def final_dim(self):
        """🌟 核心动态逻辑：真值维度 + 语义维度后，向上取整到 64 的倍数"""
        raw_total = self.truth_dim + self.semantic_dim
        return get_optimal_dim(raw_total, align=64)

    def save(self, path="model_config.json"):
        # 保存时，将 property 也固化下来方便排查
        data_to_save = self.__dict__.copy()
        data_to_save['final_dim'] = self.final_dim 
        with open(path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
            
    def load(self, path="model_config.json"):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                # 忽略加载 final_dim，因为它是由 property 动态计算的
                data.pop('final_dim', None) 
                self.__dict__.update(data)