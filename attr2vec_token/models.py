import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# 1. 纯物理真值轨道引擎 (Float64 无损重构)
# ======================================================================
class SimpleCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=4096):
        super().__init__()
        self.t = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh()
        )
        nn.init.normal_(self.t[2].weight, std=0.01)
        
    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse: 
            return torch.cat([x1, x2 + self.t(x1)], dim=1) 
        else: 
            return torch.cat([x1, x2 - self.t(x1)], dim=1)

class InvertibleNetwork(nn.Module):
    def __init__(self, dim, hidden_dim=4096, num_layers=12): 
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleCouplingLayer(dim, hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x, reverse=False):
        # 🌟 核心防弹衣：强制提升为双精度(Float64)进行绝对无损加减
        x_double = x.double()
        
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers: 
            x1, x2 = torch.chunk(x_double, 2, dim=1)
            t_out = layer.t(x1.float()).double() 
            
            if not reverse: 
                x_double = torch.cat([x1, x2 + t_out], dim=1) 
            else: 
                x_double = torch.cat([x1, x2 - t_out], dim=1)
                
        return x_double.float()

# ======================================================================
# 2. 独立语义特征提取轨道 (0.1B FT-Transformer)
# ======================================================================
class FTTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.vocab_size = config.vocab_size
        self.semantic_dim = config.semantic_dim
        self.mask_ratio = config.mask_ratio
        
        self.sem_embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=config.tf_heads, 
            batch_first=True, 
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.tf_layers)
        
        # 漏斗降维：锁定输出为 semantic_dim (默认 256维)
        self.to_semantic = nn.Sequential(
            nn.Linear(self.embed_dim, self.semantic_dim), 
            nn.LayerNorm(self.semantic_dim), 
            nn.Tanh()
        )
        # 用于计算 MAE Loss 的预测头
        self.head_word = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, string_ids):
        B, L = string_ids.shape
        if L == 0:
            return torch.zeros(B, self.semantic_dim, device=string_ids.device), torch.tensor(0.0, device=string_ids.device)
            
        x_emb = self.sem_embedding(string_ids)
        mask = None
        mae_loss = torch.tensor(0.0, device=string_ids.device)
        
        # 训练模式下进行随机 Mask
        if self.training and self.mask_ratio > 0:
            mask = (torch.rand(B, L, device=string_ids.device) < self.mask_ratio).bool()
            x_emb[mask] = self.mask_token.expand(B, L, -1)[mask]
            
        h = self.transformer(x_emb)
        v_semantic = self.to_semantic(h.mean(dim=1))
        
        # 如果有 Mask，计算交叉熵损失
        if mask is not None and mask.any():
            h_masked = h[mask]
            preds = self.head_word(h_masked)
            targets = string_ids[mask]
            mae_loss = F.cross_entropy(preds, targets)
            
        return v_semantic, mae_loss

# ======================================================================
# 3. 大一统底座总控 (自适应拼接与硬件对齐)
# ======================================================================
class Attr2Vec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 实例化真值轨道
        self.inn_core = InvertibleNetwork(
            dim=config.truth_dim, 
            hidden_dim=config.inn_hidden_dim, 
            num_layers=config.inn_layers
        )
        # 🌟 极度关键：强制冻结真值轨道网络，不参与任何反向传播
        for param in self.inn_core.parameters():
            param.requires_grad = False
            
        # 2. 实例化语义轨道 (正常计算梯度)
        self.semantic_core = FTTransformer(config)

    def forward(self, truth_vector, string_ids):
        # ==========================================
        # 🌟 轨道一：真值计算 (纯数学变换，无梯度)
        # ==========================================
        with torch.no_grad():
            if truth_vector.shape[1] > self.config.truth_dim:
                truth_padded = truth_vector[:, :self.config.truth_dim]
            else:
                truth_padded = F.pad(truth_vector, (0, self.config.truth_dim - truth_vector.shape[1]), "constant", 0)
                
            v_truth = self.inn_core(truth_padded)

        # ==========================================
        # 🌟 轨道二：语义计算 (走正常训练流计算 Loss)
        # ==========================================
        v_semantic, mae_loss = self.semantic_core(string_ids)
        
        # ==========================================
        # 🌟 最终合并与硬件友好对齐 (比如向 384 维对齐)
        # ==========================================
        raw_embedding = torch.cat([v_truth, v_semantic], dim=1)
        
        pad_len = self.config.final_dim - raw_embedding.shape[1]
        if pad_len > 0:
            final_embedding = F.pad(raw_embedding, (0, pad_len), "constant", 0)
        else:
            final_embedding = raw_embedding
            
        return final_embedding, mae_loss

    def decode_physical(self, v_attr_padded, num_cont_cols):
        """供下游任务调用：提取前置真值向量并无损解码回 Float64"""
        v_attr_core = v_attr_padded[:, :self.config.truth_dim]
        with torch.no_grad():
            dec = self.inn_core(v_attr_core, reverse=True)
            
        c_exp = dec[:, :num_cont_cols]
        c_hi = dec[:, num_cont_cols:2*num_cont_cols]
        c_lo = dec[:, 2*num_cont_cols:3*num_cont_cols]
        
        dec_cont = torch.ldexp(c_hi.double() + c_lo.double(), torch.round(c_exp).int())
        return dec_cont