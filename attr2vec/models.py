import torch
import torch.nn as nn
import torch.nn.functional as F
from ft_transformer import FTTransformer

torch.set_default_dtype(torch.float32)

class SimpleCouplingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.t = nn.Sequential(
            nn.Linear(dim // 2, dim // 2), 
            nn.ReLU(), 
            nn.Linear(dim // 2, dim // 2),
            nn.Tanh()
        )
        # 正常初始化网络权重，不再使用全零作弊
        nn.init.normal_(self.t[2].weight, std=0.01)

    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse:
            # 🌟 单向条件耦合：物理(x1)决定语义(x2)，语义绝对不篡改物理
            y1 = x1
            y2 = x2 + self.t(x1)
            return torch.cat([y1, y2], dim=1) 
        else:
            y1, y2 = torch.chunk(x, 2, dim=1)
            x1 = y1
            x2 = y2 - self.t(x1)
            return torch.cat([x1, x2], dim=1)

class InvertibleNetwork(nn.Module):
    def __init__(self, dim=512, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([SimpleCouplingLayer(dim) for _ in range(num_layers)])

    def forward(self, x, reverse=False):
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers: x = layer(x, reverse=reverse)
        return x

class NaturalResourceFoundationModel(nn.Module):
    def __init__(self, num_cont_cols, cat_cardinalities, truth_dim=256, semantic_dim=256, total_dim=512):
        super().__init__()
        self.num_cont_cols = num_cont_cols
        self.num_cat_cols = len(cat_cardinalities)
        self.truth_dim = truth_dim
        
        self.semantic_encoder = FTTransformer(
            num_cont_cols=num_cont_cols, 
            cat_cardinalities=cat_cardinalities,
            semantic_dim=semantic_dim
        )
        self.inn_core = InvertibleNetwork(dim=total_dim)

    def forward(self, x_norm, x_int, x_frac, x_cat, mask_ratio=0.25):
        v_sem, p_cont, p_cat, mask = self.semantic_encoder(x_norm, x_cat, mask_ratio)
        
        x_real = torch.cat([x_int, x_frac, x_cat], dim=1)
        x_padded = F.pad(x_real, (0, self.truth_dim - x_real.shape[1]), "constant", 0)
        
        v_attr = self.inn_core(torch.cat([x_padded, v_sem], dim=1))
        return v_attr, v_sem, p_cont, p_cat, mask

    def decode(self, v_attr):
        decoded = self.inn_core(v_attr, reverse=True)[:, :self.truth_dim]
        n = self.num_cont_cols
        
        # 🌟 重点修改：在进行最后的数学合并前，临时升级为 double (float64) 
        # 这样就能容纳下 "80万" 加 "0.1234" 需要的 10 位有效数字，证明模型没有丢数据！
        dec_int = torch.round(decoded[:, :n] * 1_000_000.0).double()
        dec_frac = decoded[:, n:2*n].double()
        dec_cat = torch.round(decoded[:, 2*n : 2*n + self.num_cat_cols] * 100.0)
        
        # 此时的 dec_cont 是 Float64，拥有 15位 精度，足够精确还原了
        dec_cont = torch.round((dec_int + dec_frac) * 10000) / 10000
        
        return dec_cont, dec_cat