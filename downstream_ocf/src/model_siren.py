# downstream_ocf/src/model_siren.py
import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class FiLMBlock(nn.Module):
    """384维 Embedding 调制模块"""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )

    def forward(self, x, v):
        # v: [B, 384], x: [B, N, hidden_dim]
        style = self.mlp(v).unsqueeze(1) # [B, 1, hidden_dim * 2]
        gamma, beta = torch.chunk(style, 2, dim=-1)
        return gamma * x + beta

class FiLMSirenOCF(nn.Module):
    def __init__(self, embed_dim=384, hidden_dim=256, num_layers=5, omega_0=30):
        super().__init__()
        self.num_layers = num_layers
        
        # 第一层：坐标投影
        self.first_layer = SineLayer(2, hidden_dim, is_first=True, omega_0=omega_0)
        
        # 中间层：Sine + FiLM
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=1.0))
            self.films.append(FiLMBlock(embed_dim, hidden_dim))
        
        self.final_linear = nn.Linear(hidden_dim, 1)

    def forward(self, coords, v):
        # coords: [B, N, 2], v: [B, 384]
        x = self.first_layer(coords)
        
        for i in range(len(self.layers)):
            identity = x # 残差路径
            x = self.layers[i](x)
            x = self.films[i](x, v)
            x = x + identity 
            
        return torch.sigmoid(self.final_linear(x))