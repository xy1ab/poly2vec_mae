import torch
import torch.nn as nn

# 强制全局单精度
torch.set_default_dtype(torch.float32)

class FTTransformer(nn.Module):
    def __init__(self, num_cont_cols, cat_cardinalities, embed_dim=768, depth=12, heads=12, semantic_dim=256, max_vocab=500):
        super().__init__()
        self.num_cont_cols = num_cont_cols
        self.num_cat_cols = len(cat_cardinalities)
        self.total_features = num_cont_cols + self.num_cat_cols
        self.max_vocab = max_vocab
        
        # 连续特征映射
        self.cont_embeddings = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_cont_cols)])
        # 核心防爆：限制 Embedding 词表大小，防止 ID 类字段导致显存爆炸
        self.cat_embeddings = nn.ModuleList([nn.Embedding(min(c, max_vocab), embed_dim) for c in cat_cardinalities])
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 0.1B 参数规模：12层，768维
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4, 
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 语义压缩层
        self.to_semantic_latent = nn.Sequential(
            nn.Linear(embed_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.Tanh()
        )
        
        self.cont_predictors = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_cont_cols)])
        self.cat_predictors = nn.ModuleList([nn.Linear(embed_dim, min(c, max_vocab)) for c in cat_cardinalities])

    def forward(self, x_cont, x_cat, mask_ratio=0.25, dynamic_mask_idx=None):
        B = x_cont.shape[0] if x_cont is not None else x_cat.shape[0]
        tokens = []
        
        # 连续特征处理 (增加 unsqueeze(1) 修复维度报错)
        if self.num_cont_cols > 0:
            for i in range(self.num_cont_cols):
                tokens.append(self.cont_embeddings[i](x_cont[:, i:i+1]).unsqueeze(1))
        
        # 类别特征处理 (取余处理防止索引越界)
        if self.num_cat_cols > 0:
            for i in range(self.num_cat_cols):
                clamped_cat = x_cat[:, i].long() % self.max_vocab
                tokens.append(self.cat_embeddings[i](clamped_cat).unsqueeze(1))
                
        x_tokens = torch.cat(tokens, dim=1) 
        mask = torch.zeros(B, self.total_features, dtype=torch.bool, device=x_tokens.device)
        
        # 25% 掩码逻辑
        if self.training and mask_ratio > 0:
            mask = torch.rand(B, self.total_features, device=x_tokens.device) < mask_ratio
            
        if dynamic_mask_idx is not None:
            mask[:, dynamic_mask_idx] = True

        if mask.any():
            x_tokens[mask] = self.mask_token.squeeze()
        
        # Transformer 编码
        x_transformer = torch.cat([self.cls_token.expand(B, -1, -1), x_tokens], dim=1)
        encoded_tokens = self.transformer(x_transformer)
        
        v_semantic = self.to_semantic_latent(encoded_tokens[:, 0, :])
        feature_tokens = encoded_tokens[:, 1:, :]
        
        # 掩码预测头输出
        pred_cont = None
        if self.num_cont_cols > 0:
            pred_cont = torch.cat([self.cont_predictors[i](feature_tokens[:, i, :]) for i in range(self.num_cont_cols)], dim=1)
            
        pred_cat = []
        if self.num_cat_cols > 0:
            pred_cat = [self.cat_predictors[i](feature_tokens[:, self.num_cont_cols + i, :]) for i in range(self.num_cat_cols)]
                
        return v_semantic, pred_cont, pred_cat, mask