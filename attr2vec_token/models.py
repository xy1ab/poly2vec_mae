import torch
import torch.nn as nn
import torch.nn.functional as F

"""class SimpleCouplingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2), nn.Tanh())
        nn.init.normal_(self.t[2].weight, std=0.01)
    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse: return torch.cat([x1, x2 + self.t(x1)], dim=1) 
        else: return torch.cat([x1, x2 - self.t(x1)], dim=1)"""

"""class InvertibleNetwork(nn.Module):
    def __init__(self, dim=256, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([SimpleCouplingLayer(dim) for _ in range(num_layers)])
    def forward(self, x, reverse=False):
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers: x = layer(x, reverse=reverse)
        return x"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCouplingLayer(nn.Module):
    # 🌟 核心修改：接收 hidden_dim，让内部网络膨胀到 4096
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

# 2026-4-9 解决大于128维属性字段进入有损区问题 + 内部隐层膨胀
class InvertibleNetwork(nn.Module):
    # 🌟 核心修改：接收 hidden_dim，并传递给 SimpleCouplingLayer
    def __init__(self, dim=256, hidden_dim=4096, num_layers=12): 
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleCouplingLayer(dim, hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x, reverse=False):
        # 🌟 终极防弹衣：在进入危险的可逆加减法前，强制提升为双精度(Float64)
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

class PhysicalTokenizer(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)
    def forward(self, x): return self.proj(x.unsqueeze(-1))

class NaturalResourceFoundationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 🌟 永远坚守的业务底线：256 + 256
        self.truth_dim, self.semantic_dim = config.get('truth_dim', 256), config.get('semantic_dim', 256)
        
        # 🌟 0.1B 架构的参数源泉
        self.embed_dim = config.get('embed_dim', 768)  # 从 64 暴涨到 768
        self.mask_ratio = config.get('mask_ratio', 0.20)
        self.vocab_size = config.get('vocab_size', 20000)
        
        # 🌟 INN 12层，内部隐藏层 4096
        self.inn_core = InvertibleNetwork(dim=self.truth_dim, hidden_dim=4096, num_layers=12)
        self.phys_tokenizer = PhysicalTokenizer(self.embed_dim)
        self.sem_embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # 🌟 Transformer 12头，12层
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=12, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        
        # 🌟 漏斗降维：无论内部多大，最后乖乖变回 semantic_dim (256维)
        self.to_semantic = nn.Sequential(nn.Linear(self.embed_dim, self.semantic_dim), nn.LayerNorm(self.semantic_dim), nn.Tanh())
        
        self.head_cont = nn.Linear(self.embed_dim, 1)
        self.head_word = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, cont_int, cont_f_hi, cont_f_lo, cont_norm, word_data, char_data):
        device = cont_int.device
        B = cont_int.shape[0] if cont_int.shape[1] > 0 else (word_data.shape[0] if word_data.shape[1] > 0 else char_data.shape[0])
        
        char_flat = char_data.reshape(B, -1) if char_data.shape[1] > 0 else torch.zeros(B, 0, device=device)
        x_real = torch.cat([cont_int, cont_f_hi, cont_f_lo, word_data.float(), char_flat.float()], dim=1)

        # ======================================================================
        # 🌟 核心安全锁：绝对禁止张量静默截断与跨行污染
        # ======================================================================
        if x_real.shape[1] > self.truth_dim:
            raise ValueError(
                f"🚨 物理潜空间爆缸！当前数据展平后需要 {x_real.shape[1]} 维容量，"
                f"但业务架构强约束上限为 {self.truth_dim} 维！\n"
                f"继续执行将导致负数 Padding 和灾难性的跨行显存污染。\n"
                f"请在 data_loader.py 中调小 max_seq_len（如 16 或 32），或筛减不必要的长文本列。"
            )
        
        x_padded = F.pad(x_real, (0, self.truth_dim - x_real.shape[1]), "constant", 0)
        v_attr = self.inn_core(x_padded)

        tokens_to_mask, all_tokens = [], []
        L_cont, L_word = 0, 0
        
        if cont_norm.shape[1] > 0:
            t = self.phys_tokenizer(cont_norm)
            tokens_to_mask.append(t)
            L_cont = t.shape[1]
            
        if word_data.shape[1] > 0:
            w = self.sem_embedding(torch.round(word_data * 32768.0).long())
            tokens_to_mask.append(w)
            L_word = w.shape[1]
        
        x_maskable = torch.cat(tokens_to_mask, dim=1) if tokens_to_mask else None
        mae_loss = sum(p.sum() * 0 for p in self.parameters())
        mask = None
        
        if x_maskable is not None and self.training and self.mask_ratio > 0:
            L_maskable = x_maskable.shape[1]
            mask = (torch.rand(B, L_maskable, device=device) < self.mask_ratio).bool()
            x_masked = x_maskable.clone()
            x_masked[mask] = self.mask_token.expand(B, L_maskable, -1)[mask]
        else: 
            x_masked = x_maskable
            
        if x_masked is not None: all_tokens.append(x_masked)
        if char_data.shape[1] > 0:
            all_tokens.append(self.sem_embedding(torch.round(char_data * 32768.0).long()).mean(dim=2))
        
        x_final = torch.cat(all_tokens, dim=1) if all_tokens else torch.zeros(B, 0, self.embed_dim, device=device)
        
        if x_final.shape[1] > 0:
            h = self.transformer(x_final)
            v_semantic = self.to_semantic(h.mean(dim=1))
            if mask is not None:
                start = 0
                if L_cont > 0:
                    m = mask[:, :L_cont]
                    if m.any():
                        # 🌟 显存拯救：仅提取有掩码的部分送入投影头
                        h_cont_masked = h[:, :L_cont][m]
                        preds = self.head_cont(h_cont_masked).squeeze(-1)
                        targets = cont_norm[m] 
                        mae_loss += F.mse_loss(preds, targets)
                    start = L_cont
                if L_word > 0:
                    m = mask[:, start:start+L_word]
                    if m.any():
                        # 🌟 显存拯救：仅过滤出被遮挡的词去算 20000 维分类，暴省 80% 显存
                        h_word_masked = h[:, start:start+L_word][m]
                        preds = self.head_word(h_word_masked)
                        targets = torch.round(word_data * 32768.0).long()[m]
                        mae_loss += F.cross_entropy(preds, targets)
        else: v_semantic = torch.zeros(B, self.semantic_dim, device=device)
            
        return torch.cat([v_attr, v_semantic], dim=1), mae_loss

    # 🌟 解封长度限制：要求必须传入 max_seq_len，不再默认截断
    def decode_physical(self, v_attr, num_cont, num_word, num_char, max_seq_len):
        dec = self.inn_core(v_attr, reverse=True)
        c_exp = dec[:, :num_cont]; c_hi = dec[:, num_cont:2*num_cont]; c_lo = dec[:, 2*num_cont:3*num_cont]
        w_sc = dec[:, 3*num_cont:3*num_cont+num_word]
        ch_sc = dec[:, 3*num_cont+num_word:3*num_cont+num_word+num_char*max_seq_len]
        dec_cont = torch.ldexp(c_hi.double() + c_lo.double(), torch.round(c_exp).int())
        dec_word = torch.round(w_sc * 32768.0).long()
        dec_char = torch.round(ch_sc * 32768.0).long().reshape(-1, num_char, max_seq_len) if num_char > 0 else torch.zeros((v_attr.shape[0], 0, max_seq_len), dtype=torch.long, device=v_attr.device)
        return dec_cont, dec_word, dec_char