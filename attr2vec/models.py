import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCouplingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2), nn.Tanh())
        nn.init.normal_(self.t[2].weight, std=0.01)
    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse: return torch.cat([x1, x2 + self.t(x1)], dim=1) 
        else: return torch.cat([x1, x2 - self.t(x1)], dim=1)

class InvertibleNetwork(nn.Module):
    def __init__(self, dim=256, num_layers=12): 
        super().__init__()
        self.layers = nn.ModuleList([SimpleCouplingLayer(dim) for _ in range(num_layers)])
    def forward(self, x, reverse=False):
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers: x = layer(x, reverse=reverse)
        return x

class PhysicalTokenizer(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)
    def forward(self, x): return self.proj(x.unsqueeze(-1))

class NaturalResourceFoundationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.truth_dim, self.semantic_dim = 256, 256
        self.embed_dim = 768 
        self.vocab_size = config.get('vocab_size', 20000)
        self.mask_ratio = 0.20
        
        self.inn_core = InvertibleNetwork(dim=self.truth_dim, num_layers=12)
        self.phys_tokenizer = PhysicalTokenizer(self.embed_dim)
        self.sem_embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=12, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.to_semantic = nn.Sequential(nn.Linear(self.embed_dim, self.semantic_dim), nn.LayerNorm(self.semantic_dim), nn.Tanh())
        
        self.head_cont = nn.Linear(self.embed_dim, 1)
        self.head_word = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, cont_int, cont_frac_hi, cont_frac_lo, cont_norm, word_data, char_data):
        device = cont_int.device; B = cont_int.shape[0]
        char_flat = char_data.reshape(B, -1) if char_data.shape[1] > 0 else torch.zeros(B, 0, device=device)
        x_real = torch.cat([cont_int, cont_frac_hi, cont_frac_lo, word_data.float(), char_flat.float()], dim=1)
        x_padded = F.pad(x_real, (0, self.truth_dim - x_real.shape[1]), "constant", 0)
        v_attr = self.inn_core(x_padded)

        # 🌟 核心防爆盾：强制限制解码 ID 范围
        def safe_idx(x):
            return torch.clamp(torch.round(x * 16384.0).long(), 0, self.vocab_size - 1)

        tokens_to_mask = []; L_cont, L_word = 0, 0
        if cont_norm.shape[1] > 0:
            t = self.phys_tokenizer(cont_norm); tokens_to_mask.append(t); L_cont = t.shape[1]
        if word_data.shape[1] > 0:
            w = self.sem_embedding(safe_idx(word_data)); tokens_to_mask.append(w); L_word = w.shape[1]
        
        x_maskable = torch.cat(tokens_to_mask, dim=1) if tokens_to_mask else None
        mae_loss = sum(p.sum() * 0 for p in self.parameters())
        if x_maskable is not None and self.training:
            L_m = x_maskable.shape[1]; mask = (torch.rand(B, L_m, device=device) < self.mask_ratio).bool()
            x_masked = x_maskable.clone(); x_masked[mask] = self.mask_token.expand(B, L_m, -1)[mask]
            all_tokens = [x_masked]
            if char_data.shape[1] > 0: all_tokens.append(self.sem_embedding(safe_idx(char_data)).mean(dim=2))
            h = self.transformer(torch.cat(all_tokens, dim=1)); v_semantic = self.to_semantic(h.mean(dim=1))
            m_cont = mask[:, :L_cont]
            if m_cont.any(): mae_loss += F.mse_loss(self.head_cont(h[:, :L_cont][m_cont]).squeeze(-1), cont_norm[m_cont])
            m_word = mask[:, L_cont:L_cont+L_word]
            if m_word.any(): mae_loss += F.cross_entropy(self.head_word(h[:, L_cont:L_cont+L_word][m_word]), safe_idx(word_data)[m_word])
        else: v_semantic = torch.zeros(B, self.semantic_dim, device=device)
            
        return torch.cat([v_attr, v_semantic], dim=1), mae_loss

    def decode_physical(self, v_attr):
        """物理保真轨道：全量无损逆向解码"""
        return self.inn_core(v_attr, reverse=True)