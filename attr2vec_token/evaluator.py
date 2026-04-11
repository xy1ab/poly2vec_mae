import torch
import os
import json
import glob
import pandas as pd
import geopandas as gpd
import fiona
import warnings
import logging
import numpy as np
from models import NaturalResourceFoundationModel
from tokenizers import Tokenizer 

warnings.filterwarnings('ignore')

# ==========================================
# 1. 审计日志配置 (汇总 3 个数据源)
# ==========================================
log_file = 'multi_source_full_audit.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def log_print(msg):
    logging.info(msg)

def run_full_audit():
    log_print("🔬 [大一统底座] 启动多源 (GDB/CSV) 1:1 端到端全要素审计...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载基建
    tokenizer = Tokenizer.from_file("zrzy_tokenizer.json")
    vocab_path = "global_vocab_auto.json"
    rev_vocab = {}
    if os.path.exists(vocab_path):
        vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
        for col, mapping in vocab.items():
            if isinstance(mapping, dict):
                rev_vocab[col] = {v: k for k, v in mapping.items()}

    # 模型加载 (Truth Dim 固定为 256)
    config = {'truth_dim': 256, 'semantic_dim': 256, 'vocab_size': 20000}
    model = NaturalResourceFoundationModel(config).to(device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load("best_model.pth", map_location=device).items()})
    model.eval()
    log_print("✅ 0.1B 核心模型已唤醒，准备处理数据源...\n")

    # 自动搜索数据源 (前 3 个)
    raw_files = sorted(list(set(glob.glob("*.gdb") + glob.glob("raw_data/*.gdb") + glob.glob("*.csv") + glob.glob("raw_data/*.csv"))))[:3]
    
    for f_idx, raw_path in enumerate(raw_files):
        ext = raw_path.split('.')[-1].lower()
        src_name = os.path.basename(raw_path).replace(f".{ext}", "")
        log_print("*" * 140)
        log_print(f"🟢 [源 {f_idx+1}] 原始文件: {raw_path}")

        # A. 提取物理硬盘首行
        try:
            if ext == 'gdb':
                layer = fiona.listlayers(raw_path)[0]
                raw_row = gpd.read_file(raw_path, layer=layer, rows=1, ignore_geometry=True, engine="pyogrio").iloc[0]
            else:
                layer = src_name
                raw_row = pd.read_csv(raw_path, nrows=1, low_memory=False).iloc[0]
        except Exception as e:
            log_print(f"❌ 原始文件读取失败: {e}"); continue

        # B. 提取对应的 PT 缓存 Embedding
        cache_path = f"cache_{src_name}.pt"
        if not os.path.exists(cache_path): log_print(f"⚠️ 缺失缓存 {cache_path}"); continue

        data_dict = torch.load(cache_path, map_location=device)
        layer_key = list(data_dict.keys())[0]
        layer_data = data_dict[layer_key]
        meta = layer_data['meta']
        
        # 准备编码张量
        inputs = [torch.tensor(layer_data[k][:1], device=device) for k in ['cont_int', 'cont_frac_hi', 'cont_frac_lo', 'cont_norm', 'word_data', 'char_data']]

        # C. 核心闭环：正向 Embedding -> 逆向解码
        with torch.no_grad():
            latent, _ = model(*inputs)
            v_attr = latent[:, :config['truth_dim']] 
            dec = model.inn_core(v_attr, reverse=True) # 这里吐出的是 256 维物理空间

        # D. 物理空间动态还原 (关键修复)
        nc, nw, nch = len(meta['cont_cols']), len(meta['word_cols']), len(meta['char_cols'])
        c_exp, c_hi, c_lo = dec[:, :nc], dec[:, nc:2*nc], dec[:, 2*nc:3*nc]
        w_sc = dec[:, 3*nc : 3*nc+nw]
        
        # 计算模型留给文本的实际长度 (117 / nch)
        rem_dim = config['truth_dim'] - (3*nc + nw)
        model_seq_len = rem_dim // nch if nch > 0 else 0
        ch_sc = dec[:, 3*nc+nw : 3*nc+nw + (nch * model_seq_len)].view(-1, nch, model_seq_len)

        # E. 渲染全量对账单
        log_print(f"🎯 图层: {layer} | 属性维度: {config['truth_dim']} | 可容纳文本长度: {model_seq_len}")
        log_print("-" * 140)
        log_print(f"{'类型':<4} | {'字段名称 (Field)':<25} | {'原始明文 (Raw)':<45} | {'模型解码 (Decoded)'}")
        log_print("-" * 140)

        # 1. 数值还原 (三路逼近)
        recon_num = torch.ldexp(c_hi.double() + c_lo.double(), c_exp.int())
        for i, col in enumerate(meta['cont_cols']):
            log_print(f"[数] | {col:<25} | {str(raw_row.get(col, 'nan')):<45} | {recon_num[0, i].item():.10f}")

        # 2. 字典还原 (Multiplier 16384)
        pred_w_ids = torch.round(w_sc * 16384.0).long()
        for i, col in enumerate(meta['word_cols']):
            raw_v = str(raw_row.get(col, '')).replace('.0','')
            if not raw_v or raw_v == 'nan': raw_v = "[空]"
            pid = pred_w_ids[0, i].item()
            dec_v = rev_vocab.get(col, {}).get(pid, f"ID:{pid}") if pid != 0 else "[空/填充]"
            log_print(f"[类] | {col:<25} | {raw_v:<45} | {dec_v}")

        # 3. 文本还原 (Multiplier 32768)
        pred_c_ids = torch.round(ch_sc * 32768.0).long()
        for i, col in enumerate(meta['char_cols']):
            raw_v = str(raw_row.get(col, ''))
            if not raw_v or raw_v == 'nan': raw_v = "[空]"
            dec_ids = pred_c_ids[0, i].tolist()
            dec_v = tokenizer.decode(dec_ids, skip_special_tokens=True).replace(" ","")
            log_print(f"[文] | {col:<25} | {raw_v:<45} | {dec_v if dec_v else '[空]'}")

        log_print("*" * 140 + "\n")

if __name__ == "__main__":
    run_full_audit()