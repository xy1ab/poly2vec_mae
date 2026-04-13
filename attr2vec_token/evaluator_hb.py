<<<<<<< HEAD
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
=======
>>>>>>> a52e8d78d0da8da6ece7f8977c2507145940b726
import torch
import os
import json
import joblib
import glob
import pandas as pd
import geopandas as gpd
import fiona
import warnings
import logging
from models import NaturalResourceFoundationModel
from tokenizers import Tokenizer 
from torch.utils.data import TensorDataset, DataLoader
import argparse
from tqdm import tqdm

def build_arg_parser() -> argparse.ArgumentParser:
    """构建南湖平台标准 CLI 解析器"""
    parser = argparse.ArgumentParser(
        description="Extract text corpus and train a custom BPE tokenizer for NRE data."
    )
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory containing raw data (CSV and GDB).")
    parser.add_argument("--output_embedding_dir", type=str, required=True, help="Directory to save the corpus and tokenizer json.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    parser.add_argument("--vocab_path", type=str, required=True, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    parser.add_argument("--model_path", type=str, required=True, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    parser.add_argument("--vocab_size", type=int, default=8192, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    parser.add_argument("--batch_size", type=int, default=2048, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    return parser


class EmbeddingInference:
    def __init__(self, model_path, tokenizer_path, vocab_size, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载分词器
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 2. 模拟/读取配置 (需与训练时保持一致)
        self.config = {
            'truth_dim': 256,
            'semantic_dim': 256,
            'vocab_size': vocab_size, # 根据实际情况调整
            'max_seq_len': 64
        }
        
        # 3. 加载模型
        self.model = NaturalResourceFoundationModel(self.config).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        # 移除可能存在的 module. 前缀
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        
        print(f"✅ 推理引擎已就绪。设备: {self.device}")

    @torch.no_grad()
    def infer_from_embedding(self, latent_tensor, meta):
        """
        核心推理逻辑：从 Embedding 还原原始数据
        :param latent_tensor: 模型生成的特征向量 (Batch, truth_dim + semantic_dim)
        :param meta: 包含列信息的元数据字典
        """
        # A. 提取物理属性向量 (前 truth_dim 维)
        v_attr = latent_tensor[:, :self.config['truth_dim']].to(self.device)
        
        # B. 通过可逆网络 (INN) 进行无损解码
        dec = self.model.inn_core(v_attr, reverse=True)
        
        # C. 物理还原
        num_cont = len(meta.get('cont_cols', []))
        num_word = len(meta.get('word_cols', []))
        num_char = len(meta.get('char_cols', []))
        max_seq_len = self.config['max_seq_len']
        
        results = []
        batch_size = dec.shape[0]

        # --- 1. 还原连续数值 ---
        cont_values = None
        if num_cont > 0:
            c_exp = dec[:, :num_cont]
            c_hi = dec[:, num_cont : 2*num_cont]
            c_lo = dec[:, 2*num_cont : 3*num_cont]
            # 这里的还原逻辑对应训练时的 torch.frexp
            cont_values = torch.ldexp(c_hi.double() + c_lo.double(), c_exp.int())

        # --- 2. 还原文本 (Character-level) ---
        char_texts = []
        if num_char > 0:
            start_idx = 3 * num_cont + num_word
            end_idx = start_idx + (num_char * max_seq_len)
            ch_sc = dec[:, start_idx:end_idx].view(batch_size, num_char, max_seq_len)
            
            # 将 0-1 的归一化数值映射回 Token ID (32768 是原始缩放因子)
            pred_char_ids = torch.round(ch_sc * 32768.0).long()
            
            for i in range(batch_size):
                sample_texts = []
                for c in range(num_char):
                    ids = pred_char_ids[i, c].cpu().tolist()
                    text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                    sample_texts.append(text if text else "")
                char_texts.append(sample_texts)

        # --- 3. 组装结果 ---
        for i in range(batch_size):
            entry = {
                "numerical": {},
                "textual": {}
            }
            if cont_values is not None:
                for idx, col in enumerate(meta['cont_cols']):
                    entry["numerical"][col] = cont_values[i, idx].item()
            
            if char_texts:
                for idx, col in enumerate(meta['char_cols']):
                    entry["textual"][col] = char_texts[i][idx]
            
            results.append(entry)
            
        return results

<<<<<<< HEAD
class DataAuditor:
    def __init__(self, tokenizer, vocab): 
        self.tokenizer = tokenizer

    def render_section(self, source, layer, samples, mode="text"):
        if not samples: return
        print(f"\n{'='*130}\n📊 数据源: {source:<15} | 图层: {layer:<25}")
        print(f"📌 模式: {'🔤 文本一致性对账' if mode=='text' else '📈 连续数值高精度对账'} (展示 {len(samples)} 条)")
        print(f"{'Idx':<4} | {'输入底座的真实原始数据 (Ground Truth)':<60} | {'大模型物理无损解码结果 (Decoded)':<60}\n{'-'*130}")
        for i, (orig, dec) in enumerate(samples):
            print(f"{i+1:<4} | {str(orig):<60} | {str(dec):<60}")
        print(f"{'='*130}")
=======
warnings.filterwarnings('ignore')
>>>>>>> a52e8d78d0da8da6ece7f8977c2507145940b726

def decode_latent_to_output(dec, meta, tokenizer):
    """
    将模型反向传播/解码出的张量还原为原始含义的数据
    """
    num_cont = len(meta.get('cont_cols', []))
    num_word = len(meta.get('word_cols', []))
    num_char = len(meta.get('char_cols', []))
    max_seq_len = meta.get('max_seq_len', 64)
    
    results = {}

    # 1. 还原连续数值 (torch.ldexp 逆操作)
    if num_cont > 0:
        c_exp = dec[:, :num_cont]
        c_hi = dec[:, num_cont : 2*num_cont]
        c_lo = dec[:, 2*num_cont : 3*num_cont]
        # 还原：(hi + lo) * 2^exp
        results['cont_val'] = torch.ldexp(c_hi.double() + c_lo.double(), c_exp.int())

    # 2. 还原字符文本
    if num_char > 0:
        # 定位 char 数据在 dec 中的切片位置
        start_idx = 3 * num_cont + num_word
        end_idx = start_idx + (num_char * max_seq_len)
        ch_sc = dec[:, start_idx:end_idx].view(-1, num_char, max_seq_len)
        
        # 将归一化数值还原为 Token ID
        pred_char_ids = torch.round(ch_sc * 32768.0).long()
        
        decoded_texts = []
        for i in range(pred_char_ids.size(0)):
            batch_texts = []
            for c in range(num_char):
                ids = pred_char_ids[i, c].cpu().tolist()
                text = tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
                batch_texts.append(text if text else "None")
            decoded_texts.append(batch_texts)
        results['char_text'] = decoded_texts

    return results

# ==========================================
# 1. 审计日志配置 (汇总 3 个数据源)
# ==========================================
<<<<<<< HEAD
def evaluate(args):
    tokenizer_path = args.tokenizer_path
    vocab_path = args.vocab_path
    model_path = args.model_path
    print("🔬 [大一统底座] 全要素无损审计系统启动...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError("❌ 未找到 zrzy_tokenizer.json 分词器，请先运行 train_tokenizer_modify.py")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    vocab = json.load(open(vocab_path, "r", encoding="utf-8")) if os.path.exists(vocab_path) else {}
    auditor = DataAuditor(tokenizer, vocab)

    config = {
        'truth_dim': 256,
        'semantic_dim': 256,
        'vocab_size': args.vocab_size,
        'max_seq_len': 64
    }
    model = NaturalResourceFoundationModel(config).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到权重文件 {model_path}，请先完成训练。")
    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
=======
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
    
    # --- 1. 加载基建 ---
    tokenizer = Tokenizer.from_file("zrzy_tokenizer.json")
    vocab_path = "global_vocab_auto.json"
    rev_vocab = {}
    if os.path.exists(vocab_path):
        vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
        for col, mapping in vocab.items():
            if isinstance(mapping, dict):
                rev_vocab[col] = {v: k for k, v in mapping.items()}

    # --- 2. 加载模型 (Truth Dim 固定为 256) ---
    config = {'truth_dim': 256, 'semantic_dim': 256, 'vocab_size': 20000}
    model = NaturalResourceFoundationModel(config).to(device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load("best_model.pth", map_location=device).items()})
>>>>>>> a52e8d78d0da8da6ece7f8977c2507145940b726
    model.eval()
    log_print("✅ 0.1B 核心模型已唤醒，准备穿透硬盘数据...\n")

<<<<<<< HEAD
    cache_files = glob.glob(args.cache_dir + "/*.joblib")
    if not cache_files:
        print("⚠️ 未找到任何 cache_*.joblib 数据缓存文件！")
        return

    all_results = {}

    with torch.no_grad():
        for cache_file in cache_files:
            source_name = os.path.basename(cache_file).replace("cache_", "").replace(".joblib", "")
            data_dict = joblib.load(cache_file)
            layer_results = {}
            for layer_name, layer_data in data_dict.items():
                meta = layer_data.get('meta', {})
                num_cont = len(meta.get('cont_cols', []))
                num_char = len(meta.get('char_cols', []))


                dataset = TensorDataset(
                    torch.tensor(layer_data['cont_int']),
                    torch.tensor(layer_data['cont_frac_hi']),
                    torch.tensor(layer_data['cont_frac_lo']),
                    torch.tensor(layer_data['cont_norm']),
                    torch.tensor(layer_data['word_data']),
                    torch.tensor(layer_data['char_data'])
                )
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                layer_latent_list = []
                layer_stats = {'cont_acc_sum': 0.0, 'char_acc_sum': 0.0, 'batch_count': 0}
                batch_pbar = tqdm(dataloader, desc=f"  -> {layer_name}", leave=False, unit="batch")
                for batch in batch_pbar:
                    # 将数据一次性移动到指定的 device
                    b = [item.to(device) for item in batch]
                    b0, b1, b2, b3, b4, b5 = b

                    latent, _ = model(b0, b1, b2, b3, b4, b5)
                    layer_latent_list.append(latent.cpu())
 
                    
                    v_attr = latent[:, :config['truth_dim']]
                    dec = model.inn_core(v_attr, reverse=True)
                    pred_out = decode_latent_to_output(dec, meta, tokenizer)

                    res = {'cont_acc': 0.0, 'char_acc': 0.0}
                    if num_cont > 0:
                        gt_cont = torch.ldexp(b1.double() + b2.double(), b0.int())
                        is_correct = torch.isclose(pred_out['cont_val'], gt_cont, rtol=1e-9, atol=1e-9)
                        res['cont_acc'] = is_correct.float().mean().item()
                    if num_char > 0:
                        gt_char_ids = torch.round(b5 * 32768.0).long()
                        # 重新从 dec 拿 ID 方便对比
                        start_idx = 3 * num_cont + len(meta.get('word_cols', []))
                        ch_sc = dec[:, start_idx : start_idx + (num_char * meta['max_seq_len'])]
                        pred_ids = torch.round(ch_sc.view(-1, num_char, meta['max_seq_len']) * 32768.0).long()
                        
                        correct = (pred_ids == gt_char_ids).float().mean()
                        res['char_acc'] = correct.item()
                    layer_stats['cont_acc_sum'] += res['cont_acc']
                    layer_stats['char_acc_sum'] += res['char_acc']
                    layer_stats['batch_count'] += 1

                # --- 计算并打印该层平均指标 ---
                avg_cont_acc = layer_stats['cont_acc_sum'] / layer_stats['batch_count']
                avg_char_acc = layer_stats['char_acc_sum'] / layer_stats['batch_count']
                
                print(f"--- Layer: {layer_name} ---")
                print(f"  数值还原准确率 (Avg): {avg_cont_acc:.2%}")
                print(f"  文本还原准确率 (Avg): {avg_char_acc:.2%}")
                layer_results[layer_name] ={
                    'latent': torch.cat(layer_latent_list, dim=0),
                    'meta': meta
                    } 
            all_results[source_name] = layer_results
               
                 
    joblib.dump(all_results, os.path.join(args.output_embedding_dir,"encoded_latents.joblib"))

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    
    evaluate(args)
    
=======
    # --- 3. 自动搜索数据源 (精准锁定前 3 个) ---
    raw_files = sorted(list(set(glob.glob("*.gdb") + glob.glob("raw_data/*.gdb") + glob.glob("*.csv") + glob.glob("raw_data/*.csv"))))[:3]
    if not raw_files:
        log_print("⚠️ 未找到任何 GDB/CSV 文件，请检查路径。")
        return

    for f_idx, raw_path in enumerate(raw_files):
        ext = raw_path.split('.')[-1].lower()
        src_name = os.path.basename(raw_path).replace(f".{ext}", "")
        log_print("*" * 140)
        log_print(f"🟢 [源 {f_idx+1}/3] 原始物理文件: {raw_path}")

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

        # B. 提取对应的 PT 缓存 (对应首行)
        cache_path = f"cache_{src_name}.pt"
        if not os.path.exists(cache_path): 
            log_print(f"⚠️ 跳过: 缺失对应的张量缓存 {cache_path}"); continue

        data_dict = torch.load(cache_path, map_location=device)
        layer_key = list(data_dict.keys())[0] # 取缓存的第一个图层
        layer_data = data_dict[layer_key]
        meta = layer_data['meta']
        
        # 准备编码张量 (只切第一行)
        inputs = [torch.tensor(layer_data[k][:1], device=device) for k in ['cont_int', 'cont_frac_hi', 'cont_frac_lo', 'cont_norm', 'word_data', 'char_data']]

        # C. 【核心闭环】正向 Embedding -> 逆向解码
        with torch.no_grad():
            latent, _ = model(*inputs)
            v_attr = latent[:, :config['truth_dim']] # 提取正向压缩后的 256 维 Embedding
            dec = model.inn_core(v_attr, reverse=True) # 逆向物理暴胀

        # D. 物理空间动态还原 (防 117 维切片崩溃)
        nc, nw, nch = len(meta['cont_cols']), len(meta['word_cols']), len(meta['char_cols'])
        c_exp, c_hi, c_lo = dec[:, :nc], dec[:, nc:2*nc], dec[:, 2*nc:3*nc]
        w_sc = dec[:, 3*nc : 3*nc+nw]
        
        rem_dim = config['truth_dim'] - (3*nc + nw)
        model_seq_len = rem_dim // nch if nch > 0 else 0
        ch_sc = dec[:, 3*nc+nw : 3*nc+nw + (nch * model_seq_len)].view(-1, nch, model_seq_len)

        # E. 渲染 1:1 对账单
        log_print(f"🎯 目标图层: {layer} | 字段总数: {nc+nw+nch}")
        log_print("-" * 140)
        log_print(f"{'类型':<4} | {'字段名称 (Field)':<25} | {'原始硬盘明文 (Raw GDB/CSV)':<45} | {'Embedding 逆向解码 (Decoded)'}")
        log_print("-" * 140)

        # 1. 数值还原
        recon_num = torch.ldexp(c_hi.double() + c_lo.double(), c_exp.int())
        for i, col in enumerate(meta['cont_cols']):
            log_print(f"[数] | {col:<25} | {str(raw_row.get(col, 'nan')):<45} | {recon_num[0, i].item():.6f}")

        # 2. 字典还原 (🌟 Multiplier 16384)
        pred_w_ids = torch.round(w_sc * 16384.0).long()
        for i, col in enumerate(meta['word_cols']):
            # 防守 Pandas 幽灵 .0
            raw_v = str(raw_row.get(col, ''))
            if raw_v.endswith('.0'): raw_v = raw_v[:-2]
            if not raw_v or raw_v == 'nan': raw_v = "[空]"
            
            pid = pred_w_ids[0, i].item()
            dec_v = rev_vocab.get(col, {}).get(pid, f"ID:{pid}") if pid != 0 else "[空/填充]"
            log_print(f"[类] | {col:<25} | {raw_v:<45} | {dec_v}")

        # 3. 文本还原 (🌟 Multiplier 32768)
        pred_c_ids = torch.round(ch_sc * 32768.0).long()
        for i, col in enumerate(meta['char_cols']):
            raw_v = str(raw_row.get(col, ''))
            if not raw_v or raw_v == 'nan': raw_v = "[空]"
            
            dec_ids = pred_c_ids[0, i].tolist()
            dec_v = tokenizer.decode(dec_ids, skip_special_tokens=True).replace(" ","")
            log_print(f"[文] | {col:<25} | {raw_v:<45} | {dec_v if dec_v else '[空]'}")

        log_print("*" * 140 + "\n")

    log_print(f"✅ 三源对账完毕！完整清单已保存至: {log_file}")

if __name__ == "__main__":
    run_full_audit()
>>>>>>> a52e8d78d0da8da6ece7f8977c2507145940b726
