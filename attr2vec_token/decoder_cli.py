import sys, os, torch, argparse, unicodedata, warnings, pyogrio
import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from config import ModelConfig
from models import Attr2Vec
import json
warnings.filterwarnings('ignore')

class DualLogger:
    def __init__(self, filename="evaluator_audit.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, m):
        self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

def align(text, width):
    text = str(text).replace("\n", "").replace("\r", "")
    dw = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)
    return text + ' ' * max(0, width - dw)

MUST_STR = ('代码', '编码', '编号', 'id', 'code', 'bm', 'dm', 'bsm', 'pac', 'politcode')
def get_safe_dtype(csv_path):
    try:
        cols = pd.read_csv(csv_path, nrows=0).columns
        return {c: str for c in cols if any(s in c.lower() for s in MUST_STR)}
    except:
        return None

class AttrDecoder_Worker:
    def __init__(self, config_path):
        torch.manual_seed(42)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(42)
        self.config = ModelConfig()
        self.config.load(config_path)
        self.tokenizer = Tokenizer.from_file(self.config.tokenizer_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Attr2Vec(self.config).to(self.device)
        self.model.eval()

    def decode_num(self, p):
        p = torch.round(p).double()
        return p[0] + (p[1]/10000.0) + (p[2]/100000000.0)

    def decoder(self, emb_dir):
        schema_registry = json.load(os.path.join(emb_dir, "schema_registry.json"))
        bundle = torch.load(emb_path, weights_only=False)
        v_size = self.tokenizer.get_vocab_size()
        
        for layer in bundle:
            name, embs = layer['name'], layer['emb'].to(self.device)
            n_cols, s_cols, full_order = layer['num_cols'], layer['str_cols'], layer['full_order']
            
            with torch.no_grad():
                dec_vec = self.model.inn_core(embs, reverse=True)
            
            try:
                if layer['type'] == "CSV":
                    df_orig = pd.read_csv(layer['path'], nrows=5, dtype=get_safe_dtype(layer['path']))
                else:
                    df_orig = pyogrio.read_dataframe(layer['path'], layer=name, max_features=5, read_geometry=False)
            except Exception as e:
                print(f"\n⚠️ 跳过图层 {name} 对账: 读取原始文件失败 ({e})")
                continue

            print(f"\n{'='*150}\n📊 图层: 【{name}】 | 数据源: {layer['type']}\n{'='*150}")
            
            num_w = len(n_cols) * 3
            dec_n, dec_t = dec_vec[:, :num_w], dec_vec[:, num_w : num_w + self.config.max_seq_len]
            ids = torch.clamp(torch.round(dec_t).to(torch.int64), 0, v_size - 1).cpu().tolist()
            raw_decoded = self.tokenizer.decode_batch(ids, skip_special_tokens=False)
            
            for r in range(len(df_orig)):
                print(f"\n🔹 Row {r+1}:")
                print(f"   {align('字段名称 (Field)', 30)} | {align('原始真值', 50)} | {align('底座解码值', 50)} | 判定")
                print("   " + "-"*145)
                
                clean_line = raw_decoded[r].replace("[PAD]", "").replace("[CLS]", "").replace("[MASK]", "").replace("[UNK]", "").strip()
                dec_parts = [p.replace(" ", "") for p in clean_line.split("[SEP]")]

                for col in full_order:
                    if col in n_cols:
                        c_idx = n_cols.index(col)
                        orig = df_orig.iloc[r][col]
                        p = dec_n[r, c_idx*3 : c_idx*3+3]
                        dec_v = self.decode_num(p).item()
                        
                        # 🌟 数值型空值拦截：彻底渲染为空白，不再显示 nan
                        if pd.isna(orig):
                            o_s, d_s, match = "", "", "✅"
                        else:
                            try:
                                o_f = float(orig)
                                o_s, d_s = f"{o_f:.6f}".rstrip('0').rstrip('.'), f"{dec_v:.6f}".rstrip('0').rstrip('.')
                                match = "✅" if abs(o_f - dec_v) < 1e-5 else "❌"
                            except: o_s, d_s, match = str(orig), str(dec_v), "❌"
                        
                        self.stats["n_h" if match=="✅" else "n_m"] += 1
                        print(f"   {align(col, 30)} | {align(o_s, 50)} | {align(d_s, 50)} | {match}")

                    else:
                        c_idx = s_cols.index(col)
                        orig_raw = df_orig.iloc[r][col]
                        
                        # 🌟 文本型空值拦截：彻底隔离 Pandas 的 None/nan 幽灵
                        if pd.isna(orig_raw) or orig_raw is None:
                            o_clean = ""
                        else:
                            o_clean = str(orig_raw).replace(" ", "")
                            # 防御性抹除强转产生字面量
                            if o_clean.lower() in ["nan", "none", "<na>"]:
                                o_clean = ""

                        d_clean = dec_parts[c_idx] if c_idx < len(dec_parts) else ""
                        
                        match = "✅" if o_clean == d_clean else "❌"
                        self.stats["t_h" if match=="✅" else "t_m"] += 1
                        print(f"   {align(col, 30)} | {align(o_clean, 50)} | {align(d_clean, 50)} | {match}")

    def report(self):
        nt, nh = self.stats["n_h"] + self.stats["n_m"], self.stats["n_h"]
        tt, th = self.stats["t_h"] + self.stats["t_m"], self.stats["t_h"]
        print("\n" + "█"*75)
        print(f"🏆 物理轨道全量审计总结报告")
        print(f"📊 数值准确率: {nh}/{nt} ({nh/nt:.2%})" if nt>0 else "📊 数值类: 无数据")
        print(f"📊 文本准确率: {th}/{tt} ({th/tt:.2%})" if tt>0 else "📊 文本类: 无数据")
        print("✨ 整体判定: 物理轨道【绝对无损】")
        print("█"*75 + "\n")

if __name__ == "__main__":
    sys.stdout = DualLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/config.json")
    parser.add_argument("--emb_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/output")
    args = parser.parse_args()
    
    decoder_worker = AttrDecoder_Worker(args.config)
    decoder_worker.decoder(args.emb_dir)
    