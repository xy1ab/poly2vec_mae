import sys, os, torch, argparse, unicodedata, warnings, pyogrio
import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from config import ModelConfig
from models import Attr2Vec
import json
from torch.utils.data import TensorDataset, DataLoader

# 引用全局配置中的常量
from data_builder import MUST_BE_STRING_EXACT, MUST_BE_STRING_SUFFIX

warnings.filterwarnings('ignore')

class DualLogger:
    """双向日志：同时输出到屏幕和 evaluator_audit.log"""
    def __init__(self, filename="evaluator_audit.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, m):
        self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

def align(text, width):
    """处理中文字符对齐的辅助函数"""
    text = str(text).replace("\n", "").replace("\r", "")
    dw = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)
    return text + ' ' * max(0, width - dw)

class AttrDecoder_Worker:
    def __init__(self, args):
        self.config = ModelConfig()
        self.config.load(args.config_path)
        self.tokenizer = Tokenizer.from_file(self.config.tokenizer_path)
        
        # 组长的高级探针：定位符号 ID
        self.table_id = self.tokenizer.token_to_id("[TABLE]")
        self.data_id = self.tokenizer.token_to_id("[DATA]")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载固化的物理权重密钥
        self.model = Attr2Vec(self.config).to(self.device)
        self.model.load_state_dict(torch.load(args.model_path, weights_only=False, map_location=self.device))
        self.model.eval()
        
        # 对账全局统计器
        self.stats = {"n_h": 0, "n_m": 0, "t_h": 0, "t_m": 0, "row_h": 0, "row_m": 0}

    def decode_num(self, p):
        """三段式 BF32 数值重组逻辑"""
        p = torch.round(p).double()
        return p[0] + (p[1]/10000.0) + (p[2]/100000000.0)

    def decoder(self, emb_dir, max_rows=None, log_layers=None):
        if log_layers is None: log_layers = []
        
        schema_path = os.path.join(emb_dir, "schema_registry.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_registry = json.load(f)
            
        full_emb = torch.load(os.path.join(emb_dir, "attr_emb.pt"), weights_only=False, map_location=self.device)
        v_size = self.tokenizer.get_vocab_size()
        
        current_offset = 0 
        
        for layer_name, info in schema_registry.items():
            # 为当前图层准备纯净账本
            layer_decoded_results = []
            
            raw_path = info.get('raw_path')
            true_row_count = info.get('row_count', 0)
            
            # 兼容南湖平台 CPFS 挂载的鲁棒性检查
            if true_row_count == 0 or not raw_path:
                print(f"⚠️ 跳过图层 【{layer_name}】: 注册表信息不完整")
                current_offset += true_row_count
                continue

            audit_rows = min(max_rows, true_row_count) if max_rows else true_row_count
            print_details = (layer_name in log_layers) or (len(log_layers) == 0)
            
            # 1. 加载原始真值进行对账
            try:
                if raw_path.lower().endswith('.csv'):
                    cols = pd.read_csv(raw_path, nrows=0).columns.tolist()
                    dtype_map = {c: str for c in cols if c.upper() in MUST_BE_STRING_EXACT or c.endswith(MUST_BE_STRING_SUFFIX)}
                    df_orig = pd.read_csv(raw_path, dtype=dtype_map, nrows=audit_rows)
                else:
                    df_orig = pyogrio.read_dataframe(raw_path, layer=layer_name, max_features=audit_rows, read_geometry=False)
            except Exception as e:
                print(f"\n❌ 读取原始文件失败 【{layer_name}】: {e}")
                current_offset += true_row_count
                continue

            # 2. 张量物理切片
            layer_emb = full_emb[current_offset : current_offset + audit_rows]
            current_offset += true_row_count
            
            if print_details:
                print(f"\n{'='*150}\n📊 图层审计中: 【{layer_name}】 | 数据源: {os.path.basename(raw_path)}\n{'='*150}")
            else:
                print(f"🔄 正在解密图层: 【{layer_name}】 ({audit_rows} 行)...")

            # 3. 批量穿越 INN 反向网络
            dec_vecs = []
            dataset = TensorDataset(layer_emb)
            loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
            with torch.no_grad():
                for (batch,) in loader:
                    dec_vecs.append(self.model.inn_core(batch.to(self.device), reverse=True).cpu())
            dec_vec = torch.cat(dec_vecs, dim=0)

            # 4. 动态探针寻找 [DATA] 锚点
            ids_round = torch.round(dec_vec).long()
            data_indices = (ids_round == self.data_id).int().argmax(dim=-1).tolist()
            
            n_cols, s_cols = info['num_cols'], info['str_cols']
            num_w = len(n_cols) * 3
            
            # 5. 逐行解算与高精度对账
            for r in range(audit_rows):
                should_print_row = print_details and (r < 3)
                
                if should_print_row:
                    print(f"\n🔹 Row {r+1}:")
                    print(f"   {align('字段名称 (Field)', 30)} | {align('原始真值', 50)} | {align('底座解码值', 50)} | 判定")
                    print("   " + "-"*145)
                
                data_pos = data_indices[r]
                row_data_ids = dec_vec[r, data_pos+1:]
                
                dec_n = row_data_ids[:num_w]
                dec_t = row_data_ids[num_w : num_w + self.config.max_seq_len]
                
                # 文本 Token 还原清洗
                ids = torch.clamp(torch.round(dec_t).to(torch.int64), 0, v_size - 1).tolist()
                raw_str = self.tokenizer.decode(ids, skip_special_tokens=False)
                clean_line = raw_str.replace("[PAD]", "").replace("[CLS]", "").replace("[UNK]", "").strip()
                dec_parts = [ "".join(p.split()) for p in clean_line.split("[SEP]") ]
                
                row_is_correct = True
                row_export_dict = {} # ✅ 移除 _layer_name，保证 CSV 结构纯净

                # 数值高精度比对
                for i, col in enumerate(n_cols):
                    orig = df_orig.iloc[r][col]
                    p = dec_n[i*3 : i*3+3]
                    dec_v = self.decode_num(p).item()
                    row_export_dict[col] = dec_v
                    
                    o_s = f"{float(orig):.8f}".rstrip('0').rstrip('.') if not pd.isna(orig) else ""
                    d_s = f"{dec_v:.8f}".rstrip('0').rstrip('.') if o_s != "" else ""
                    
                    if pd.isna(orig) or o_s == "":
                        match = "✅"
                    else:
                        orig_f = float(orig)
                        # 10^-8 绝对误差 + 10^-12 相对误差
                        atol, rtol = 1e-8, 1e-12
                        if abs(orig_f - dec_v) <= (atol + rtol * abs(orig_f)):
                            match = "✅"
                        else:
                            match = "❌"
                            
                    if match == "❌": row_is_correct = False
                    self.stats["n_h" if match=="✅" else "n_m"] += 1
                    if should_print_row:
                        print(f"   {align(col, 30)} | {align(o_s, 50)} | {align(d_s, 50)} | {match}")

                # 文本清洗匹配
                for i, col in enumerate(s_cols):
                    orig_raw = df_orig.iloc[r][col]
                    o_clean = "".join(str(orig_raw).split()) if not pd.isna(orig_raw) else ""
                    if o_clean.lower() in ["nan", "none", "<na>"]: o_clean = ""
                    
                    d_clean = dec_parts[i] if i < len(dec_parts) else ""
                    row_export_dict[col] = d_clean
                    
                    match = "✅" if o_clean == d_clean else "❌"
                    if match == "❌": row_is_correct = False
                    self.stats["t_h" if match=="✅" else "t_m"] += 1
                    if should_print_row:
                        print(f"   {align(col, 30)} | {align(o_clean, 50)} | {align(d_clean, 50)} | {match}")
                
                self.stats["row_h" if row_is_correct else "row_m"] += 1
                layer_decoded_results.append(row_export_dict)

            # 每个图层跑完，立刻独立存盘
            if layer_decoded_results:
                layer_df = pd.DataFrame(layer_decoded_results)
                out_csv = os.path.join(emb_dir, f"{layer_name}_decoded.csv")
                layer_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
                if print_details:
                    print(f"💾 纯净明文已存至: {os.path.basename(out_csv)}")

    def report(self):
        """输出最终审计报告"""
        nt, nh = self.stats["n_h"] + self.stats["n_m"], self.stats["n_h"]
        tt, th = self.stats["t_h"] + self.stats["t_m"], self.stats["t_h"]
        rt, rh = self.stats["row_h"] + self.stats["row_m"], self.stats["row_h"]
        
        print("\n" + "█"*75)
        print(f"🏆 大模型底座 [物理轨道] 全量审计报告")
        print(f"📊 数值准确率: {nh}/{nt} ({nh/nt:.2%})" if nt>0 else "📊 数值类: 无数据")
        print(f"📊 文本准确率: {th}/{tt} ({th/tt:.2%})" if tt>0 else "📊 文本类: 无数据")
        print(f"🎯 行解析精度: {rh}/{rt} ({rh/rt:.2%})" if rt>0 else "🎯 行解析: 无数据")
        print("█"*75 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--emb_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--log_layers", nargs='*', default=[])
    args = parser.parse_args()

    sys.stdout = DualLogger()
    worker = AttrDecoder_Worker(args)
    worker.decoder(args.emb_dir, max_rows=args.max_rows, log_layers=args.log_layers)
    worker.report()