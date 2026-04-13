import geopandas as gpd
import pandas as pd
import numpy as np
import pyogrio
import joblib
import os
import json
import warnings
from tokenizers import Tokenizer
import glob

warnings.filterwarnings('ignore')

# ==========================================
# 🛡️ 核心基建配置：绝对文本列白名单
# ==========================================
MUST_BE_STRING_EXACT = ['BSM', 'YSDM', 'PAC', 'OBJECTID']  
MUST_BE_STRING_SUFFIX = ('代码', '编码', '编号', 'ID')      

def load_csv_safely(csv_path):
    """工业级安全读取 CSV：利用白名单阻断 Pandas 的自动数值推断"""
    try:
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        dtype_mapping = {}
        for col in columns:
            if col.upper() in MUST_BE_STRING_EXACT or col.endswith(MUST_BE_STRING_SUFFIX):
                dtype_mapping[col] = str
                
        df = pd.read_csv(csv_path, dtype=dtype_mapping, low_memory=False)
        return df
    except Exception as e:
        print(f"❌ 读取 CSV 失败 [{csv_path}]: {e}")
        return None

class NRE_DataPump:
    def __init__(self, vocab_path, tokenizer_path, max_seq_len=64):
        self.vocab_path = vocab_path
        self.max_seq_len = max_seq_len
        self.vocab = {}
        
        # 🌟 核心修改 1：加载真正的 BPE 智能分词器
        if os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            print("⚠️ 警告：未找到 zrzy_tokenizer.json，请先运行 train_tokenizer.py！")
            self.tokenizer = None
            
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)

    def _route_columns(self, df):
        cont_cols, word_cols, char_cols = [], [], []
        FORCE_CHAR_KEYWORDS = ['NAME', 'NAME_CH', 'MC', 'BZ', 'REMARK', 'NOTE', 'DESC', 'ADDR', 'SM', 'MS']

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['geometry', 'shape'] or col_lower.endswith('id') or col_lower.endswith('uuid'):
                continue
            if any(key in col.upper() for key in FORCE_CHAR_KEYWORDS):
                char_cols.append(col)
                continue
            if col in self.vocab and col != "__SHARED_CHARS__":
                word_cols.append(col)
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                cont_cols.append(col)
            else:
                char_cols.append(col)
        return cont_cols, word_cols, char_cols

    def _process_single_dataframe(self, df):
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        cont_cols, word_cols, char_cols = self._route_columns(df)
        
        # =========================================================
        # 🌟 核心升级：动态嗅探与计算绝对安全文本长度
        # =========================================================
        TRUTH_DIM = 256 # 业务硬性底线
        used_dims = len(cont_cols) * 3 + len(word_cols) * 1
        remaining_dims = TRUTH_DIM - used_dims
        
        dynamic_max_len = 0
        if len(char_cols) > 0:
            # 1. 计算理论安全上限
            safe_limit = remaining_dims // len(char_cols)
            
            if safe_limit <= 0:
                print(f"      ⚠️ 警告: 基础属性已耗尽 {used_dims} 维！毫无文本空间，强制清空文本。")
                actual_max = 0
            else:
                # 2. 扫描表中真实的最长 Token 数量
                actual_max = 0
                for col in char_cols:
                    if self.tokenizer:
                        # 🌟 核心升级：不再数汉字个数，而是用分词器编码后数 Token 的个数
                        lens = df[col].dropna().astype(str).map(lambda x: len(self.tokenizer.encode(x.strip()).ids))
                        col_max = lens.max() if len(lens) > 0 else 0
                    else:
                        col_max = 0
                        
                    if col_max > actual_max:
                        actual_max = int(col_max)
                        
            # 3. 最终采纳长度 = Min(真实最长字数, 物理安全上限, 全局硬上限)
            dynamic_max_len = min(actual_max, safe_limit, self.max_seq_len)
            
            print(f"      📏 动态嗅探 -> 安全上限:{safe_limit} | 真实最长:{actual_max} | 全局硬上限:{self.max_seq_len} => 最终采纳:{dynamic_max_len}字")
        # =========================================================

        if cont_cols:
            df[cont_cols] = df[cont_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            raw_64 = df[cont_cols].values.astype(np.float64)
            raw_64 = np.nan_to_num(raw_64, nan=0.0, posinf=0.0, neginf=0.0)
            
            cont_mantissa, cont_exponent = np.frexp(raw_64)
            cont_int = cont_exponent.astype(np.float32)
            cont_frac_hi = cont_mantissa.astype(np.float32)
            cont_frac_lo = (cont_mantissa - cont_frac_hi.astype(np.float64)).astype(np.float32)
            
            std = np.std(raw_64, axis=0)
            std[std == 0] = 1.0
            cont_norm = ((raw_64 - np.mean(raw_64, axis=0)) / std).astype(np.float32)
            cont_norm = np.nan_to_num(cont_norm, nan=0.0)
        else:
            cont_int = cont_frac_hi = cont_frac_lo = cont_norm = np.zeros((len(df), 0), dtype=np.float32)

        word_list = []
        for col in word_cols:
            col_vocab = self.vocab.get(col, {})
            
            # 🛡️ 终极清洗装甲：去空值、去首尾空格、去幽灵 .0
            def safe_map(val):
                if pd.isna(val): return 0
                s = str(val).strip()
                if s.endswith('.0'): s = s[:-2]
                return col_vocab.get(s, 0)
                
            encoded = df[col].map(safe_map).fillna(0).values
            word_list.append(encoded)
            
        word_data = (np.stack(word_list, axis=1).astype(np.float32) / 16384.0) if word_list else np.zeros((len(df), 0), dtype=np.float32)

        char_tensors = []
        for col in char_cols:
            col_chars = []
            for text in df[col].astype(str).fillna(""):
                text = text.strip()
                if text.lower() == 'nan': text = ""
                
                # 🌟 核心修改 2：调用智能分词器进行高维压缩
                if self.tokenizer and text:
                    char_ids = self.tokenizer.encode(text).ids
                else:
                    char_ids = []
                
                if len(char_ids) < dynamic_max_len:
                    char_ids.extend([0] * (dynamic_max_len - len(char_ids)))
                else:
                    char_ids = char_ids[:dynamic_max_len]
                col_chars.append(char_ids)
            char_tensors.append(col_chars)
            
        # 🌟 核心修改 3：将缩放基数从 16384.0 升级为 32768.0 以容纳两万词表
        char_data = (np.array(char_tensors).transpose(1, 0, 2).astype(np.float32) / 32768.0) if char_tensors else np.zeros((len(df), 0, dynamic_max_len), dtype=np.float32)

        return {
            'cont_int': cont_int, 'cont_frac_hi': cont_frac_hi, 'cont_frac_lo': cont_frac_lo, 'cont_norm': cont_norm,
            'word_data': word_data, 'char_data': char_data,
            # 🌟 核心：将该图层专属的 dynamic_max_len 写入 meta，让解码器 evaluator 知道长度
            'meta': {'cont_cols': cont_cols, 'word_cols': word_cols, 'char_cols': char_cols, 'max_seq_len': dynamic_max_len, 'total_samples': len(df)}
        }

    def build_cache(self, file_path, cache_path="data_cache.pt"):
        print(f"🚀 正在解析 {file_path} 并执行【高低位双轨切割】全量无损张量化...")
        all_layers_data = {}
        
        ext = os.path.splitext(file_path)[-1].lower()
        tabular_exts = ['.csv', '.txt', '.xlsx', '.xls', '.parquet']

        if ext in tabular_exts:
            print(f"📄 识别为无空间属性表格文件 ({ext})，开始提取属性...")
            if ext in ['.csv', '.txt']:
                # 🌟 调用安全读取引擎！
                df = load_csv_safely(file_path)
                if df is None:
                    return all_layers_data # 阻断错误
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif ext == '.parquet':
                df = pd.read_parquet(file_path)
                
            if df is not None:
                all_layers_data["TABULAR_DEFAULT"] = self._process_single_dataframe(df)
            
        else:
            try:
                layers = pyogrio.list_layers(file_path)
                print(f"🌍 识别为空间矢量文件，智能探测到 {len(layers)} 个图层...")
                for layer_name, geom_type in layers:
                    print(f"   -> 正在抽取图层: [{layer_name}]")
                    try:
                        gdf = gpd.read_file(file_path, layer=layer_name, engine="pyogrio")
                        df = pd.DataFrame(gdf).drop(columns=['geometry'], errors='ignore')
                        if len(df) == 0:
                            print(f"   ⚠️ 图层 [{layer_name}] 为空或无有效台账，已跳过。")
                            continue
                        all_layers_data[layer_name] = self._process_single_dataframe(df)
                    except Exception as e:
                        print(f"   ❌ 解析图层 [{layer_name}] 失败，跳过该层。错误: {e}")
            except Exception as e:
                raise ValueError(f"❌ 严重错误：无法解析文件 {file_path}。核心报错: {e}")
                    
        joblib.dump(all_layers_data, cache_path)
        print(f"✅ 编译完成！共收录 {len(all_layers_data)} 个有效图层，张量缓存已保存至 {cache_path}")
        return all_layers_data

import argparse
def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for batch forward export."""
    parser = argparse.ArgumentParser(
        description="Batch forward triangulated polygon shards into embeddings and MAE frequency maps."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing triangulated shard `.pt` files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing triangulated shard `.pt` files.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Directory containing triangulated shard `.pt` files.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Directory containing triangulated shard `.pt` files.")
    
    return parser

if __name__ == "__main__":
    print("=== data_loader.py 全自动张量化流水线启动 ===")

    
    args = build_arg_parser().parse_args()
    pump = NRE_DataPump(args.vocab_path, args.tokenizer_path)
    RAW_DATA_DIR = args.data_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        
    DATA_SOURCES = []
    for file_path in glob.glob(os.path.join(RAW_DATA_DIR, "*")):
        if file_path.endswith('.csv') or file_path.endswith('.gdb'):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            cache_name = os.path.join(output_dir, f"cache_{base_name}.joblib")
            DATA_SOURCES.append({"file": file_path, "cache": cache_name})
            
    print(f"📦 共规划了 {len(DATA_SOURCES)} 个数据源的缓存生成任务。")
    
    for source in DATA_SOURCES:
        file_path = source["file"]
        cache_path = source["cache"]
        
        if os.path.exists(cache_path):
            print(f"⏩ 发现现有缓存 [{cache_path}]，为节省时间已跳过。如需重构请先删除原缓存。")
            continue
            
        print(f"\n📂 正在切入数据源: [{file_path}]")
        pump.build_cache(file_path, cache_path)