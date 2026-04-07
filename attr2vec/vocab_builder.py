import pandas as pd
import geopandas as gpd
import fiona
import json
import os
import numpy as np
import glob

# ==========================================
# 第一层：通用数据适配器 (Data Connector)
# ==========================================
class DataConnector:
    """负责将异构数据源统一清洗为标准 DataFrame 生成器，保护内存"""
    @staticmethod
    def stream_data(source_config):
        source_type = source_config.get('type').lower()
        path_or_conn = source_config.get('path')

        if source_type == 'csv':
            print(f"📡 [连接器] 接入 CSV 单表数据: {path_or_conn}")
            try:
                yield pd.read_csv(path_or_conn, low_memory=False)
            except Exception as e:
                print(f"❌ 读取 CSV 失败: {e}")

        elif source_type == 'gdb':
            print(f"📡 [连接器] 接入 GDB 空间数据库: {path_or_conn}")
            try:
                layers = fiona.listlayers(path_or_conn)
                for i, layer_name in enumerate(layers):
                    print(f"  -> 抽取图层 [{i+1}/{len(layers)}]: {layer_name}")
                    yield gpd.read_file(path_or_conn, layer=layer_name, ignore_geometry=True)
            except Exception as e:
                print(f"❌ 读取 GDB 失败: {e}")
        else:
            print(f"❌ 未知的数据源类型: {source_type}")

# ==========================================
# 第二层：自适应层级密码本构建器 (Schema-Free + 增量版)
# ==========================================
class NRE_UniversalVocabBuilder:
    def __init__(self, config):
        self.config = config
        self.vocab_path = config.get('vocab_path', 'global_vocab_auto.json')
        
        if os.path.exists(self.vocab_path):
            print(f"🔄 发现已有字典 {self.vocab_path}，启动【增量追加模式】...")
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            
            max_id = 1
            for namespace, tokens in self.vocab.items():
                if tokens: 
                    max_id = max(max_id, max(tokens.values()))
            self.idx = max_id + 1
            print(f"   -> 历史字典解析完毕，新词条将从 ID={self.idx} 开始分配。\n")
        else:
            print("🆕 未发现旧字典，启动【全新构建模式】...\n")
            self.vocab = {
                "__SHARED_CHARS__": {"<PAD>": 0, "<UNK>": 1}
            }
            self.idx = 2

    def _auto_profiling(self, df):
        word_cols = []
        char_cols = []
        FORCE_CHAR_KEYWORDS = [
            'NAME', 'NAME_CH', 'MC', 'BZ', 'REMARK', 'NOTE', 'DESC', 
            'ADDR', 'PINYIN', 'LABEL', 'TEXT', 'ZJ', 'SM', 'MS'
        ]

        for col in df.columns:
            if df[col].isnull().all() or col.lower() in ['geometry', 'shape', 'fid']:
                continue
            if pd.api.types.is_float_dtype(df[col]):
                continue
                
            valid_data = df[col].dropna()
            if len(valid_data) == 0: continue
            
            if any(key in col.upper() for key in FORCE_CHAR_KEYWORDS):
                char_cols.append(col)
                continue 

            total_count = len(valid_data)
            unique_count = valid_data.nunique()
            unique_ratio = unique_count / total_count
            str_data = valid_data.astype(str)
            max_len = str_data.str.len().max()
            
            if pd.api.types.is_integer_dtype(df[col]) and unique_ratio > 0.99:
                continue

            if unique_count < 100 or unique_ratio < 0.05:
                if max_len <= 15:
                    word_cols.append(col)
                else:
                    char_cols.append(col)
            else:
                if not pd.api.types.is_numeric_dtype(df[col]): 
                    char_cols.append(col)
                    
        return word_cols, char_cols

    def fit_dataframe(self, df):
        word_cols, char_cols = self._auto_profiling(df)
        print(f"      [雷达命中] 整体词汇字段 ({len(word_cols)}个): {word_cols[:5]}...")
        if char_cols:
            print(f"      [雷达命中] 不定长文本字段 ({len(char_cols)}个): {char_cols[:5]}...")

        for col in word_cols:
            if col not in self.vocab:
                self.vocab[col] = {} 
                
            unique_vals = df[col].dropna().astype(str).unique()
            for val in unique_vals:
                if val not in self.vocab[col]:
                    self.vocab[col][val] = self.idx
                    self.idx += 1

        for col in char_cols:
            text_data = df[col].dropna().astype(str)
            for text in text_data:
                for char in text:
                    if char not in self.vocab["__SHARED_CHARS__"]:
                        self.vocab["__SHARED_CHARS__"][char] = self.idx
                        self.idx += 1

    def save_vocab(self):
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        print(f"\n💾 全局大字典已固化至: {self.vocab_path}")

# ==========================================
# 统一调度流 (实战入口：一键通吃所有数据)
# ==========================================
if __name__ == "__main__":
    vocab_config = {'vocab_path': 'global_vocab_auto.json'}

    RAW_DATA_DIR = "./raw_data"
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"📁 已自动创建数据存放目录: {RAW_DATA_DIR}，请将GDB和CSV放入此目录！")
    
    data_sources = []
    for file_path in glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")):
        data_sources.append({'type': 'csv', 'path': file_path})
    for file_path in glob.glob(os.path.join(RAW_DATA_DIR, "*.gdb")):
        data_sources.append({'type': 'gdb', 'path': file_path})
        
    print(f"🔍 扫描到 {len(data_sources)} 个数据源，准备构建/更新字典...")

    builder = NRE_UniversalVocabBuilder(vocab_config)
    for source in data_sources:
        print(f"\n==============================================")
        print(f"🚀 开始执行新任务流: {source['path']}")
        for df_chunk in DataConnector.stream_data(source):
            builder.fit_dataframe(df_chunk)
            
    builder.save_vocab()