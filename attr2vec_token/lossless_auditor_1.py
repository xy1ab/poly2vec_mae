## 增加了对不同编码格式的兼容

import sys
import os
import glob
import torch
import argparse
import unicodedata
import warnings
import json
import re
import numpy as np
import pandas as pd
import pyogrio
from tokenizers import Tokenizer
from config import ModelConfig

# 复用工程里的安全加载与编码逻辑
from data_builder import load_csv_safely
from data_loader import process_dataframe

warnings.filterwarnings('ignore')

class DualLogger:
    """双向日志：同时输出到屏幕和 lossless_audit_report.log"""
    def __init__(self, filename="/mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/output/lossless_audit_report.log"):
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

class LosslessAuditor:
    def __init__(self, config_path, schema_path):
        print("\n⏳ [Auditor] 正在初始化全量自动化无损审计引擎 (精准狙击版)...")
        self.config = ModelConfig()
        self.config.load(config_path)
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema_registry = json.load(f)
            
        self.tokenizer = Tokenizer.from_file(self.config.tokenizer_path)
        
        self.global_stats = {
            "n_h": 0, "n_m": 0,
            "t_h": 0, "t_m": 0,
            "row_h": 0, "row_m": 0 
        }
        self.layer_errors = {}
        print("✅ [Auditor] 引擎就绪！")

    def _decode_tensor_to_dict(self, input_ids, layer_name):
        if layer_name not in self.schema_registry:
            raise ValueError(f"图层 {layer_name} 不在 schema 注册表中")
            
        info = self.schema_registry[layer_name]
        
        str_cols = info.get('str_cols', [])
        num_cols = info.get('num_cols', [])
        if not str_cols and not num_cols and 'fields' in info:
            for col, dtype in info['fields'].items():
                if 'float' in dtype.lower() or 'int' in dtype.lower(): num_cols.append(col)
                else: str_cols.append(col)
                
        meta_str = f"[TABLE] {layer_name} [DATA] "
        offset = len(self.tokenizer.encode(meta_str).ids)
        
        N = input_ids.shape[0]
        layer_tensor = torch.tensor(input_ids, dtype=torch.float32)
        
        num_results = {}
        for f_idx, col_name in enumerate(num_cols):
            col_start = offset + f_idx * 3
            chunk_3 = layer_tensor[:, col_start:col_start+3]
            p = torch.round(chunk_3).double()
            real_vals = p[:, 0] + (p[:, 1]/10000.0) + (p[:, 2]/100000000.0)
            num_results[col_name] = real_vals.numpy()
            
        text_start = offset + len(num_cols) * 3
        text_tensor = layer_tensor[:, text_start:]
        text_lists = text_tensor.long().tolist()
        raw_texts = self.tokenizer.decode_batch(text_lists, skip_special_tokens=False)
        
        decoded_rows = []
        for i in range(N):
            row_dict = {}
            for col_name in num_cols:
                row_dict[col_name] = round(float(num_results.get(col_name, np.zeros(N))[i]), 6)
                
            raw_t = raw_texts[i].replace("[PAD]", "").replace("[CLS]", "")
            text_vals = raw_t.split("[SEP]")
            
            for s_idx, col_name in enumerate(str_cols):
                if s_idx < len(text_vals):
                    clean_str = text_vals[s_idx].replace("Ġ", "").replace("\u2581", "")
                    clean_str = re.sub(r'\s+', '', clean_str)
                    row_dict[col_name] = clean_str
                else:
                    row_dict[col_name] = ""
                    
            decoded_rows.append(row_dict)
            
        return decoded_rows, num_cols, str_cols

    def audit_dataframe(self, df, layer_name, file_path):
        if df is None or len(df) == 0:
            return
            
        print(f"\n🔍 正在审计图层: {layer_name} (来源: {os.path.basename(file_path)})")
        
        result = process_dataframe(df, layer_name, self.tokenizer, self.config, self.schema_registry, file_path)
        if result[0] is None: return
        input_ids, _ = result  
            
        decoded_rows, num_cols, str_cols = self._decode_tensor_to_dict(input_ids, layer_name)
        
        N = len(df)
        display_limit = 3 
        layer_has_error = False
        file_basename = os.path.basename(file_path)
        
        for i in range(N):
            row_all_match = True
            orig_row = df.iloc[i]
            dec_row = decoded_rows[i]
            
            if i < display_limit:
                print("-" * 60)
                print(f"🔹 行号 {i}")
            
            for col in num_cols:
                orig_val = orig_row[col]
                dec_val = dec_row.get(col, 0.0)
                if pd.isna(orig_val): orig_val = 0.0
                is_match = abs(float(orig_val) - float(dec_val)) <= 1e-6
                
                if is_match:
                    self.global_stats["n_h"] += 1
                else:
                    self.global_stats["n_m"] += 1
                    row_all_match = False
                    layer_has_error = True
                    self.layer_errors.setdefault(layer_name, set()).add(f"数值失真: {col}")
                    # 🌟 狙击探针：遇到数值错误立刻报警
                    print(f"🚨 [精准捕获] 文件: {file_basename} | 行号: {i} | 字段: {col} | 原值: '{orig_val}' -> 解码: '{dec_val}'")
                    
                if i < display_limit:
                    status = "✅" if is_match else "❌"
                    orig_str = align(f"原: {orig_val:.6f}", 18)
                    dec_str = align(f"解: {dec_val:.6f}", 18)
                    print(f"   {status} 数值 | {align(col, 15)} | {orig_str} | {dec_str}")
            
            for col in str_cols:
                orig_val = str(orig_row[col]) if pd.notna(orig_row[col]) else ""
                orig_val_clean = re.sub(r'\s+', '', orig_val) 
                dec_val_clean = str(dec_row.get(col, ""))
                
                is_match = (orig_val_clean == dec_val_clean)
                
                if is_match:
                    self.global_stats["t_h"] += 1
                else:
                    self.global_stats["t_m"] += 1
                    row_all_match = False
                    layer_has_error = True
                    self.layer_errors.setdefault(layer_name, set()).add(f"文本失真: {col}")
                    
                    # 🌟🌟🌟 狙击探针：无论行号多深，只要文本错误立刻当场报警！
                    print(f"🚨 [精准捕获] 文件: {file_basename} | 局部行号: {i} | 字段: {col} | 原值: '{orig_val_clean}' -> 解码: '{dec_val_clean}'")
                    
                if i < display_limit:
                    status = "✅" if is_match else "❌"
                    orig_str = align(f"原: {orig_val_clean}", 20)
                    dec_str = align(f"解: {dec_val_clean}", 20)
                    print(f"   {status} 文本 | {align(col, 15)} | {orig_str} | {dec_str}")
            
            if row_all_match:
                self.global_stats["row_h"] += 1
            else:
                self.global_stats["row_m"] += 1

        print("-" * 60)
        if layer_has_error:
            print(f"⚠️ 该图层在后台全量审计中发现异常！影响字段: {', '.join(self.layer_errors[layer_name])}")
        else:
            print(f"🎉 该图层全部 {N} 行数据后台静默审计通过，100% 无损！")


    def run_all(self):
        raw_data_dir = self.config.data_dir
        search_pattern = os.path.join(raw_data_dir, "**", "*")
        
        files_to_process = []
        for file_path in sorted(glob.glob(search_pattern, recursive=True)):
            if file_path.lower().endswith(('.csv', '.gdb', '.shp')):
                files_to_process.append(file_path)
                
        print(f"\n📂 自动嗅探到 {len(files_to_process)} 个底层数据源，开始进行穿透式回归测试...")

        for i, file_path in enumerate(files_to_process):
            print(f"\n" + "="*70)
            print(f"📦 进度 [{i+1}/{len(files_to_process)}] 正在加载物理文件: {file_path}")
            
            if file_path.lower().endswith('.csv'):
                layer_name = os.path.basename(file_path).split('.')[0]
                df = load_csv_safely(file_path)
                self.audit_dataframe(df, layer_name, file_path)
                
            elif file_path.lower().endswith('.gdb'):
                try:
                    layers = pyogrio.list_layers(file_path)
                    for layer_name, geom_type in layers:
                        try:
                            df = pyogrio.read_dataframe(file_path, layer=layer_name, read_geometry=False, encoding='utf-8')
                        except:
                            df = pyogrio.read_dataframe(file_path, layer=layer_name, read_geometry=False, encoding='gb18030')
                        self.audit_dataframe(df, layer_name, file_path)
                except Exception as e:
                    print(f"❌ 读取 GDB 失败 [{file_path}]: {e}")
                    
            elif file_path.lower().endswith('.shp'):
                try:
                    layer_name = os.path.basename(file_path).split('.')[0]
                    try:
                        df = pyogrio.read_dataframe(file_path, read_geometry=False, encoding='utf-8')
                    except:
                        df = pyogrio.read_dataframe(file_path, read_geometry=False, encoding='gb18030')
                    self.audit_dataframe(df, layer_name, file_path)
                except Exception as e:
                    print(f"❌ 读取 SHP 失败 [{file_path}]: {e}")

    def report(self):
        nt, nh = self.global_stats["n_h"] + self.global_stats["n_m"], self.global_stats["n_h"]
        tt, th = self.global_stats["t_h"] + self.global_stats["t_m"], self.global_stats["t_h"]
        rt, rh = self.global_stats["row_h"] + self.global_stats["row_m"], self.global_stats["row_h"]
        
        print("\n" + "█"*75)
        print(f"🏆 大模型底座 [全量物理数据源] 端到端闭环审计报告")
        print(f"📊 累计审计单元格数: {(nt + tt):,}")
        print(f"📊 数值编解码精度: {nh:,}/{nt:,} ({nh/nt:.4%} 无损)" if nt>0 else "📊 数值类: 无数据")
        print(f"📊 文本编解码精度: {th:,}/{tt:,} ({th/tt:.4%} 无损)" if tt>0 else "📊 文本类: 无数据")
        print(f"🎯 全局行级一致率: {rh:,}/{rt:,} ({rh/rt:.4%} 无损)" if rt>0 else "🎯 行解析: 无数据")
        
        if self.layer_errors:
            print("\n⚠️ 异常警告记录:")
            for layer, errs in self.layer_errors.items():
                print(f"   - {layer}: {', '.join(errs)}")
        print("█"*75 + "\n")


if __name__ == "__main__":
    sys.stdout = DualLogger()
    
    BASE_DIR = "/mnt/data/yqmeng/ZRZYB/NRE_GIT_V8"
    CONFIG_PATH = f"{BASE_DIR}/output/config.json"
    SCHEMA_PATH = f"{BASE_DIR}/output/schema_registry.json"
    
    try:
        auditor = LosslessAuditor(CONFIG_PATH, SCHEMA_PATH)
        auditor.run_all()
        auditor.report()
    except Exception as e:
        print(f"❌ 审计过程发生严重错误: {e}")