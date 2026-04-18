import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import pyogrio
import torch
from tokenizers import Tokenizer
from config import ModelConfig
from data_builder import load_csv_safely
import warnings
warnings.filterwarnings('ignore')


def float64_to_three_float32(arr):
    arr = np.nan_to_num(arr, nan=0.0) 
    int_part = np.trunc(arr).astype(np.float32)
    frac = arr - int_part
    frac_hi = np.trunc(frac * 10000).astype(np.float32)
    frac_lo = np.trunc((frac * 10000 - frac_hi) * 10000).astype(np.float32)
    return int_part, frac_hi, frac_lo

import numpy as np

def three_float32_to_float64(p):
    """
    [逆向转换逻辑] 将 3 个 Float32 重新组合为 1 个 Float64
    p[0] int_part: 整数部分
    p[1] frac_hi: 前 4 位小数 (乘以 1e-4)
    p[2] frac_lo: 后 4 位小数 (乘以 1e-8)
    """
    p = torch.round(p).double()
    return p[0] + (p[1]/10000.0) + (p[2]/100000000.0)


# 🌟 注意：函数签名增加了 file_path 参数
def process_dataframe(df, layer_name, tokenizer, config, schema_registry, file_path):
    if len(df) == 0:
        return None
        
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    str_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ['geometry', 'shape']]
    
    metadata_prefix = f"[TABLE] {layer_name} [COLUMN] {' [SEP] '.join(str_cols)} [DATA] "
    metadata_ids = tokenizer.encode(metadata_prefix).ids
    
    # 🌟 登记元数据注册表，顺手把文件的绝对坐标写进去
    if layer_name not in schema_registry:
        schema_registry[layer_name] = {
            "num_cols": num_cols,
            "str_cols": str_cols,
            "num_count": len(num_cols),
            "str_count": len(str_cols),
            "metadata_prefix": metadata_prefix,
            "metadata_token_ids": metadata_ids,
            "metadata_token_len": len(metadata_ids),
            "raw_path": file_path  # 👈 记录出生地坐标
        }

    metadata_short = f"[TABLE] {layer_name} [DATA] "
    metadata_short_ids = tokenizer.encode(metadata_short).ids
    
    N = len(df)
    if len(num_cols) > 0:
        num_data = df[num_cols].values.astype(np.float64)
        int_p, frac_h, frac_l = float64_to_three_float32(num_data)
        num_ids = np.empty((N, len(num_cols) * 3), dtype=np.float32)
        for i in range(len(num_cols)):
            num_ids[:, i*3] = int_p[:, i]
            num_ids[:, i*3+1] = frac_h[:, i]
            num_ids[:, i*3+2] = frac_l[:, i]
    else:
        num_ids = np.empty((N, 0), dtype=np.float32)

    text_ids = np.zeros((N, config.max_seq_len), dtype=np.int64)
    if len(str_cols) > 0:
        str_data = df[str_cols].fillna("").astype(str)
        combined_strings = str_data.agg(' [SEP] '.join, axis=1).tolist()
        encoded = tokenizer.encode_batch(combined_strings)
        for i, enc in enumerate(encoded):
            ids = enc.ids
            if len(ids) > config.max_seq_len:
                ids = ids[:config.max_seq_len]
            text_ids[i, :len(ids)] = ids

    input_ids = np.zeros((N, config.truth_dim), dtype=np.float32)
    input_ids[:,:len(metadata_short_ids)] = metadata_short_ids
    num_width = num_ids.shape[1]
    input_ids[:, len(metadata_short_ids):(num_width+len(metadata_short_ids))] = num_ids
    input_ids[:, (num_width+len(metadata_short_ids)) : (num_width+len(metadata_short_ids) + config.max_seq_len)] = text_ids.astype(np.float32)
    
    return input_ids

def build_tensors(config_path):
    print("🚀 启动集群优化版数据张量化流水线 (流式边算边存)...")
    
    config = ModelConfig()
    config.load(config_path)
    raw_data_dir = config.data_dir
    if not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError("❌ 找不到 tokenizer，请先运行 data_builder.py！")
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    
    schema_registry = {}
    CHUNK_SIZE = 100000        
    data_buffer = []                 
    buffer_rows = 0            
    chunk_idx = 0              
    total_processed_rows = 0   

    def flush_buffer_to_disk(force_all=False):
        nonlocal data_buffer, buffer_rows, chunk_idx, total_processed_rows
        while buffer_rows >= CHUNK_SIZE or (force_all and buffer_rows > 0):
            stacked_data = np.vstack(data_buffer)
            rows_to_save = min(CHUNK_SIZE, buffer_rows)
            chunk_data = stacked_data[:rows_to_save]
            out_path = os.path.join(config.output_dir, f"cache_chunk_{chunk_idx}.pt")
            torch.save({"data_ids": chunk_data}, out_path)
            print(f"✅ 流式落盘触发: 成功保存 {out_path} (释放内存: {rows_to_save} 条)")
            chunk_idx += 1
            total_processed_rows += rows_to_save
            if rows_to_save < stacked_data.shape[0]:
                data_buffer = [stacked_data[rows_to_save:]]
                buffer_rows = data_buffer[0].shape[0]
            else:
                data_buffer = []
                buffer_rows = 0

    files_to_process = []
    search_pattern = os.path.join(raw_data_dir, "**", "*")
    for file_path in sorted(glob.glob(search_pattern, recursive=True)):
        if file_path.lower().endswith('.csv') or file_path.lower().endswith('.gdb') or file_path.lower().endswith('.shp'):
            files_to_process.append(file_path)

    for i, file_path in enumerate(files_to_process):
        print(f"🔍 处理进度 [{i+1}/{len(files_to_process)}]: {os.path.basename(file_path)}")
        
        if file_path.lower().endswith('.csv'):
            layer_name = os.path.basename(file_path).split('.')[0]
            df = load_csv_safely(file_path)
            if df is not None:
                # 🌟 注意这里传入了 file_path
                input_ids = process_dataframe(df, layer_name, tokenizer, config, schema_registry, file_path)
                if input_ids is not None:
                    data_buffer.append(input_ids)
                    buffer_rows += input_ids.shape[0]
                    flush_buffer_to_disk(force_all=False)
                    
        elif file_path.lower().endswith('.gdb'):
            try:
                layers = pyogrio.list_layers(file_path)
                for layer_name, geom_type in layers:
                    df = pyogrio.read_dataframe(file_path, layer=layer_name, read_geometry=False)
                    # 🌟 注意这里传入了 file_path
                    input_ids = process_dataframe(df, layer_name, tokenizer, config, schema_registry, file_path)
                    if input_ids is not None:
                        data_buffer.append(input_ids)
                        buffer_rows += input_ids.shape[0]
                        flush_buffer_to_disk(force_all=False)
            except Exception as e:
                print(f"❌ 读取 GDB 失败 [{file_path}]: {e}")
                
        elif file_path.lower().endswith('.shp'):
            try:
                layer_name = os.path.basename(file_path).split('.')[0]
                df = pyogrio.read_dataframe(file_path, read_geometry=False)
                # 🌟 注意这里传入了 file_path
                input_ids = process_dataframe(df, layer_name, tokenizer, config, schema_registry, file_path)
                if input_ids is not None:
                    data_buffer.append(input_ids)
                    buffer_rows += input_ids.shape[0]
                    flush_buffer_to_disk(force_all=False)
            except Exception as e:
                print(f"❌ 读取 SHP 失败 [{file_path}]: {e}")

    flush_buffer_to_disk(force_all=True)

    schema_path = os.path.join(config.output_dir, "schema_registry.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_registry, f, ensure_ascii=False, indent=4)
    print(f"✅ 成功生成带文件坐标的元数据注册表: {schema_path}")
    
    print("\n" + "="*60)
    print(f"🎉 全部集群流式处理完成！共生成 {total_processed_rows} 条标准训练样本。")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/mnt/data/yqmeng/ZRZYB/NRE_GIT_V5_1/output/config.json")
    args = parser.parse_args()
    build_tensors(args.config_path)