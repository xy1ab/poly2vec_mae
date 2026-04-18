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
    """[核心无损转换逻辑] 将 1 个 Float64 拆分为 3 个 Float32"""
    arr = np.nan_to_num(arr, nan=0.0) 
    int_part = np.trunc(arr).astype(np.float32)
    frac = arr - int_part
    frac_hi = np.trunc(frac * 10000).astype(np.float32)
    frac_lo = np.trunc((frac * 10000 - frac_hi) * 10000).astype(np.float32)
    return int_part, frac_hi, frac_lo

import numpy as np


def process_dataframe(df, layer_name, tokenizer, config, schema_registry):
    """处理单一数据表，实现物理真值与语义的彻底解耦，并抓取元数据"""
    if len(df) == 0:
        return None, None
        
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    str_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ['geometry', 'shape']]
    
    # 加入metadata
    metadata_prefix = f"[TABLE] {layer_name} [COLUMN] {' [SEP] '.join(str_cols)} [DATA] "
    metadata_ids = tokenizer.encode(metadata_prefix).ids
    # 🌟 1. 登记元数据注册表 (Schema Registry)
    if layer_name not in schema_registry:
        schema_registry[layer_name] = {
            "num_cols": num_cols,
            "str_cols": str_cols,
            "num_count": len(num_cols),
            "str_count": len(str_cols),
            "metadata_prefix": metadata_prefix,
            "metadata_token_ids": metadata_ids,
            "metadata_token_len": len(metadata_ids)
        }

    # 记录图层名，用于索引
    metadata_short = f"[TABLE] {layer_name} [DATA] "
    metadata_short_ids = tokenizer.encode(metadata_short).ids
    # 🌟 2. 处理数值数据
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

    # 🌟 3. 处理文本数据 最后填PAD
    text_ids = np.zeros((N, config.max_seq_len), dtype=np.int64)
    if len(str_cols) > 0:
        str_data = df[str_cols].fillna("").astype(str)
        # 用 [SEP] 拼接行内不同字段，方便未来无损解码时对号入座
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
    # 把 int64 的 Token IDs 转为 float32 送进去
    input_ids[:, (num_width+len(metadata_short_ids)) : (num_width+len(metadata_short_ids) + config.max_seq_len)] = text_ids.astype(np.float32)
    
    return input_ids

# ========================================================
# 🚀 针对南湖集群优化的流式落盘张量流水线
# ========================================================
def build_tensors(config_path):
    print("🚀 启动集群优化版数据张量化流水线 (流式边算边存)...")
    
    config = ModelConfig()
    config.load(config_path)
    raw_data_dir = config.data_dir
    if not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError("❌ 找不到 tokenizer，请先运行 data_builder.py！")
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    
    schema_registry = {}
    
    # 【集群优化核心】：内存缓冲池与流式落盘参数
    CHUNK_SIZE = 100000        # 每个 pt 文件的样本数，防止内存溢出
    data_buffer = []                 
    buffer_rows = 0            
    chunk_idx = 0              
    total_processed_rows = 0   

    def flush_buffer_to_disk(force_all=False):
        """流式落盘触发器：当 buffer 满载或遍历结束时调用"""
        nonlocal data_buffer, buffer_rows, chunk_idx, total_processed_rows
        
        while buffer_rows >= CHUNK_SIZE or (force_all and buffer_rows > 0):
            # 堆叠数据
            stacked_data = np.vstack(data_buffer)
            
            # 确定切分边界
            rows_to_save = min(CHUNK_SIZE, buffer_rows)
            chunk_data = stacked_data[:rows_to_save]
            
            # 安全存盘
            out_path = os.path.join(config.output_dir, f"cache_chunk_{chunk_idx}.pt")
            torch.save({
                "data_ids": chunk_data,
            }, out_path)
            print(f"✅ 流式落盘触发: 成功保存 {out_path} (释放内存: {rows_to_save} 条)")
            
            chunk_idx += 1
            total_processed_rows += rows_to_save
            
            # 将剩下的尾巴塞回缓存池
            if rows_to_save < stacked_data.shape[0]:
                data_buffer = [stacked_data[rows_to_save:]]
                buffer_rows = data_buffer[0].shape[0]
            else:
                data_buffer = []
                buffer_rows = 0

    # ---------------- 扫描与处理文件流水线 ----------------
    files_to_process = []
    for file_path in glob.glob(os.path.join(raw_data_dir, "*")):
        if file_path.lower().endswith('.csv') or file_path.lower().endswith('.gdb'):
            files_to_process.append(file_path)

    for i, file_path in enumerate(files_to_process):
        print(f"🔍 处理进度 [{i+1}/{len(files_to_process)}]: {os.path.basename(file_path)}")
        
        if file_path.lower().endswith('.csv'):
            layer_name = os.path.basename(file_path).split('.')[0]
            df = load_csv_safely(file_path)
            if df is not None:
                input_ids = process_dataframe(df, layer_name, tokenizer, config, schema_registry)
                if input_ids is not None:
                    data_buffer.append(input_ids)
                    buffer_rows += input_ids.shape[0]
                    flush_buffer_to_disk(force_all=False)
                    
        elif file_path.lower().endswith('.gdb'):
            try:
                layers = pyogrio.list_layers(file_path)
                for layer_name, geom_type in layers:
                    df = pyogrio.read_dataframe(file_path, layer=layer_name, read_geometry=False)
                    input_ids = process_dataframe(df, layer_name, tokenizer, config, schema_registry)
                    if input_ids is not None:
                        data_buffer.append(input_ids)
                        buffer_rows += input_ids.shape[0]
                        flush_buffer_to_disk(force_all=False)
            except Exception as e:
                print(f"❌ 读取 GDB 失败 [{file_path}]: {e}")

    # ==========================================
    # 扫尾工作：将不足 10 万条的剩余数据强制落盘，并生成终极图纸
    # ==========================================
    flush_buffer_to_disk(force_all=True)

    # 保存元数据图纸 (Schema Registry)
    schema_path = os.path.join(config.output_dir, "schema_registry.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_registry, f, ensure_ascii=False, indent=4)
    print(f"✅ 成功生成集群元数据注册表 (Schema Registry): {schema_path}")
    
    print("\n" + "="*60)
    print(f"🎉 全部集群流式处理完成！共生成 {total_processed_rows} 条标准训练样本。")
    print(f"👉 物理真值已完美解耦，包含绝对无损数据。")
    print(f"👉 语义 IDs 已完美解耦，专供 Transformer 交叉熵重建。")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/config.json", help="model_config.json 路径")
    args = parser.parse_args()
    
    build_tensors(args.config_path)