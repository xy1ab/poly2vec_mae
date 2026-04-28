import os
import glob
import json
import torch
import sys
import numpy as np
import pandas as pd
import re 
import argparse
from collections import defaultdict
from tqdm import tqdm
from tokenizers import Tokenizer
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

class TensorExcelDecoder:
    def __init__(self, schema_path, tokenizer_path, device="cpu", chunk_size=100000):
        print("⏳ [Decoder] 正在启动张量逆向解析引擎 (带复合ID寻址坐标版)...")
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            
            self.layer_signatures = {}
            for layer, info in self.schema.items():
                meta_str = f"[TABLE] {layer} [DATA] "
                meta_ids = self.tokenizer.encode(meta_str).ids
                
                str_cols = info.get('str_cols', [])
                num_cols = info.get('num_cols', [])
                
                if not str_cols and not num_cols and 'fields' in info:
                    for col, dtype in info['fields'].items():
                        if 'float' in dtype.lower() or 'int' in dtype.lower():
                            num_cols.append(col)
                        else:
                            str_cols.append(col)
                            
                self.layer_signatures[layer] = {
                    "offset": len(meta_ids),
                    "str_cols": str_cols,
                    "num_cols": num_cols,
                    "start_idx": int(info.get('start_idx', -1)),
                    "end_idx": int(info.get('end_idx', -1)),
                    "ids": meta_ids
                }
            print(f"✅ [Decoder] 字典加载完成，成功适配 {len(self.schema)} 个图层协议。")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            sys.exit(1)

    def decode_tensor_to_df(self, pt_path, meta_path=None):
        tensor_data = torch.load(pt_path, map_location=self.device, weights_only=False)
        
        is_chunk = False
        chunk_idx = 0
        basename = os.path.basename(pt_path)
        if "cache_chunk_" in basename:
            try:
                chunk_idx = int(basename.replace("cache_chunk_", "").replace(".pt", ""))
                is_chunk = True
            except: pass

        if isinstance(tensor_data, dict) and "data_ids" in tensor_data:
            tensor_data = tensor_data["data_ids"]
            
        if not isinstance(tensor_data, torch.Tensor):
            tensor_data = torch.tensor(tensor_data, dtype=torch.float32, device=self.device)
        elif tensor_data.device != self.device:
            tensor_data = tensor_data.to(self.device)
            
        total_rows = tensor_data.shape[0]
        global_indices = np.full(total_rows, -1, dtype=np.int64)
        is_subset = False
        gids_list = [] # 🌟 接住引擎甩来的 GID
        
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
                gids_list = meta_data.get("gids", []) 
                
                g_idx_list = meta_data.get("indices", [])
                length = min(total_rows, len(g_idx_list))
                global_indices[:length] = g_idx_list[:length]
            is_subset = True
        elif is_chunk:
            global_indices = np.arange(chunk_idx * self.chunk_size, chunk_idx * self.chunk_size + total_rows, dtype=np.int64)
        else:
            global_indices = np.arange(total_rows, dtype=np.int64)

        layer_to_local_idx = {}
        error_log = {}
        has_valid_coords = any(sig["start_idx"] != -1 for sig in self.layer_signatures.values())
        
        if has_valid_coords and (is_chunk or is_subset):
            for layer, sig in self.layer_signatures.items():
                if sig["start_idx"] != -1 and sig["end_idx"] != -1:
                    mask = (global_indices >= sig["start_idx"]) & (global_indices < sig["end_idx"])
                    local_idx = np.where(mask)[0]
                    if len(local_idx) > 0:
                        layer_to_local_idx[layer] = local_idx
        else:
            for layer, sig in self.layer_signatures.items():
                offset = sig["offset"]
                if tensor_data.shape[1] < offset: continue
                sig_tensor = torch.tensor(sig["ids"], dtype=torch.float32, device=self.device)
                match_mask = (tensor_data[:, :offset] == sig_tensor).all(dim=1)
                local_idx = torch.where(match_mask)[0].cpu().numpy()
                
                if len(local_idx) > 0:
                    layer_to_local_idx[layer] = local_idx

        if not layer_to_local_idx:
            error_log["严重错误"] = "坐标失效，GPU 未命中图层！"
            return pd.DataFrame(), error_log

        decoded_rows = []

        for layer, local_idx in layer_to_local_idx.items():
            sig = self.layer_signatures[layer]
            layer_tensor = tensor_data[local_idx] 
            N = layer_tensor.shape[0]
            
            offset = sig["offset"]
            num_cols = sig["num_cols"]
            str_cols = sig["str_cols"]
            
            num_results = {}
            try:
                for f_idx, col_name in enumerate(num_cols):
                    col_start = offset + f_idx * 3
                    chunk_3 = layer_tensor[:, col_start:col_start+3] 
                    p = torch.round(chunk_3).double()
                    real_vals = p[:, 0] + (p[:, 1]/10000.0) + (p[:, 2]/100000000.0)
                    num_results[col_name] = real_vals.cpu().numpy()
            except: continue

            try:
                text_start = offset + len(num_cols) * 3
                text_tensor = layer_tensor[:, text_start:]
                text_lists = text_tensor.long().cpu().tolist()
                raw_texts = self.tokenizer.decode_batch(text_lists, skip_special_tokens=False)
            except: continue
                
            for i in range(N):
                global_id = global_indices[local_idx[i]]
                row_dict = {
                    "全局绝对行号": global_id,
                    "切片编号(Chunk_ID)": global_id // self.chunk_size,
                    "局部行号(Local_FID)": global_id % self.chunk_size,
                    "来源图层": layer
                }
                
                for col_name in num_cols:
                    row_dict[col_name] = round(float(num_results[col_name][i]), 6)
                    
                raw_t = raw_texts[i].replace("[PAD]", "").replace("[CLS]", "")
                text_vals = raw_t.split("[SEP]")
                
                for s_idx, col_name in enumerate(str_cols):
                    if s_idx < len(text_vals):
                        clean_str = text_vals[s_idx]
                        clean_str = clean_str.replace("Ġ", "").replace("\u2581", "")
                        clean_str = re.sub(r'\s+', '', clean_str)
                        row_dict[col_name] = clean_str
                    else:
                        row_dict[col_name] = ""
                        
                decoded_rows.append(row_dict)
                
        df = pd.DataFrame(decoded_rows)
        # 🌟 如果带有 GID 列，将其完美钉在表格第 1 列！
        if gids_list and len(gids_list) == len(df):
            df.insert(0, '全局GID (几何关联主键)', gids_list)

        return df, error_log

    # ========================================================
    # 🌟 核心：基于 Schema 的反向提纯与分表导出
    # ========================================================
    def export_to_csv_files(self, df, save_dir, timestamp):
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            cols_to_drop = [col for col in ['☑️ 核对状态', '✔核对状态', '✔ 核对状态'] if col in df.columns]
            if cols_to_drop: df = df.drop(columns=cols_to_drop)
                
            schema_groups = defaultdict(list)
            unique_layers = df['来源图层'].unique()
            
            for layer in unique_layers:
                layer_df = df[df['来源图层'] == layer].copy()
                
                # 依据物理底图进行强制裁剪，拒绝 Pandas 多层拼接造成的幻影表头
                if layer in self.schema:
                    info = self.schema[layer]
                    expected_cols = ['全局GID (几何关联主键)', '全局绝对行号', '切片编号(Chunk_ID)', '局部行号(Local_FID)', '来源图层']
                    expected_cols += info.get('num_cols', []) + info.get('str_cols', [])
                else:
                    expected_cols = df.columns.tolist()
                    
                actual_cols = [c for c in expected_cols if c in layer_df.columns]
                layer_df = layer_df[actual_cols]
                
                layer_df = layer_df.replace(r'^\s*$', np.nan, regex=True).dropna(axis=1, how='all')
                actual_cols = layer_df.columns.tolist()
                
                schema_signature = tuple(actual_cols)
                schema_groups[schema_signature].append(layer_df)
                
            for idx, (schema, dfs) in enumerate(schema_groups.items()):
                merged_df = pd.concat(dfs, ignore_index=True)
                layers_included = merged_df['来源图层'].unique()
                
                if len(layers_included) > 3:
                    layer_name_str = f"{layers_included[0]}等{len(layers_included)}个同构图层"
                else:
                    layer_name_str = "_".join(map(str, layers_included))
                    
                safe_layer = str(layer_name_str).replace(':', '').replace('/', '_').replace('?', '').replace('*', '')
                csv_save_path = os.path.join(save_dir, f"聚合结构组{idx+1}_{timestamp}_{safe_layer}.csv")
                
                merged_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
                
            print(f"   🎉 智能严格聚合完成！完美拆分为 {len(schema_groups)} 个独立表头的 CSV 文件。")
            return save_dir
        except Exception as e:
            import traceback
            print(f"❌ CSV 导出失败: {e}")
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description="Vector Decoder")
    parser.add_argument("--input_dir", type=str, default=None, help="包含 .pt 碎片的 export 目录路径")
    parser.add_argument("--schema_path", type=str, required=True, help="底座输出的 schema_registry.json 路径")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="底座输出的 zrzy_tokenizer.json 路径")
    args = parser.parse_args()

    decoder = TensorExcelDecoder(args.schema_path, args.tokenizer_path)
    
    if args.input_dir:
        path_input = args.input_dir
        print(f"\n⚡ 检测到超参传递，自动接管目标张量路径: {path_input}")
    else:
        while True:
            print("\n" + "="*65)
            print("📥 物理张量目录级自动解析与报表对账台 (CSV智能聚合版)")
            print("="*65)
            
            path_input = input("\n👉 请输入提取目标文件夹路径 (输入 'q' 退出): ").strip()
            
            if path_input.lower() == 'q': return
            if not os.path.exists(path_input):
                print("   ⚠️ 找不到该路径，请核实。")
                continue
            break
        
    pt_files = []
    if os.path.isfile(path_input) and path_input.endswith('.pt'):
        pt_files.append(path_input)
    elif os.path.isdir(path_input):
        pt_files = sorted(glob.glob(os.path.join(path_input, "*.pt")))
        if not pt_files:
            print(f"   ⚠️ 未在 {path_input} 找到任何 .pt 文件！程序退出。")
            sys.exit(1)

    all_dfs = []
    global_errors = {}
    print("\n⚡ 正在调度 GPU 算子执行全矩阵反演...")
    
    for pt_file in tqdm(pt_files, desc="张量解码进度", colour="green", ncols=80):
        meta_file = pt_file.replace(".pt", "_meta.json")
        df, err_log = decoder.decode_tensor_to_df(pt_file, meta_file)
        
        if not df.empty:
            all_dfs.append(df)
            
        for k, v in err_log.items():
            global_errors[k] = global_errors.get(k, 0) + v

    if not all_dfs:
        print("\n" + "❗"*65)
        print("   ⚠️ 严重警告：提取失败！请查看异常原因:")
        if not global_errors:
            print("      - 未匹配到合法图层，张量可能为 0。")
        for k, v in global_errors.items():
            if isinstance(v, int): print(f"      - {k} (共发生 {v} 次)")
            else: print(f"      - {k}: {v}")
        print("❗"*65)
        sys.exit(1)
        
    print("\n⏳ 正在清洗特征矩阵，执行基于物理底图的多图层聚合...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = path_input if os.path.isdir(path_input) else os.path.dirname(path_input)
    
    save_path = decoder.export_to_csv_files(final_df, save_dir, timestamp)
    
    print("\n🏆 全量明文对账单生成成功:")
    print(f"   - 总共反演恢复数据: {len(final_df)} 行")
    print(f"   - 目标文件位置    : {save_path}")

if __name__ == "__main__":
    main()