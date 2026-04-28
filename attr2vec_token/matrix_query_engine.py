import os
import glob
import json
import torch
import gc
import sys
import time
import numpy as np
import re
import argparse
from tqdm import tqdm
from tokenizers import Tokenizer
from data_loader import three_float32_to_float64

import warnings
warnings.filterwarnings('ignore')

class TensorDatabaseEngine:
    def __init__(self, tensor_dir, schema_path, tokenizer_path, device="cpu", chunk_size=100000):
        print("⏳ [Search Engine] 正在启动张量跨层检索引擎 (工业级高精版)...")
        self.device = torch.device(device)
        self.tensor_dir = tensor_dir
        self.chunk_size = chunk_size  
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            
            # 🛡️ 强制数字提取排序，杜绝切片越界隐患
            raw_files = glob.glob(os.path.join(tensor_dir, "cache_chunk_*.pt"))
            # 🌟 过滤掉 gids 文件，只让纯数据文件进入排查序列
            data_files = [f for f in raw_files if not f.endswith("_gids.pt")]
            self.chunk_files = sorted(data_files, key=lambda x: int(re.search(r'cache_chunk_(\d+)\.pt', os.path.basename(x)).group(1)))
            
            if not self.chunk_files:
                raise FileNotFoundError(f"❌ 在 {tensor_dir} 未发现任何数据碎片！")
            
            # 🌟 启动时自动构建或极速加载 GID 路由字典
            self._build_gid_index()    
                
            print(f"✅ [Search Engine] 引擎就绪！成功挂载 {len(self.chunk_files)} 个数据碎片。")
        except Exception as e:
            print(f"❌ 引擎初始化失败: {e}")
            sys.exit(1)

    # ========================================================
    # 🌟 多模态 GID 平行双轨检索与路由模块 (带极速缓存加速)
    # ========================================================
    def _build_gid_index(self):
        index_path = os.path.join(self.tensor_dir, "global_gid_index.pt")
        
        if os.path.exists(index_path):
            print(f"⚡ 检测到全局 GID 路由缓存 ({index_path})，正在极速加载...")
            self.gid_to_address = torch.load(index_path)
            print(f"✅ GID 路由表就绪！共映射 {len(self.gid_to_address):,} 条全局 ID。")
            return

        print("⏳ 正在扫描底层影子文件，构建全局 GID 路由表 (首次运行或缓存丢失)...")
        self.gid_to_address = {}
        gid_files = glob.glob(os.path.join(self.tensor_dir, "*_gids.pt"))
        
        for file_path in gid_files:
            chunk_idx = int(re.search(r'cache_chunk_(\d+)_gids\.pt', os.path.basename(file_path)).group(1))
            gids = torch.load(file_path).numpy()
            for local_idx, gid in enumerate(gids):
                if gid != -1:
                    self.gid_to_address[int(gid)] = (chunk_idx, local_idx)
                    
        print("💾 正在将 GID 路由表落盘缓存，以便下次启动达 0.1秒级...")
        torch.save(self.gid_to_address, index_path)
        print(f"✅ GID 路由表就绪！共映射 {len(self.gid_to_address):,} 条全局 ID。")

    def fetch_tensors_by_gids(self, target_gids):
        """🌟 终极代码 API：正向提取。供 Python 代码内部极速调用"""
        from collections import defaultdict
        chunk_requests = defaultdict(list)
        results_map = {}
        
        for gid in target_gids:
            gid_int = int(gid)
            if gid_int not in self.gid_to_address:
                continue
            chunk_idx, local_idx = self.gid_to_address[gid_int]
            chunk_requests[chunk_idx].append((gid_int, local_idx))
            
        for chunk_idx, requests in chunk_requests.items():
            chunk_path = os.path.join(self.tensor_dir, f"cache_chunk_{chunk_idx}.pt")
            chunk_dict = torch.load(chunk_path, map_location=self.device, weights_only=False)
            chunk_tensor = chunk_dict["data_ids"] if isinstance(chunk_dict, dict) else chunk_dict
            if not isinstance(chunk_tensor, torch.Tensor): 
                chunk_tensor = torch.tensor(chunk_tensor, dtype=torch.float32, device=self.device)
            elif chunk_tensor.device != self.device:
                chunk_tensor = chunk_tensor.to(self.device).float()
                
            for gid, local_idx in requests:
                results_map[gid] = chunk_tensor[local_idx].clone()
                
        final_tensors = [results_map[int(gid)] for gid in target_gids if int(gid) in results_map]
        return torch.stack(final_tensors) if final_tensors else None

    # ========================================================
    # 🌟 业务逻辑核心：复合属性检索与探针下钻统计
    # ========================================================
    def execute_global_compound_query(self, global_text_reqs, global_num_reqs):
        valid_layers_info = {}
        text_target_tensors = {}
        self.breakdown_stats = {}  

        for req in global_text_reqs:
            field = req['field'].lower()
            text_target_tensors[field] = {}
            self.breakdown_stats[field] = {str(v): 0 for v in req['vals']}
            
            for val in req['vals']:
                val_str = str(val)
                ids_exact = self.tokenizer.encode(val_str).ids
                ids_space = self.tokenizer.encode(" " + val_str).ids
                
                variants = []
                if ids_exact: variants.append((torch.tensor(ids_exact, dtype=torch.float32, device=self.device), len(ids_exact)))
                if ids_space and ids_space != ids_exact: variants.append((torch.tensor(ids_space, dtype=torch.float32, device=self.device), len(ids_space)))
                    
                text_target_tensors[field][val_str] = variants

        for layer_name, info in self.schema.items():
            layer_str_cols_lower = [c.lower() for c in info.get('str_cols', [])]
            layer_num_cols_lower = [c.lower() for c in info.get('num_cols', [])]
            
            if not layer_str_cols_lower and not layer_num_cols_lower and 'fields' in info:
                for col, dtype in info['fields'].items():
                    if 'float' in dtype.lower() or 'int' in dtype.lower(): layer_num_cols_lower.append(col.lower())
                    else: layer_str_cols_lower.append(col.lower())

            has_all_fields = True
            local_num_reqs = []

            for req in global_num_reqs:
                req_f_lower = req['field'].lower()
                if req_f_lower in layer_num_cols_lower:
                    f_idx = layer_num_cols_lower.index(req_f_lower)
                    local_num_reqs.append({"field_idx": f_idx, "op": req['op'], "val": req['val']})
                else:
                    has_all_fields = False
                    break
            if not has_all_fields: continue

            for req in global_text_reqs:
                if req['field'].lower() not in layer_str_cols_lower:
                    has_all_fields = False
                    break
            if not has_all_fields: continue

            meta_short = f"[TABLE] {layer_name} [DATA] "
            meta_ids = self.tokenizer.encode(meta_short).ids
            valid_layers_info[layer_name] = {
                "offset": len(meta_ids),
                "layer_tensor": torch.tensor(meta_ids, dtype=torch.float32, device=self.device),
                "num_reqs": local_num_reqs,
                "text_req_fields": [req['field'].lower() for req in global_text_reqs],
                "num_count": info.get('num_count', len(layer_num_cols_lower))
            }

        if not valid_layers_info:
            return torch.tensor([], dtype=torch.long), []

        matched_global_indices = []
        hit_layers_summary = set() 
        
        sep_id = self.tokenizer.encode(" [SEP] ").ids[0]
        pad_id = 0

        print("\n" + "░"*20 + " [ 引擎全功率矩阵扫描 ] " + "░"*20)
        
        for chunk_file in tqdm(self.chunk_files, desc="🚀 检索进度", colour="cyan", ncols=80, leave=False):
            c_idx = int(re.search(r'cache_chunk_(\d+)\.pt', os.path.basename(chunk_file)).group(1))
            
            chunk_dict = torch.load(chunk_file, map_location=self.device, weights_only=False)
            chunk_tensor = chunk_dict["data_ids"] if isinstance(chunk_dict, dict) else chunk_dict
            if not isinstance(chunk_tensor, torch.Tensor): chunk_tensor = torch.tensor(chunk_tensor, dtype=torch.float32, device=self.device)
            elif chunk_tensor.device != self.device: chunk_tensor = chunk_tensor.to(self.device).float()

            chunk_global_start = c_idx * self.chunk_size

            for layer_name, l_info in valid_layers_info.items():
                offset = l_info["offset"]
                if chunk_tensor.shape[1] < offset: continue
                meta_chunk = chunk_tensor[:, :offset]
                layer_mask = (meta_chunk == l_info["layer_tensor"]).all(dim=-1)

                if not layer_mask.any(): continue
                global_mask = layer_mask.clone()

                for req in l_info["num_reqs"]:
                    field_idx = req["field_idx"]
                    col_start = offset + field_idx * 3
                    chunk_3 = chunk_tensor[:, col_start:col_start+3]
                    real_vals = three_float32_to_float64(chunk_3.T).to(self.device)
                    op, val = req["op"], req["val"]
                    if op == '>': mask = real_vals > val
                    elif op == '<': mask = real_vals < val
                    elif op == '==': mask = real_vals == val
                    elif op == '>=': mask = real_vals >= val
                    elif op == '<=': mask = real_vals <= val
                    global_mask &= mask

                if l_info["text_req_fields"]:
                    text_start = offset + l_info["num_count"] * 3
                    text_chunk = chunk_tensor[:, text_start:]

                    for req_field in l_info["text_req_fields"]:
                        value_variants_dict = text_target_tensors[req_field]
                        field_mask = torch.zeros(global_mask.shape, dtype=torch.bool, device=self.device)
                        
                        for val_str, variants in value_variants_dict.items():
                            val_mask = torch.zeros(global_mask.shape, dtype=torch.bool, device=self.device)
                            
                            for target_tensor, seq_len in variants:
                                if seq_len <= text_chunk.shape[1]:
                                    windows = text_chunk.unfold(dimension=1, size=seq_len, step=1)
                                    matches = (windows == target_tensor).all(dim=-1)
                                    
                                    if matches.any():
                                        num_windows = matches.shape[1]
                                        
                                        # 🛡️ 边界防护
                                        left_valid = torch.zeros_like(matches)
                                        left_valid[:, 0] = True 
                                        if num_windows > 1:
                                            left_valid[:, 1:] = (text_chunk[:, :num_windows-1] == sep_id)

                                        right_valid = torch.zeros_like(matches)
                                        if num_windows > 1:
                                            right_valid[:, :-1] = (text_chunk[:, seq_len:] == sep_id) | (text_chunk[:, seq_len:] == pad_id)
                                        right_valid[:, -1] = True 

                                        strict_matches = matches & left_valid & right_valid
                                        val_mask |= strict_matches.any(dim=-1)
                            
                            # 📊 下钻防污染统计
                            true_hit_mask = val_mask & global_mask
                            self.breakdown_stats[req_field][val_str] += true_hit_mask.sum().item()
                            field_mask |= val_mask
                                        
                        global_mask &= field_mask 

                hit_local = torch.where(global_mask)[0]
                if len(hit_local) > 0:
                    matched_global_indices.append(hit_local + chunk_global_start)
                    hit_layers_summary.add(layer_name)

            del chunk_dict, chunk_tensor
            if self.device.type == "cuda": torch.cuda.empty_cache()
            gc.collect()

        if not matched_global_indices: return torch.tensor([], dtype=torch.long), list(hit_layers_summary)
        return torch.cat(matched_global_indices), list(hit_layers_summary)

    # ========================================================
    # 🌟 闭环核心：将导出的张量与 GID 打包封印
    # ========================================================
    def export_tensors(self, matched_global_indices, output_dir="./export"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        matched_global_indices = torch.sort(matched_global_indices)[0]
        chunks_to_process = {}
        for g_idx in matched_global_indices:
            c_idx = g_idx.item() // self.chunk_size
            if c_idx not in chunks_to_process:
                chunks_to_process[c_idx] = []
            chunks_to_process[c_idx].append(g_idx.item() % self.chunk_size)
            
        extracted_tensors = []
        extracted_gids = []  
        
        print("\n💾 正在跨碎片提取命中的高维特征与平行 GID...")
        for c_idx, local_indices in tqdm(chunks_to_process.items(), desc="提取进度", colour="green", ncols=80):
            chunk_file = os.path.join(self.tensor_dir, f"cache_chunk_{c_idx}.pt")
            gid_file = os.path.join(self.tensor_dir, f"cache_chunk_{c_idx}_gids.pt") 
            
            chunk_dict = torch.load(chunk_file, map_location=self.device, weights_only=False)
            chunk_tensor = chunk_dict["data_ids"] if isinstance(chunk_dict, dict) else chunk_dict
            if not isinstance(chunk_tensor, torch.Tensor): 
                chunk_tensor = torch.tensor(chunk_tensor, dtype=torch.float32, device=self.device)
            
            chunk_gids = torch.load(gid_file).numpy() 
            idx_tensor = torch.tensor(local_indices, dtype=torch.long, device=self.device)
            extracted_tensors.append(chunk_tensor[idx_tensor].cpu())
            
            for l_idx in local_indices:
                extracted_gids.append(int(chunk_gids[l_idx]))
                
            del chunk_dict, chunk_tensor, chunk_gids
            gc.collect()
            
        final_tensor = torch.cat(extracted_tensors, dim=0)
        tensor_save_path = os.path.join(output_dir, f"embs_{timestamp}.pt")
        torch.save(final_tensor, tensor_save_path)
        
        meta_save_path = os.path.join(output_dir, f"embs_{timestamp}_meta.json")
        meta_data = {
            "timestamp": timestamp, 
            "match_count": len(matched_global_indices), 
            "indices": matched_global_indices.tolist(),
            "gids": extracted_gids # 🌟 输出字典自带 GID 主键
        }
        with open(meta_save_path, "w", encoding="utf-8") as f: 
            json.dump(meta_data, f, ensure_ascii=False)
            
        print(f"\n✅ 抽取成功！\n   [特征张量]: {tensor_save_path}\n   [溯源字典 (内含 GID 主键)]: {meta_save_path}")

def main():
    parser = argparse.ArgumentParser(description="Matrix Query Engine")
    parser.add_argument("--tensor_dir", type=str, required=True, help="模型输出目录(包含 cache_chunk)")
    parser.add_argument("--config_dir", type=str, default="", help="可选配置目录")
    parser.add_argument("--export_dir", type=str, required=True, help="检索命中碎片的导出目标目录")
    
    # 🌟🌟🌟 新增：外部系统通信 API 参数
    parser.add_argument("--gids", type=str, default="", help="[API接口] 传入目标 GID 列表(逗号分隔)，静默执行直接导出！")
    args = parser.parse_args()
    
    engine = TensorDatabaseEngine(
        tensor_dir=args.tensor_dir,
        schema_path=os.path.join(args.tensor_dir, "schema_registry.json"),
        tokenizer_path=os.path.join(args.tensor_dir, "zrzy_tokenizer.json"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # ========================================================
    # 📡 接收到外部 API 请求：拦截并静默执行！
    # ========================================================
    if args.gids:
        target_gids = [int(x.strip()) for x in args.gids.split(',') if x.strip()]
        print(f"\n📡 [系统通信接口激活] 收到上游系统发来的提取指令，包含 {len(target_gids)} 个 GID")
        start_time = time.time()
        
        # 换算回全局绝对坐标，复用底层最强大的极速导出组件
        matched_global_indices = []
        for gid in target_gids:
            if gid in engine.gid_to_address:
                c_idx, l_idx = engine.gid_to_address[gid]
                g_idx = c_idx * engine.chunk_size + l_idx
                matched_global_indices.append(g_idx)
                
        if matched_global_indices:
            idx_tensor = torch.tensor(matched_global_indices, dtype=torch.long)
            cost_time = (time.time() - start_time) * 1000
            print(f"✅ GID 映射完成！有效寻址: {len(matched_global_indices)} 条，耗时: {cost_time:.2f} 毫秒")
            # 调用导出模块落盘
            engine.export_tensors(idx_tensor, output_dir=args.export_dir)
            print("👋 [系统通信接口] 交付完成，服务静默退出。")
        else:
            print("⚠️ 警告：接收到的 GID 在底层张量库中均不存在！")
            
        sys.exit(0) # 🌟 核心：执行完毕直接退出，绝不进入交互界面！
    # ========================================================

    # 如果没有传递 --gids，则正常进入用户命令行交互模式
    global_str_cols = set()
    global_num_cols = set()
    for info in engine.schema.values():
        global_str_cols.update([c.lower() for c in info.get('str_cols', [])])
        global_num_cols.update([c.lower() for c in info.get('num_cols', [])])
        if 'fields' in info:
            for c, d in info['fields'].items():
                if 'float' in d.lower() or 'int' in d.lower(): global_num_cols.add(c.lower())
                else: global_str_cols.add(c.lower())
        
    global_str_cols = sorted(list(global_str_cols))
    global_num_cols = sorted(list(global_num_cols))

    while True:
        print("\n" + "="*70)
        print("🎯 自然资源跨层张量检索引擎 (支持并发、边界防护与漏斗探针)")
        print("="*70)

        text_reqs, num_reqs = [], []

        print("\n" + "░"*20 + " [ 阶段 1/2: 文本属性过滤 ] " + "░"*20)
        while True:
            cat_input = input("\n👉 请输入【文本字段名】(直接回车跳过，输入 'q' 退出): ").strip().lower()
            if cat_input == 'q': sys.exit(0)
            if not cat_input: break
            if cat_input not in global_str_cols:
                print(f"   ⚠️ 字段 '{cat_input}' 不存在，请检查。")
                continue
            
            val_input = input(f"   ✏️ 请输入 '{cat_input}' 检索值 (多值逗号分隔): ").strip()
            if val_input:
                vals = [v.strip() for v in val_input.split(',')]
                text_reqs.append({"field": cat_input, "vals": vals})
                print(f"   ✅ 已添加 (包含 {len(vals)} 个检索目标)。")

        print("\n" + "░"*20 + " [ 阶段 2/2: 数值属性过滤 ] " + "░"*20)
        while True:
            num_input = input("\n👉 请输入【数值字段名】(直接回车执行检索): ").strip().lower()
            if not num_input: break
            if num_input not in global_num_cols:
                print(f"   ⚠️ 字段 '{num_input}' 不存在，请检查。")
                continue
            min_input = input(f"   📉 请输入最小值(回车不限): ").strip()
            max_input = input(f"   📈 请输入最大值(回车不限): ").strip()
            if min_input: num_reqs.append({"field": num_input, "op": ">=", "val": float(min_input)})
            if max_input: num_reqs.append({"field": num_input, "op": "<=", "val": float(max_input)})

        if not text_reqs and not num_reqs: 
            print("\n⚠️ 未添加任何条件！")
            continue

        start_time = time.time()
        idx_tensor, hit_layers = engine.execute_global_compound_query(text_reqs, num_reqs)
        cost_time = (time.time() - start_time) * 1000
        match_count = len(idx_tensor)

        print("\n" + "📊 "*2 + "【各项参数独立命中下钻统计表】" + " 📊"*2)
        for field, stats in engine.breakdown_stats.items():
            print(f" 🔹 字段 [{field}]:")
            sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
            for val, count in sorted_stats:
                if count > 0:
                    print(f"    ├─ {val:<12} : {count:>8,} 行")
            
            zero_count = sum(1 for _, c in stats.items() if c == 0)
            if zero_count > 0:
                print(f"    └─ (另有 {zero_count} 个参数未命中任何数据)")

        print("\n" + "█"*70)
        if match_count == 0:
            print("⚠️ 未找到符合条件的数据记录。")
        else:
            print(f"🎉 跨层检索完成! (硬件寻址总耗时: {cost_time:.2f} 毫秒)")
            print(f"📦 共提取到: {match_count} 个物理特征，横跨图层: [{', '.join(hit_layers)}]")
            do_export = input("\n💡 是否将命中的结果导出，留给 Decoder 生成 CSV？(y/n): ").strip().lower()
            if do_export == 'y':
                engine.export_tensors(idx_tensor, output_dir=args.export_dir)
        print("█"*70)

if __name__ == "__main__":
    main()