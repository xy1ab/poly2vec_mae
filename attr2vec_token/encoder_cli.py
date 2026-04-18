import os, glob, torch, argparse, pyogrio, warnings
import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from config import ModelConfig
from models import Attr2Vec
from data_loader import float64_to_three_float32, process_dataframe
from data_builder import MUST_BE_STRING_SUFFIX, load_csv_safely
import json

# 🌟 屏蔽底层的空间数据类型转换警告，保持控制台整洁
warnings.filterwarnings('ignore', category=UserWarning)

def encoder_data(args, model):
    emb_list = []
    schema_registry = {}
    files_to_process = []
    
    # ✅ 深度递归扫描，不漏掉任何子文件夹中的数据
    search_pattern = os.path.join(args.data_dir, "**", "*")
    for file_path in sorted(glob.glob(search_pattern, recursive=True)):
        # ✅ 全面支持 CSV、GDB 以及 SHP
        if file_path.lower().endswith('.csv') or file_path.lower().endswith('.gdb') or file_path.lower().endswith('.shp'):
            files_to_process.append(file_path)

    for i, file_path in enumerate(files_to_process):
        print(f"🔍 处理进度 [{i+1}/{len(files_to_process)}]: {os.path.basename(file_path)}")
        
        if file_path.lower().endswith('.csv'):
            layer_name = os.path.basename(file_path).split('.')[0]
            df = load_csv_safely(file_path)
            # ✅ 您的核心优势：空表防御机制
            if df is not None and not df.empty:
                # ✅ 您的核心优势：参数齐全，打通物理寻址
                input_ids = process_dataframe(df, layer_name, tokenizer, config, schema_registry, file_path)
                if input_ids is not None:
                    # ✅ 您的核心优势：完善的物理防线备案 (支撑高并发切片与溯源)
                    schema_registry[layer_name]["row_count"] = len(df)
                    schema_registry[layer_name]["raw_path"] = file_path
                    with torch.no_grad():
                        emb = model.inn_core(torch.tensor(input_ids).to(device), reverse=False)
                        emb_list.append(emb.detach().cpu())
                        
        elif file_path.lower().endswith('.gdb'):
            try:
                layers = pyogrio.list_layers(file_path)
                for layer_name, geom_type in layers:
                    df = pyogrio.read_dataframe(file_path, layer=layer_name, read_geometry=False)
                    if len(df) == 0: continue
                    input_ids = process_dataframe(df, layer_name, tokenizer, config, schema_registry, file_path)
                    if input_ids is not None:
                        schema_registry[layer_name]["row_count"] = len(df)
                        schema_registry[layer_name]["raw_path"] = file_path
                        with torch.no_grad():
                            emb = model.inn_core(torch.tensor(input_ids).to(device), reverse=False)
                            emb_list.append(emb.detach().cpu())
            except Exception as e:
                print(f"❌ 读取 GDB 失败 [{file_path}]: {e}")
                
        elif file_path.lower().endswith('.shp'):
            try:
                layer_name = os.path.basename(file_path).split('.')[0]
                df = pyogrio.read_dataframe(file_path, read_geometry=False)
                if len(df) == 0: continue
                input_ids = process_dataframe(df, layer_name, tokenizer, config, schema_registry, file_path)
                if input_ids is not None:
                    schema_registry[layer_name]["row_count"] = len(df)
                    schema_registry[layer_name]["raw_path"] = file_path
                    with torch.no_grad():
                        emb = model.inn_core(torch.tensor(input_ids).to(device), reverse=False)
                        emb_list.append(emb.detach().cpu())
            except Exception as e:
                print(f"❌ 读取 SHP 失败 [{file_path}]: {e}")

    if len(emb_list) > 0:
        emb = torch.concat(emb_list)
        torch.save(emb, os.path.join(args.output_dir, "attr_emb.pt"))
        
        # 🚀 组长的核心优势：彻底抛弃随机种子，将生成当前 Embedding 的模型参数固化落盘
        model_save_path = os.path.join(args.output_dir, "attr_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ 模型解密密钥已固化至: {model_save_path}")
        
        schema_path = os.path.join(args.output_dir, "schema_registry.json")
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_registry, f, ensure_ascii=False, indent=4)
            
        print(f"✅ 编码完成，共成功穿过网络打包 {len(emb_list)} 个图层。")
    else:
        print("⚠️ 警告：未提取到任何有效数据特征，跳过编码。")

if __name__ == "__main__":
    # ❌ 随机种子机制已被彻底移除 (已删除 torch.manual_seed)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/config.json")
    parser.add_argument("--data_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/data/raw")
    parser.add_argument("--output_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = ModelConfig()
    config.load(args.config_path)
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🌟 每次启动脚本，模型都会基于当前系统状态自然进行纯随机初始化
    # 随后这些用于特征解缠的随机权重会在任务结束时被永久记录为 .pth 密钥
    model = Attr2Vec(config).to(device)
    model.eval()

    encoder_data(args, model)