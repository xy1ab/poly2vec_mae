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
    for file_path in glob.glob(os.path.join(args.data_dir, "*")):
        if file_path.lower().endswith('.csv') or file_path.lower().endswith('.gdb'):
            files_to_process.append(file_path)

    for i, file_path in enumerate(files_to_process):
        print(f"🔍 处理进度 [{i+1}/{len(files_to_process)}]: {os.path.basename(file_path)}")
        if file_path.lower().endswith('.csv'):
            layer_name = os.path.basename(file_path).split('.')[0]
            df = load_csv_safely(file_path)
            if df is not None:
                input_ids = process_dataframe(df, layer_name, tokenizer, config, schema_registry)
                with torch.no_grad():
                    emb = model.inn_core(torch.tensor(input_ids).to(device), reverse=False)
                    emb_list.append(emb)
        elif file_path.lower().endswith('.gdb'):
            try:
                layers = pyogrio.list_layers(file_path)
                for layer_name, geom_type in layers:
                    df = pyogrio.read_dataframe(file_path, layer=layer_name, read_geometry=False)
                    input_ids = process_dataframe(df, layer_name, tokenizer, config, schema_registry)
                    with torch.no_grad():
                        emb = model.inn_core(torch.tensor(input_ids).to(device), reverse=False)
                        emb_list.append(emb.detach().cpu())
            except Exception as e:
                print(f"❌ 读取 GDB 失败 [{file_path}]: {e}")
    emb = torch.concat(emb_list)
    torch.save(emb, os.path.join(args.output_dir, "attr_emb.pt"))
    schema_path = os.path.join(args.output_dir, "schema_registry.json")
    torch.save(model.state_dict(), os.path.join(args.output_dir,"attr_model.pth"))
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_registry, f, ensure_ascii=False, indent=4)
    print(f"✅ 编码完成，共成功打包 {len(emb_list)} 个图层。")
    return emb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/config.json")
    parser.add_argument("--data_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/data/raw")
    parser.add_argument("--output_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = ModelConfig()
    config.load(args.config_path)
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Attr2Vec(config).to(device)
    model.eval()
    encoder_data(args, model)
