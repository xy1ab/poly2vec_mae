import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
import os
import json
import joblib
import glob
import numpy as np
from models import NaturalResourceFoundationModel
from tokenizers import Tokenizer 
from torch.utils.data import TensorDataset, DataLoader
import argparse
from tqdm import tqdm

def build_arg_parser() -> argparse.ArgumentParser:
    """构建南湖平台标准 CLI 解析器"""
    parser = argparse.ArgumentParser(
        description="Extract text corpus and train a custom BPE tokenizer for NRE data."
    )
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    parser.add_argument("--vocab_path", type=str, required=True, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    parser.add_argument("--model_path", type=str, required=True, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    parser.add_argument("--vocab_size", type=int, default=8192, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    parser.add_argument("--batch_size", type=int, default=2048, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    parser.add_argument("--embedding_path", type=str, required=True, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    
    return parser


class BatchEmbeddingInference:
    def __init__(self, model_path, tokenizer_path, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 基础配置，需与模型训练参数对齐
        self.config = config or {
            'truth_dim': 256,
            'semantic_dim': 256,
            'vocab_size': 8192,
            'max_seq_len': 64
        }
        
        # 加载模型
        self.model = NaturalResourceFoundationModel(self.config).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    @torch.no_grad()
    def process_layer(self, latent_tensor, meta):
        """处理单层数据的推理还原"""
        v_attr = latent_tensor[:, :self.config['truth_dim']].to(self.device)
        dec = self.model.inn_core(v_attr, reverse=True)
        
        num_cont = len(meta.get('cont_cols', []))
        num_word = len(meta.get('word_cols', []))
        num_char = len(meta.get('char_cols', []))
        max_seq_len = meta.get('max_seq_len', 64)
        
        results = []
        
        # 1. 数值还原 (ldexp 逆操作)
        cont_values = None
        if num_cont > 0:
            c_exp = dec[:, :num_cont]
            c_hi = dec[:, num_cont : 2*num_cont]
            c_lo = dec[:, 2*num_cont : 3*num_cont]
            cont_values = torch.ldexp(c_hi.double() + c_lo.double(), c_exp.int())

        # 2. 文本还原 (BPE Decode)
        char_results = []
        if num_char > 0:
            start_idx = 3 * num_cont + num_word
            end_idx = start_idx + (num_char * max_seq_len)
            ch_sc = dec[:, start_idx:end_idx].view(-1, num_char, max_seq_len)
            pred_char_ids = torch.round(ch_sc * 32768.0).long()
            
            for i in range(pred_char_ids.size(0)):
                row_texts = []
                for c in range(num_char):
                    ids = pred_char_ids[i, c].cpu().tolist()
                    text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                    row_texts.append(text)
                char_results.append(row_texts)

        # 3. 结果映射到列名
        for i in range(dec.size(0)):
            row_data = {}
            if cont_values is not None:
                for idx, col in enumerate(meta['cont_cols']):
                    row_data[col] = cont_values[i, idx].item()
            if char_results:
                for idx, col in enumerate(meta['char_cols']):
                    row_data[col] = char_results[i][idx]
            results.append(row_data)
            
        return results

def run_full_inference(latent_file, model_path, tokenizer_path):
    """
    latent_file: joblib 文件路径，包含所有层的 embeddings
    """
    inference_engine = BatchEmbeddingInference(model_path, tokenizer_path)
    
    # 加载 Embedding 数据
    all_latents = joblib.load(latent_file)
    final_output = {}

    print(f"📂 开始遍历推理，共 {len(all_latents)} 个数据项...")
    
    for task, task_result in all_latents.items():
        # 假设 key 的格式是 "source_layer"，我们需要从中提取层名来匹配 meta
        # 如果你的 key 直接就是层名，直接用 key 即可
        print(f"🔍 处理文件: {task}")
        layer_output = {}
        for layer, layer_result in task_result.items():
            print(f"🔍 处理layer: {layer}")
            latent_tensor = layer_result['latent']
            meta = layer_result['meta']

            
        
            # 分 Batch 推理（防止显存溢出）
            batch_size = 1024
            layer_results = []
            for i in tqdm(range(0, latent_tensor.size(0), batch_size), desc=f"Processing {layer}"):
                batch_latent = latent_tensor[i : i + batch_size]
                batch_res = inference_engine.process_layer(batch_latent, meta)
                layer_results.extend(batch_res)
                
            layer_output[layer] = layer_results
        final_output[task] = layer_output
    return final_output

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    
    ## read gdb and extract data --- IGNORE ---
    # import fiona
    # import geopandas as gpd
    # layers = fiona.listlayers("/mnt/git-data/HB/poly2vec_mae/data/raw/LCXZ_TEST.gdb")
    # for i, layer_name in enumerate(layers):
    #     print(f"  -> 抽取图层 [{i+1}/{len(layers)}]: {layer_name}")
    #     # ignore_geometry=True 抛弃空间图形，只读属性表，极大提速
    #     tb = gpd.read_file("/mnt/git-data/HB/poly2vec_mae/data/raw/LCXZ_TEST.gdb", layer=layer_name, ignore_geometry=True)
    #     print(tb)
    ##

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    vocab_size = args.vocab_size
    embedding_path = args.embedding_path
    results = run_full_inference(
            latent_file=embedding_path,
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )
    
