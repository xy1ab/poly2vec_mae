import sys, os, torch, argparse, unicodedata, warnings, pyogrio
import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from config import ModelConfig
from models import Attr2Vec
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from data_loader import three_float32_to_float64
warnings.filterwarnings('ignore')



class AttrDecoder_Worker:
    def __init__(self, args):
        config_path = args.config_path
        model_path = args.model_path
        self.config = ModelConfig()
        self.config.load(config_path)
        self.tokenizer = Tokenizer.from_file(self.config.tokenizer_path)
        self.table_id = self.tokenizer.token_to_id("[TABLE]")
        self.data_id = self.tokenizer.token_to_id("[DATA]")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Attr2Vec(self.config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, weights_only=False,map_location=self.device))
        self.model.eval()

    def get_meta_data(self, input_ids):
        ids_long = torch.round(input_ids).long()
        table_indices = (ids_long == self.table_id).int().argmax(dim=-1).tolist()
        data_indices = (ids_long == self.data_id).int().argmax(dim=-1).tolist()
  
        ids_cpu_list = ids_long.tolist()

        all_meta_ids = [row[t+1 : d] for row, t, d in zip(ids_cpu_list, table_indices, data_indices)]
        meta_texts = self.tokenizer.decode_batch(all_meta_ids, skip_special_tokens=True)
        
        return meta_texts, data_indices

    
    # --- 还原文本 (Character-level) ---
    def decode_str(self, batch_size, max_seq_len,data_ids, str_cont, num_cont ):
        char_texts = []
        if str_cont > 0:
            start_idx = 3 * num_cont
            end_idx = start_idx + (str_cont * max_seq_len)
            ch_sc = data_ids[:, start_idx:end_idx].view(batch_size, str_cont, max_seq_len)
            
            # 将 0-1 的归一化数值映射回 Token ID (32768 是原始缩放因子)
            pred_char_ids = torch.round(ch_sc * 32768.0).long()
            
            for i in range(batch_size):
                sample_texts = []
                for c in range(str_cont):
                    ids = pred_char_ids[i, c].cpu().tolist()
                    text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                    sample_texts.append(text if text else "")
                char_texts.append(sample_texts)


    def decoder(self, emb_dir):
        with open(os.path.join(emb_dir, "schema_registry.json"), "r", encoding="utf-8") as f:
            schema_registry = json.load(f)
        embeddings = torch.load(os.path.join(emb_dir, "attr_emb.pt"), weights_only=False)
        dataset = TensorDataset(embeddings)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, pin_memory=True, shuffle=False)
        v_size = self.tokenizer.get_vocab_size()
        with torch.no_grad():
            pbar = tqdm(loader, desc="Decoding Embeddings", unit="batch")

            results = []
            for (batch,) in pbar:
                emb_data = self.model.inn_core(batch.to(self.device), reverse=True).cpu()
                meta_texts, data_indices = self.get_meta_data(emb_data)
                schema_groups = defaultdict(list)
                for i, m_text in enumerate(meta_texts):
                    schema_groups[m_text].append(i)
                for m_text, indices in schema_groups.items():
                    meta_info = schema_registry.get(m_text)
                    if not meta_info: continue
                    n_cols, s_cols = meta_info['num_cols'], meta_info['str_cols']
                    data_pos = data_indices[indices[0]]
                    data_ids = emb_data[indices][:,data_pos+1:]
                    
                    num_w = len(n_cols) * 3
                    dec_n, dec_t = data_ids[:, :num_w], data_ids[:, num_w : num_w + self.config.max_seq_len]
                    ids = torch.clamp(torch.round(dec_t).to(torch.int64), 0, v_size - 1).cpu().tolist()
                    ## get num part
                    for i, col in enumerate(n_cols):
                        p = dec_n[:, i*3 : i*3+3]
                        dec_v = three_float32_to_float64(p).item()
                    ## get str part
                    raw_decoded = self.tokenizer.decode_batch(ids, skip_special_tokens=False)
                    clean_line = raw_decoded.replace("[PAD]", "").strip()
                    dec_parts = [p.replace(" ", "") for p in clean_line.split("[SEP]")]

                    # for r in range(len(data_ids)):
                        
                    #     clean_line = raw_decoded[r].replace("[PAD]", "").replace("[CLS]", "").replace("[MASK]", "").replace("[UNK]", "").strip()
                    #     dec_parts = [p.replace(" ", "") for p in clean_line.split("[SEP]")]

                    #     p = dec_n[r, c_idx*3 : c_idx*3+3]
                    #     dec_v = self.decode_num(p).item()
                                
                               
 
                    #     d_clean = dec_parts[c_idx] if c_idx < len(dec_parts) else ""
                                
 
    def report(self):
        nt, nh = self.stats["n_h"] + self.stats["n_m"], self.stats["n_h"]
        tt, th = self.stats["t_h"] + self.stats["t_m"], self.stats["t_h"]
        print("\n" + "█"*75)
        print(f"🏆 物理轨道全量审计总结报告")
        print(f"📊 数值准确率: {nh}/{nt} ({nh/nt:.2%})" if nt>0 else "📊 数值类: 无数据")
        print(f"📊 文本准确率: {th}/{tt} ({th/tt:.2%})" if tt>0 else "📊 文本类: 无数据")
        print("█"*75 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/config.json")
    parser.add_argument("--emb_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/output")
    parser.add_argument("--model_path", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/attr/output/attr_model.pth")    
    args = parser.parse_args()
    
    decoder_worker = AttrDecoder_Worker(args)
    decoder_worker.decoder(args.emb_dir)
    