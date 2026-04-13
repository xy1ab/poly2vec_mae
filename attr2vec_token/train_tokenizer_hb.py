import os
import glob
import argparse
import pandas as pd
import fiona
import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ==========================================
# 🛡️ 核心基建配置：绝对文本列白名单
# 凡是包含以下后缀，或完全匹配的列，强制作为 String 读取，绝对不转数字！
# ==========================================
MUST_BE_STRING_EXACT = ['BSM', 'YSDM', 'PAC', 'OBJECTID']  # 完全匹配名单
MUST_BE_STRING_SUFFIX = ('代码', '编码', '编号', 'ID')      # 后缀匹配名单

def load_csv_safely(csv_path):
    """工业级安全读取 CSV：利用白名单阻断 Pandas 的自动数值推断"""
    try:
        # 1. 探路：只读取表头
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        
        # 2. 发护身符：动态生成 dtype 映射字典
        dtype_mapping = {}
        for col in columns:
            # 统一转大写比对，防止出现 bsm 和 BSM 的大小写差异导致漏判
            if col.upper() in MUST_BE_STRING_EXACT or col.endswith(MUST_BE_STRING_SUFFIX):
                dtype_mapping[col] = str
                
        # 3. 正式读取：只有在 mapping 里的列会被强制按字符串读，其余依然会自动推断
        df = pd.read_csv(csv_path, dtype=dtype_mapping, low_memory=False)
        return df
    except Exception as e:
        print(f"❌ 读取 CSV 失败 [{csv_path}]: {e}")
        return None

# ==========================================
# 命令行接口适配
# ==========================================
def build_arg_parser() -> argparse.ArgumentParser:
    """构建南湖平台标准 CLI 解析器"""
    parser = argparse.ArgumentParser(
        description="Extract text corpus and train a custom BPE tokenizer for NRE data."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing raw data (CSV and GDB).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the corpus and tokenizer json.")
    parser.add_argument("--tokenizer_vocab_size", type=int, default=8192, help="Vocabulary size for the BPE tokenizer (default: 20000).")
    return parser

# ==========================================
# 1. 提取所有文本语料 (动态扫描版)
# ==========================================
def extract_all_texts(data_dir, output_dir):
    print(f"⏳ 正在从数据目录 [{data_dir}] 动态扫描并提取文本语料，请稍候...")
    corpus = []
    
    # 🌟 核心防御：匹配纯数字、小数、负数。这些即使作为字符串读进来，也不要喂给 Tokenizer 去占用词表名额！
    # pure_number_pattern = re.compile(r'^-?\d+(\.\d+)?$')
    
    # 动态扫荡所有 CSV
    for csv_file in glob.glob(os.path.join(data_dir, "*.csv")):
        print(f"  -> 解析 CSV: {os.path.basename(csv_file)}")
        
        # 使用我们的安全读取引擎替代原先的 pd.read_csv
        df = load_csv_safely(csv_file)
        if df is not None:
            for col in df.columns:
                unique_texts = df[col].dropna().unique()
                for text in unique_texts:
                    if isinstance(text, str):
                        text = text.strip()
                        corpus.append(text)
                
    # 动态扫荡所有 GDB
    for gdb_file in glob.glob(os.path.join(data_dir, "*.gdb")):
        print(f"  -> 解析 GDB: {os.path.basename(gdb_file)}")
        try:
            layers = fiona.listlayers(gdb_file)
            for layer in layers:
                with fiona.open(gdb_file, layer=layer) as src:
                    for feat in src:
                        props = feat.get('properties', {})
                        for k, v in props.items():
                            if isinstance(v, str):
                                text = v.strip()
                                # GDB里的文本也加上过滤，防患于未然
                                # if text and not pure_number_pattern.match(text):
                                corpus.append(text)
        except Exception as e:
            print(f"❌ 读取 GDB 失败 [{gdb_file}]: {e}")

    # 去重并保存至指定的输出目录
    corpus = list(set(corpus))
    corpus_path = os.path.join(output_dir, "zrzy_corpus.txt")
    
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in corpus:
            f.write(text + "\n")
            
    print(f"✅ 语料提取完毕！共提取 {len(corpus)} 条独立有效文本。已保存至: {corpus_path}")
    return corpus_path

# ==========================================
# 2. 训练 BPE 分词器 (自适应路径版)
# ==========================================
def train_bpe_tokenizer(corpus_path, output_dir,vocab_size=8192):

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], vocab_size=vocab_size)
    
    # 喂入刚生成的语料
    tokenizer.train(files=[corpus_path], trainer=trainer)
    
    # 保存至指定输出目录
    tokenizer_path = os.path.join(output_dir, "zrzy_tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"🚀 训练成功！专属分词器已保存至: {tokenizer_path}")

# ==========================================
# 执行总控
# ==========================================
if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"❌ 数据目录不存在: {args.data_dir}")
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"📁 已自动创建输出存放目录: {args.output_dir}")

    # 1. 抽取聚合语料
    generated_corpus_path = extract_all_texts(args.data_dir, args.output_dir)
    
    # 2. 训练出炉 Tokenizer
    train_bpe_tokenizer(generated_corpus_path, args.output_dir,args.tokenizer_vocab_size)