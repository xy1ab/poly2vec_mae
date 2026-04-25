# 导入系统级操作和路径操作库
import os
import glob
import argparse

# 导入数据处理利器
import pandas as pd
import pyogrio

# 导入 HuggingFace 的 Tokenizers 库，用于构建专属的 BPE 分词器
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 导入我们自定义的全局参数配置类
from config import ModelConfig, get_optimal_dim

# # 导入警告控制库，屏蔽一些不影响运行的第三方库底层警告信息
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 🛡️ 核心基建配置：绝对文本列白名单
# ==========================================
MUST_BE_STRING_EXACT = ['BSM', 'YSDM', 'PAC', 'OBJECTID', 'DLBM']  

MUST_BE_STRING_SUFFIX = (
    '代码', '编码', '编号', 'ID', 'id', 'Id', 
    'CODE', 'code', 'Code',
    'bm', 'BM', 'dm', 'DM'
)      

def load_csv_safely(csv_path):
    """工业级安全读取 CSV：利用白名单阻断 Pandas 的自动数值推断"""
    try:
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
        dtype_mapping = {}
        for col in columns:
            if col.upper() in MUST_BE_STRING_EXACT or col.endswith(MUST_BE_STRING_SUFFIX):
                dtype_mapping[col] = str
                
        df = pd.read_csv(csv_path, dtype=dtype_mapping, low_memory=False)
        return df
    except Exception as e:
        print(f"❌ 读取 CSV 失败 [{csv_path}]: {e}")
        return None

def build_tokenizer_and_sniff_dims(config: ModelConfig, raw_data_dir: str):
    """核心扫描函数：嗅探真值上限维度 + 提取语料训练分词器"""
    print("🚀 启动大一统底座预处理：真值维度精准嗅探与 BPE 训练...")
    print(f"🔍 正在扫描数据目录: {raw_data_dir}")
    
    if not os.path.exists(raw_data_dir):
        print(f"\n❌ 严重错误：系统找不到路径 '{raw_data_dir}'，请检查挂载状态！")
        return

    corpus = []
    
    # 🌟 核心数据结构：记录每张表的 (数值维度, [该表的所有不重复长文本])
    table_records = []
    
    csv_files = []
    gdb_files = []
    shp_files = [] # 🌟 新增：SHP 文件列表
    
    # ==========================================
    # 1. 扫描目录下的所有 CSV, GDB 和 SHP
    # ==========================================
    # 使用 ** 匹配所有子目录，recursive=True 开启深度扫描
    search_pattern = os.path.join(raw_data_dir, "**", "*")
    for file_path in sorted(glob.glob(search_pattern, recursive=True)):
        if file_path.lower().endswith('.csv'):
            csv_files.append(file_path)
        elif file_path.lower().endswith('.gdb'):
            gdb_files.append(file_path)
        elif file_path.lower().endswith('.shp'): # 🌟 新增：扫描 SHP 文件
            shp_files.append(file_path)

    total_gdb_layers = 0
    for gdb in gdb_files:
        try:
            total_gdb_layers += len(pyogrio.list_layers(gdb))
        except:
            pass

    print(f"\n📂 扫描完毕！共发现 {len(csv_files)} 个 CSV, {len(gdb_files)} 个 GDB (包含 {total_gdb_layers} 个图层), 以及 {len(shp_files)} 个 SHP 文件。")

    if len(csv_files) == 0 and len(gdb_files) == 0 and len(shp_files) == 0:
        print("❌ 错误：在该路径未发现任何支持的数据文件，程序熔断！")
        return

    # ==========================================
    # 2. 解析 CSV 数据表
    # ==========================================
    print("📂 开始解析 CSV 数据...")
    for csv_file in csv_files:
        df = load_csv_safely(csv_file)
        if df is None: continue

        # 🌟 针对 CSV：拦截并剔除 gid
        gid_col = next((c for c in df.columns if c.lower() == 'gid'), None)
        if gid_col:
            df = df.drop(columns=[gid_col])

        layer_name = os.path.basename(csv_file).split('.')[0]

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        str_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ['geometry', 'shape']]
        # 加入metadata
        metadata_prefix = f"[TABLE] {layer_name} [COLUMN] {' [SEP] '.join(str_cols)} [DATA] "
        metadata_short = f"[TABLE] {layer_name} [DATA] "
        corpus.append(layer_name)
        corpus.extend(str_cols)
        for col in str_cols:
            valid_texts = df[col].dropna().astype(str).unique().tolist()
            corpus.extend([t.strip() for t in valid_texts if t.strip()])
            
        num_dim = len(num_cols) * 3
        unique_row_strings = []
        if str_cols:
            unique_row_strings = metadata_short + df[str_cols].fillna("").astype(str).agg(' [SEP] '.join, axis=1).unique()
        
        table_records.append((num_dim, unique_row_strings))

    # ==========================================
    # 3. 解析 GDB 数据库图层
    # ==========================================
    print("📂 开始解析 GDB 数据库...")
    for gdb_file in gdb_files:
        try:
            layers = pyogrio.list_layers(gdb_file)
            for layer_name, geom_type in layers:
                
                df = pyogrio.read_dataframe(gdb_file, layer=layer_name, read_geometry=False)
                if len(df) == 0: continue

                # 🌟 针对 GDB：拦截并剔除 gid
                gid_col = next((c for c in df.columns if c.lower() == 'gid'), None)
                if gid_col:
                    df = df.drop(columns=[gid_col])

                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                str_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ['geometry', 'shape']]
                
                metadata_prefix = f"[TABLE] {layer_name} [COLUMN] {' [SEP] '.join(str_cols)} [DATA] "
                metadata_short = f"[TABLE] {layer_name} [DATA] "
                corpus.append(layer_name)
                corpus.extend(str_cols)
                for col in str_cols:
                    valid_texts = df[col].dropna().astype(str).unique().tolist()
                    corpus.extend([t.strip() for t in valid_texts if t.strip()])
                    
                num_dim = len(num_cols) * 3
                unique_row_strings = []
                if str_cols:
                    unique_row_strings = metadata_short + df[str_cols].fillna("").astype(str).agg(' [SEP] '.join, axis=1).unique()
                
                table_records.append((num_dim, unique_row_strings))
        except Exception as e:
            print(f"❌ 读取 GDB 失败 [{gdb_file}]: {e}")

    # ==========================================
    # 3.5 🌟 新增：解析 SHP 文件
    # ==========================================
    print("📂 开始解析 SHP 数据...")
    for shp_file in shp_files:
        try:
            layer_name = os.path.basename(shp_file).split('.')[0]
            # 利用 pyogrio 高效读取 SHP 属性表 (不读 geometry)
            df = pyogrio.read_dataframe(shp_file, read_geometry=False)
            if len(df) == 0: continue
            
            # 🌟 针对 SHP：拦截并剔除 gid
            gid_col = next((c for c in df.columns if c.lower() == 'gid'), None)
            if gid_col:
                df = df.drop(columns=[gid_col])

            # ----------- ✅ SHP 专属验证打印 -----------
            print(f"\n✅ [SHP 验证] 成功连接文件: {os.path.basename(shp_file)}")
            print(f"📊 样本量: {len(df)} 行 | 属性维度: {len(df.columns)} 列")
            # ----------------------------------------

            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            str_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ['geometry', 'shape']]
            
            metadata_prefix = f"[TABLE] {layer_name} [COLUMN] {' [SEP] '.join(str_cols)} [DATA] "
            metadata_short = f"[TABLE] {layer_name} [DATA] "
            corpus.append(layer_name)
            corpus.extend(str_cols)
            for col in str_cols:
                valid_texts = df[col].dropna().astype(str).unique().tolist()
                corpus.extend([t.strip() for t in valid_texts if t.strip()])
                
            num_dim = len(num_cols) * 3
            unique_row_strings = []
            if str_cols:
                unique_row_strings = metadata_short + df[str_cols].fillna("").astype(str).agg(' [SEP] '.join, axis=1).unique()
            
            table_records.append((num_dim, unique_row_strings))
        except Exception as e:
            print(f"❌ 读取 SHP 失败 [{shp_file}]: {e}")

    # ==========================================
    # 4. 训练自然资源专属 BPE 分词器
    # ==========================================
    print(f"\n🧠 语料收集完毕，共提取 {len(corpus)} 条短语，开始训练 Tokenizer...")
    
    if len(corpus) == 0:
        print("⚠️ 警告：提取到的语料为空！将跳过分词器训练。")
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        custom_tokens = ["[TABLE]", "[COLUMN]", "[DATA]"]
        all_special = ["[PAD]", "[UNK]", "[SEP]"] + custom_tokens
        trainer = BpeTrainer(special_tokens=all_special, vocab_size=config.vocab_size)
        
        os.makedirs(config.output_dir, exist_ok=True)
        temp_corpus_path = os.path.join(config.output_dir, "temp_zrzy_corpus.json")
        
        with open(temp_corpus_path, "w", encoding="utf-8") as f:
            f.write("\n".join(corpus))
            
        tokenizer.train(files=[temp_corpus_path], trainer=trainer)
        tokenizer.save(config.tokenizer_path)
        print(f"✅ 分词器训练完成，已保存至 {config.tokenizer_path}")

    # ==========================================
    # 5. 精确计算所有图层单行数据的无损最小安全维度
    # ==========================================
    print("\n🔍 正在精确计算满足所有表单行数据的无损最小安全维度...")
    max_global_truth_dim = 0
    max_global_seq_len = 0
    
    for num_dim, unique_row_strings in table_records:
        table_max_seq_len = 0
        for row_str in unique_row_strings:
            if not row_str.strip(): continue
            seq_len = len(tokenizer.encode(row_str).ids)
            if seq_len > table_max_seq_len:
                table_max_seq_len = seq_len
            if seq_len > max_global_seq_len:
                max_global_seq_len = seq_len
                
        table_max_dim = num_dim + table_max_seq_len
        if table_max_dim > max_global_truth_dim:
            max_global_truth_dim = table_max_dim

    print(f"📊 扫描计算完毕！全量数据中，单行所需的最大维度为: {max_global_truth_dim}")

    config.truth_dim = get_optimal_dim(max_global_truth_dim, align=64)
    config.max_seq_len = max_global_seq_len 
    config.save(args.config_path)
    
    print("\n" + "="*65)
    print("✅ [架构第一步] 数据扫描与超参自适应配置完成")
    print(f"📐 物理真值已精准分配维度: {config.truth_dim} 维 (最小冗余 & 100%无损)")
    print("="*65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZrZy Data Builder: Sniff Dims & Train Tokenizer")
    parser.add_argument("--data_dir", type=str, default="/mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/raw_data/")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/output")
    parser.add_argument("--config_path", type=str, default="/mnt/data/yqmeng/ZRZYB/NRE_GIT_V8/output/config.json")
    args = parser.parse_args()
    
    cfg = ModelConfig()
    cfg.data_dir = args.data_dir
    cfg.output_dir = args.output_dir
    cfg.tokenizer_path = os.path.join(args.output_dir, "zrzy_tokenizer.json")
    
    build_tokenizer_and_sniff_dims(cfg, args.data_dir)