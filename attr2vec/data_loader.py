import geopandas as gpd
import pandas as pd
import numpy as np
import pyogrio
import torch
import os
import warnings
warnings.filterwarnings('ignore')
from lincao_dict import FIELD_METADATA, LINCAO_CODE_DICT

def load_and_preprocess_gdb(gdb_path, layer_name, cache_path="data_cache.pt"):
    # 🌟 护甲机制 1：缓存自愈与健康度核验
    if os.path.exists(cache_path):
        print(f"⚡ 检测到数据缓存，正在核验数据健康度...")
        try:
            data_dict = torch.load(cache_path)
            if np.isnan(data_dict['raw_64']).any() or np.isnan(data_dict['cont_norm']).any():
                print("⚠️ 警告：检测到旧版缓存中残留 NaN 毒数据！正在自动销毁并重建...")
                os.remove(cache_path)
            else:
                print("✅ 缓存健康 (0 NaN)，瞬间载入成功！")
                return data_dict
        except Exception as e:
            print(f"⚠️ 缓存读取异常 ({e})，正在重建...")
            os.remove(cache_path)

    print(f"\n🚀 [之江实验室] 正在解析 GDB 并执行全要素强制脱毒与无损压缩...")
    gdf = gpd.read_file(gdb_path, layer=layer_name, engine="pyogrio", use_arrow=True)
    df = pd.DataFrame(gdf).drop(columns=['geometry'], errors='ignore')
    
    # 暴力清除所有空白字符
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    
    upper_meta = {k.upper(): k for k in FIELD_METADATA.keys()}
    valid_cols = [col for col in df.columns if col.upper() in upper_meta]
    df = df[valid_cols]

    continuous_cols, categorical_cols = [], []
    for col in valid_cols:
        if FIELD_METADATA[upper_meta[col.upper()]]['decimals'] > 0:
            continuous_cols.append(col)
        else:
            categorical_cols.append(col)

    # 🌟 护甲机制 2：连续字段多重脱毒与【稠密伪装压缩】
    for col in continuous_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
    raw_64 = df[continuous_cols].values.astype(np.float64)
    raw_64 = np.nan_to_num(raw_64, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 将大数值除以 1,000,000，伪装成大模型喜欢的 [0, 1] 稠密区间
    cont_int = (np.trunc(raw_64) / 1_000_000.0).astype(np.float32)
    cont_frac = (raw_64 - np.trunc(raw_64)).astype(np.float32)
    
    std = np.std(raw_64, axis=0)
    std[std == 0] = 1.0
    cont_norm = ((raw_64 - np.mean(raw_64, axis=0)) / std).astype(np.float32)
    cont_norm = np.nan_to_num(cont_norm, nan=0.0)

    # 🌟 护甲机制 3：离散字段绝对映射与【百级压缩】
    cat_list = []
    for col in categorical_cols:
        mapping = LINCAO_CODE_DICT.get(col.upper(), {})
        encoded, _ = pd.factorize(df[col].astype(str).map(mapping).fillna(df[col].astype(str)))
        cat_list.append(encoded)
        
    # 分类 ID 除以 100.0，进一步平滑 512 维特征空间
    cat_data = (np.stack(cat_list, axis=1) / 100.0).astype(np.float32)

    data_dict = {
        'cont_norm': cont_norm, 'cont_int': cont_int, 'cont_frac': cont_frac,
        'cat_data': cat_data, 'raw_64': raw_64,
        'cat_cardinalities': [len(np.unique(c)) for c in cat_list],
        'cont_names': continuous_cols, 'cat_names': categorical_cols
    }
    
    torch.save(data_dict, cache_path)
    print(f"✅ 脱毒与压缩预编译完成，全新健康缓存已保存至 {cache_path}")
    return data_dict