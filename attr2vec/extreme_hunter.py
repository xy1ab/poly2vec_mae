import torch, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from models import NaturalResourceFoundationModel
from data_loader import NRE_DataPump

def run_ultimate_probe():
    print("🚀 [大一统底座] 终极自证程序启动...")
    
    pump = NRE_DataPump(vocab_path='global_vocab_auto.json')
    rev_chars = {v: k for k, v in pump.shared_chars.items()}; rev_chars[0] = ""
    config = {'truth_dim': 256, 'semantic_dim': 256, 'vocab_size': 20000}
    model = NaturalResourceFoundationModel(config).cuda().eval()
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    
    # ==========================================
    # 任务 1：修复版极限猎手 (精准锁定最大值所在列)
    # ==========================================
    print("\n" + "="*80)
    print("🛸 正在执行: 全库天文数字精准对账...")
    cache_files = ["cache_aanp.pt", "cache_lincao.pt", "cache_fujian.pt"]
    global_max_val = -1.0; max_val_info = {}

    for file in cache_files:
        if not os.path.exists(file): continue
        data_all = torch.load(file, map_location='cpu', weights_only=False)
        for layer_name, layer in data_all.items():
            if layer['meta']['total_samples'] == 0: continue
            if layer['cont_int'].shape[1] > 0:
                orig_cont = np.ldexp(
                    layer['cont_frac_hi'].astype(np.float64) + layer['cont_frac_lo'].astype(np.float64), 
                    layer['cont_int'].astype(np.int32)
                )
                local_max = np.max(orig_cont)
                if local_max > global_max_val:
                    global_max_val = local_max
                    # 🌟 修复 Bug: 同时记录最大值所在的 行(idx_row) 和 列(idx_col)
                    idx_row, idx_col = np.unravel_index(np.argmax(orig_cont), orig_cont.shape)
                    max_val_info = {'file': file, 'layer': layer_name, 'idx_row': idx_row, 'idx_col': idx_col, 'val': local_max, 
                                    'raw_data': {k: layer[k][idx_row:idx_row+1] for k in ['cont_int','cont_frac_hi','cont_frac_lo','cont_norm','word_data','char_data']}}

    if max_val_info:
        print(f"🔥 发现全库极值点!")
        print(f"📍 来源: {max_val_info['file']} -> 图层: {max_val_info['layer']} (列索引: {max_val_info['idx_col']})")
        print(f"📏 原始物理真值 (100%未截断): {max_val_info['val']:.8f}")
        
        b = [torch.from_numpy(max_val_info['raw_data'][k]).cuda() for k in ['cont_int','cont_frac_hi','cont_frac_lo','cont_norm','word_data','char_data']]
        with torch.no_grad():
            latent, _ = model(*b)
            dec_cont, _, _ = model.decode_physical(latent[:, :256], b[0].shape[1], b[4].shape[1], b[5].shape[1], b[5].shape[2] if b[5].shape[1]>0 else 0)
            # 🌟 修复 Bug: 精准提取对应列的解码值
            dec_val = dec_cont[0, max_val_info['idx_col']].item()
        
        print(f"🧠 底座潜空间解码: {dec_val:.8f}")
        if abs(max_val_info['val'] - dec_val) < 1e-1: print("✅ 结论: 千亿级天文数字重构完美，双单精度切分生效！绝对没有作弊！")
        else: print("❌ 结论: 存在偏差。")
    print("="*80 + "\n")

    # ==========================================
    # 任务 2：10类别跨源大乱斗星图
    # ==========================================
    print("🎨 正在执行: [AANP x LINCAO x FUJIAN] 跨域大乱斗聚类...")
    tasks = [
        ("cache_aanp.pt", "TABULAR_DEFAULT", "AANP: Global Places"),
        ("cache_lincao.pt", "LCXZ_TEST01_GYL", "LINCAO: Forest Metrics"),
        ("cache_lincao.pt", "LCXZ_TEST01_XZ", "LINCAO: Admin Bounds"),
        ("cache_fujian.pt", "民用机场", "FUJIAN: Airports"),
        ("cache_fujian.pt", "铁路", "FUJIAN: Railways"),
        ("cache_fujian.pt", "高速公路", "FUJIAN: Expressways"),
        ("cache_fujian.pt", "山脉", "FUJIAN: Mountains"),
        ("cache_fujian.pt", "河流", "FUJIAN: Rivers"),
        ("cache_fujian.pt", "水库注记", "FUJIAN: Reservoirs"),
        ("cache_fujian.pt", "县级行政区", "FUJIAN: County Admin")
    ]
    
    all_semantics = []; all_labels = []
    
    for file_path, layer_name, label in tasks:
        if not os.path.exists(file_path): continue
        data_all = torch.load(file_path, map_location='cpu', weights_only=False)
        if layer_name not in data_all: continue
        
        layer = data_all[layer_name]
        samples = min(150, layer['meta']['total_samples'])
        if samples == 0: continue
        
        b = [torch.from_numpy(layer[k][:samples]).cuda() for k in ['cont_int','cont_frac_hi','cont_frac_lo','cont_norm','word_data','char_data']]
        with torch.no_grad():
            latent, _ = model(*b)
            all_semantics.append(latent[:, 256:].cpu().numpy())
        all_labels.extend([label] * samples)

    X = np.vstack(all_semantics)
    print(f"📊 提取特征总数: {X.shape[0]} 个跨域样本 (涵盖 10 大类别)。")
    
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(16, 12))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=all_labels, palette="tab10", s=90, alpha=0.85, edgecolor='none')
    plt.title("Ultimate Cross-Source Semantic Alignment (10-Class Manifold)", fontsize=22, color='white', pad=20)
    plt.grid(color='#333333', linestyle='--', linewidth=0.5)
    
    legend = plt.legend(title="Cross-Domain Ontology", bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#111111', edgecolor='none')
    plt.setp(legend.get_texts(), color='w'); plt.setp(legend.get_title(), color='w', fontsize=14)
    
    plt.tight_layout()
    plt.savefig("ultimate_multisource_clusters.png", dpi=300, facecolor='#111111')
    print("✅ 终极大片已出片！星图保存至: ultimate_multisource_clusters.png")

if __name__ == "__main__":
    run_ultimate_probe()