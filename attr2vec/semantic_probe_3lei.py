import torch, glob, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from models import NaturalResourceFoundationModel

device = torch.device("musa" if hasattr(torch, "musa") and torch.musa.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def run_multisource_probe():
    print(f"🔬 正在启动 [跨源大一统] 语义对齐探针 (后端:{device})...")
    
    config = {'truth_dim': 256, 'semantic_dim': 256, 'vocab_size': 20000}
    model = NaturalResourceFoundationModel(config).to(device).eval()
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    
    # 🌟 自动从不同缓存文件中抽取图层
    cache_files = glob.glob("cache_*.pt")
    tasks = []
    for cf in cache_files:
        src_name = cf.replace("cache_", "").replace(".pt", "").upper()
        data_all = torch.load(cf, map_location='cpu', weights_only=False)
        
        # 每个数据源挑选最多 2 个大图层
        valid_layers = [(lname, ldata) for lname, ldata in data_all.items() if ldata['meta']['total_samples'] >= 50]
        valid_layers.sort(key=lambda x: x[1]['meta']['total_samples'], reverse=True)
        for lname, ldata in valid_layers[:2]:
            tasks.append((ldata, f"{src_name}: {lname[:8]}"))
            
    all_semantics, all_labels = [], []
    for layer, label in tasks:
        samples = min(150, layer['meta']['total_samples'])
        b = [torch.from_numpy(layer[k][:samples]).to(device) for k in ['cont_int','cont_frac_hi','cont_frac_lo','cont_norm','word_data','char_data']]
        with torch.no_grad():
            latent, _ = model(*b)
            all_semantics.append(latent[:, 256:].cpu().numpy())
            all_labels.extend([label] * samples)

    if not all_semantics:
        print("❌ 数据不足。")
        return

    X = np.vstack(all_semantics)
    print(f"🎨 多维流形演算中 (跨源样本: {X.shape[0]})...")
    
    tsne = TSNE(n_components=2, perplexity=min(30, X.shape[0]-1), random_state=42)
    X_2d = tsne.fit_transform(X)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 9))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=all_labels, palette="Set2", s=100, alpha=0.85, edgecolor='none')
    plt.title("Cross-Source Semantic Alignment", fontsize=18, color='white', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("semantic_clusters_multisource_auto.png", dpi=300, facecolor='#111111')
    print("✅ 跨源融合星图已保存至: semantic_clusters_multisource_auto.png")

if __name__ == "__main__":
    run_multisource_probe()