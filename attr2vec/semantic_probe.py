import torch, glob, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from models import NaturalResourceFoundationModel

device = torch.device("musa" if hasattr(torch, "musa") and torch.musa.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def run_semantic_probe():
    print(f"🔬 正在启动大一统底座语义探针 (智能识别模式, 后端:{device})...")
    
    config = {'truth_dim': 256, 'semantic_dim': 256, 'vocab_size': 20000}
    model = NaturalResourceFoundationModel(config).to(device).eval()
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    
    # 🌟 自动收集样本量最大的图层
    cache_files = glob.glob("cache_*.pt")
    available_layers = []
    
    for cf in cache_files:
        data_all = torch.load(cf, map_location='cpu', weights_only=False)
        for lname, layer in data_all.items():
            if layer['meta']['total_samples'] >= 50:
                available_layers.append((layer, f"{os.path.basename(cf)[:8]}_{lname}"))
                
    # 挑选最大的 8 个图层
    available_layers.sort(key=lambda x: x[0]['meta']['total_samples'], reverse=True)
    target_layers = available_layers[:8]
    
    all_semantics, all_labels = [], []
    for layer, label_name in target_layers:
        samples = min(150, layer['meta']['total_samples'])
        b = [torch.from_numpy(layer[k][:samples]).to(device) for k in ['cont_int','cont_frac_hi','cont_frac_lo','cont_norm','word_data','char_data']]
        
        with torch.no_grad():
            latent, _ = model(*b)
            semantic_vecs = latent[:, 256:].cpu().numpy()
            
        all_semantics.append(semantic_vecs)
        all_labels.extend([label_name] * samples)

    if not all_semantics:
        print("❌ 未发现充足的图层数据，无法绘图。")
        return

    X = np.vstack(all_semantics)
    print(f"🎨 正在进行极大规模 t-SNE 降维流形演算 (样本数: {X.shape[0]})...")
    
    tsne = TSNE(n_components=2, perplexity=min(30, X.shape[0]-1), random_state=42)
    X_2d = tsne.fit_transform(X)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 10))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=all_labels, palette="tab10", s=80, alpha=0.85, edgecolor='none')
    plt.title("Semantic Encoding Manifold (Auto-Selected Layers)", fontsize=20, color='white', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("semantic_clusters_auto.png", dpi=300, facecolor='#111111')
    print("✅ 大规模探测完成！聚类星图已保存至: semantic_clusters_auto.png")

if __name__ == "__main__":
    run_semantic_probe()