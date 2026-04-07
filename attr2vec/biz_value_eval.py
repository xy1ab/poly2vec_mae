import torch, os, json, glob, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, pairwise_distances
from models import NaturalResourceFoundationModel
from data_loader import NRE_DataPump

device = torch.device("musa" if hasattr(torch, "musa") and torch.musa.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def run_business_value_evaluation():
    print(f"🚀 [地空中心大一统底座] 业务价值智能评估启动 (后端:{device})...")
    
    pump = NRE_DataPump(vocab_path='global_vocab_auto.json')
    rev_chars = {v: k for k, v in pump.shared_chars.items()}; rev_chars[0] = ""
    
    config = {'truth_dim': 256, 'semantic_dim': 256, 'vocab_size': 20000}
    model = NaturalResourceFoundationModel(config).to(device).eval()
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    
    # 🌟 自动扫描搜集 10 个有效图层作为分类目标
    cache_files = glob.glob("cache_*.pt")
    tasks = []
    for cf in cache_files:
        src_name = cf.replace("cache_", "").replace(".pt", "").upper()
        data_all = torch.load(cf, map_location='cpu', weights_only=False)
        for lname, layer in data_all.items():
            if layer['meta']['total_samples'] >= 50:
                tasks.append((layer, f"{src_name}_{lname[:5]}"))
    tasks = tasks[:10] # 最多展示10类避免混淆矩阵过挤
    
    X_physical, X_full, y_labels, text_names = [], [], [], []
    for layer, label in tasks:
        samples = min(200, layer['meta']['total_samples'])
        b = [torch.from_numpy(layer[k][:samples]).to(device) for k in ['cont_int','cont_frac_hi','cont_frac_lo','cont_norm','word_data','char_data']]
        
        with torch.no_grad():
            latent, _ = model(*b)
            X_physical.append(latent[:, :256].cpu().numpy())
            X_full.append(latent.cpu().numpy())
            
        y_labels.extend([label] * samples)
        char_ids = np.round(layer['char_data'][:samples] * 16384.0).astype(np.int64)
        for i in range(samples):
            s = "".join([rev_chars.get(x, "") for x in char_ids[i, 0] if x != 0]) if char_ids.shape[1] > 0 else ""
            text_names.append(s if s else f"纯数值_{label}")

    if not X_full: return
    X_phys_arr, X_full_arr = np.vstack(X_physical), np.vstack(X_full)
    y_arr, text_names = np.array(y_labels), np.array(text_names)
    
    indices = np.arange(len(y_arr))
    idx_train, idx_test, y_train, y_test = train_test_split(indices, y_arr, test_size=0.2, random_state=42, stratify=y_arr)
    
    clf_phys = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
    clf_phys.fit(X_phys_arr[idx_train], y_train)
    acc_phys = accuracy_score(y_test, clf_phys.predict(X_phys_arr[idx_test]))
    
    clf_full = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
    clf_full.fit(X_full_arr[idx_train], y_train)
    y_pred_full = clf_full.predict(X_full_arr[idx_test])
    acc_full = accuracy_score(y_test, y_pred_full)
    
    print(f"\n📉 纯物理几何准确率: {acc_phys*100:.2f}% | 📈 物理+语义全息准确率: {acc_full*100:.2f}%")

    # 绘图
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    bars = ax1.bar(['Physical Only', 'Dual-Track'], [acc_phys*100, acc_full*100], color=['#FF6B6B', '#4ECDC4'], width=0.5)
    for bar in bars:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()-5, f'{bar.get_height():.1f}%', ha='center', color='white', fontsize=16)
    ax1.set_title("Ablation Study Accuracy", fontsize=16)
    
    cm = confusion_matrix(y_test, y_pred_full)
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax2, cbar=False)
    ax2.set_title("Dual-Track Confusion Matrix", fontsize=16)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig("business_value_report_auto.png", dpi=300, facecolor='#111111')
    print("✅ 业务评估大片已出图！保存为: business_value_report_auto.png")

if __name__ == "__main__":
    run_business_value_evaluation()