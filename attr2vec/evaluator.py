import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import torch, os, json, numpy as np
from models import NaturalResourceFoundationModel
from data_loader import NRE_DataPump

class DataAuditor:
    def __init__(self, rev_chars): self.rev_chars = rev_chars
    def render_section(self, source, layer, samples, mode="text"):
        if not samples: return
        print(f"\n{'='*115}\n📊 数据源: {source:<15} | 图层: {layer:<25}")
        print(f"📌 模式: {'🔤 文本一致性' if mode=='text' else '📈 数值高精度'} (展示 {len(samples)} 条)")
        print(f"{'Idx':<4} | {'原始数据 (Original Source)':<50} | {'底座解码 (Decoded Result)':<50}\n{'-'*115}")
        for i, (orig, dec) in enumerate(samples):
            # 放宽打印长度到 48，避免在日志中被强制阻断视觉效果
            o_str = str(orig)[:48] if len(str(orig)) <= 48 else str(orig)[:45] + "..."
            d_str = str(dec)[:48] if len(str(dec)) <= 48 else str(dec)[:45] + "..."
            print(f"{i+1:<4} | {o_str:<50} | {d_str:<50}")
        print(f"{'='*115}")

def evaluate():
    print("🔬 [大一统底座] 全要素无损审计系统启动...")
    pump = NRE_DataPump(vocab_path='global_vocab_auto.json')
    rev_chars = {v: k for k, v in pump.shared_chars.items()}; rev_chars[0] = ""
    auditor = DataAuditor(rev_chars)

    # 🌟 同步 20000 词表配置
    config = {'truth_dim': 256, 'semantic_dim': 256, 'vocab_size': 20000}
    model = NaturalResourceFoundationModel(config).cuda().eval()
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
        print("✅ 成功装载最优生产权重 [best_model.pth]")

    sources = {"AANP": "cache_aanp.pt", "LINCAO": "cache_lincao.pt", "FUJIAN": "cache_fujian.pt"}
    for src_name, cache_path in sources.items():
        if not os.path.exists(cache_path): continue
        data_all = torch.load(cache_path, map_location='cpu', weights_only=False)
        for layer_name, layer in data_all.items():
            try:
                rows = layer['meta']['total_samples']
                if rows == 0: continue
                b = [torch.from_numpy(layer[k][:512]).cuda() for k in ['cont_int','cont_frac_hi','cont_frac_lo','cont_norm','word_data','char_data']]
                with torch.no_grad():
                    latent, _ = model(*b)
                    
                    # 🌟 动态计算本图层的文本长度极值
                    dynamic_max_len = b[5].shape[2] if b[5].shape[1] > 0 else 0
                    
                    # 传入真实长度进行解码
                    dec_cont, dec_word, dec_char = model.decode_physical(
                        latent[:, :256], b[0].shape[1], b[4].shape[1], b[5].shape[1], dynamic_max_len
                    )
                    orig_cont = torch.ldexp(b[1].double() + b[2].double(), b[0].int()) if b[0].shape[1] > 0 else None
                
                if b[0].shape[1] > 0 and orig_cont is not None:
                    auditor.render_section(src_name, layer_name, [(f"{orig_cont[i,0].item():.8f}", f"{dec_cont[i,0].item():.8f}") for i in range(min(15, orig_cont.shape[0]))], mode="num")
                if b[5].shape[1] > 0:
                    auditor.render_section(src_name, layer_name, [("".join([rev_chars.get(x.item(), "") for x in torch.round(b[5][i, 0] * 16384.0).long() if x != 0]), "".join([rev_chars.get(x.item(), "") for x in dec_char[i, 0] if x != 0])) for i in range(min(15, b[5].shape[0]))], mode="text")
            except: continue

    print("\n🎨 正在渲染极客风报告图表...")
    plt.style.use('dark_background') 
    
    if os.path.exists("train_history.json"):
        with open("train_history.json", "r") as f: h = json.load(f)
        plt.figure(figsize=(12, 6))
        epochs = np.arange(len(h['loss']))
        losses = np.array(h['loss'])
        plt.plot(epochs, losses, color='#00ffcc', linewidth=2.5, label='Semantic Knowledge Loss')
        plt.fill_between(epochs, losses, color='#00ffcc', alpha=0.15)
        plt.yscale('log')
        plt.title("🚀 Global Semantic Distillation Convergence", fontsize=18, color='white', pad=20)
        plt.xlabel("Epochs", fontsize=12, color='lightgray'); plt.ylabel("Log Loss", fontsize=12, color='lightgray')
        plt.grid(color='#333333', linestyle='--', linewidth=0.5)
        plt.legend(loc="upper right", facecolor='#111111', edgecolor='none')
        plt.savefig("vis_loss_curve.png", dpi=300, facecolor='#111111', bbox_inches='tight')

    batch_latent = latent.cpu().numpy()[:40, :]
    plt.figure(figsize=(16, 8))
    ax = sns.heatmap(batch_latent, cmap='mako', cbar_kws={'label': 'Energy Activation'})
    plt.axvline(x=256, color='#ff007f', linestyle='--', linewidth=3, label='Physical/Semantic Boundary')
    plt.title("🌌 Dual-Track Latent Manifold (Left: Logic | Right: Knowledge)", fontsize=18, color='white', pad=20)
    ax.tick_params(colors='lightgray')
    plt.savefig("vis_latent_heatmap.png", dpi=300, facecolor='#111111', bbox_inches='tight')
    print("🏁 报告大片生成完毕！")

if __name__ == "__main__": evaluate()