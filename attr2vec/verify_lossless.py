import os, torch, traceback, glob
from data_loader import NRE_DataPump
from models import NaturalResourceFoundationModel

# 🌟 自动扫描数据存放目录
RAW_DATA_DIR = "./raw_data"
DATA_SOURCES = []
for file_path in glob.glob(os.path.join(RAW_DATA_DIR, "*")):
    if file_path.endswith('.csv') or file_path.endswith('.gdb'):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        DATA_SOURCES.append({"file": file_path, "cache": f"cache_{base_name}.pt"})

# MUSA 兼容
device = torch.device("musa" if hasattr(torch, "musa") and torch.musa.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def verify_all_sources():
    print(f"🚀 [大一统双轨制底座] —— 启动全域数据源防爆验证 (计算后端: {device})\n")
    print("=" * 60)
    
    pump = NRE_DataPump(vocab_path='global_vocab_auto.json', max_seq_len=64) # 同步为64
    rev_chars = {v: k for k, v in pump.shared_chars.items()}
    rev_chars[0] = "" 
    
    config = {'truth_dim': 256, 'semantic_dim': 256, 'vocab_size': 20000}
    model = NaturalResourceFoundationModel(config).to(device)
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    model.eval()

    for source in DATA_SOURCES:
        file_path = source["file"]
        cache_path = source["cache"]
        
        print(f"\n📂 正在切入数据源: [{file_path}]")
        if not os.path.exists(cache_path): 
            print(f"⚠️ 缓存 {cache_path} 未就绪，已跳过。")
            continue
            
        try:
            layers_data = torch.load(cache_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"🚨 缓存读取失败: {e}")
            continue
        
        for layer_name, data in layers_data.items():
            print(f"\n   🔹 正在体检图层: [{layer_name}]")
            total_samples = data['meta']['total_samples']
            if total_samples == 0: continue
            n_cont, n_word, n_char = data['cont_int'].shape[1], data['word_data'].shape[1], data['char_data'].shape[1]
            
            print(f"      📊 台账规模: 总样本数 = {total_samples} | 特征维度: 连续={n_cont}, 分类={n_word}, 文本={n_char}")

            batch_size = 2048
            max_cont_diff = 0.0
            word_pass_all, char_pass_all = True, True
            largest_val_orig, largest_val_dec, largest_val_diff = 0.0, 0.0, 0.0
            theoretical_err_limit_max = 0.0
            max_char_len = -1
            longest_str_orig, longest_str_dec = "", ""
            
            with torch.no_grad():
                for i in range(0, total_samples, batch_size):
                    b_int = torch.tensor(data['cont_int'][i:i+batch_size]).to(device)
                    b_frac_hi = torch.tensor(data['cont_frac_hi'][i:i+batch_size]).to(device)
                    b_frac_lo = torch.tensor(data['cont_frac_lo'][i:i+batch_size]).to(device)
                    b_norm = torch.tensor(data['cont_norm'][i:i+batch_size]).to(device)
                    b_word = torch.tensor(data['word_data'][i:i+batch_size]).to(device)
                    b_char = torch.tensor(data['char_data'][i:i+batch_size]).to(device)
                    
                    try:
                        latent, _ = model(b_int, b_frac_hi, b_frac_lo, b_norm, b_word, b_char)
                        v_attr = latent[:, :config['truth_dim']]
                        dec_cont, dec_word, dec_char = model.decode_physical(v_attr, n_cont, n_word, n_char, b_char.shape[2] if n_char>0 else 0)

                        if n_cont > 0:
                            original_cont = torch.ldexp(b_frac_hi.double() + b_frac_lo.double(), b_int.int())
                            diff = torch.abs(dec_cont - original_cont)
                            if diff.max().item() > max_cont_diff: max_cont_diff = diff.max().item()
                            batch_limit_max = (torch.abs(original_cont) * (2 ** -47) + 1e-8).max().item()
                            if batch_limit_max > theoretical_err_limit_max: theoretical_err_limit_max = batch_limit_max

                        if n_word > 0:
                            if not torch.equal(dec_word, torch.round(b_word * 16384.0).long()): word_pass_all = False
                                
                        if n_char > 0:
                            target_char = torch.round(b_char * 16384.0).long()
                            if not torch.equal(dec_char, target_char): char_pass_all = False
                            
                    except Exception as e:
                        print(f"      ❌ 推理失败: {e}")
                        cont_pass_all = word_pass_all = char_pass_all = False
                        break

            if 'e' not in locals(): 
                cont_pass_all = max_cont_diff <= theoretical_err_limit_max
                if cont_pass_all and word_pass_all and char_pass_all: print("      ✅ 该图层无损还原通过！")
                else: print(f"      🚨 图层被击穿！连续无损: {cont_pass_all} | 字典无损: {word_pass_all} | 文本无损: {char_pass_all}")

    print("\n" + "=" * 60 + "\n🏁 极限高压质检完毕。")

if __name__ == "__main__":
    verify_all_sources()