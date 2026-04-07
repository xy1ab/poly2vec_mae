import torch, os, numpy as np, glob
from models import NaturalResourceFoundationModel
from data_loader import NRE_DataPump

# 🌟 南湖平台 MUSA 兼容层
device = torch.device("musa" if hasattr(torch, "musa") and torch.musa.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

print(f"📜 [极限文本猎手] 启动！当前计算后端: {device}")
pump = NRE_DataPump(vocab_path='global_vocab_auto.json')
rev_chars = {v: k for k, v in pump.shared_chars.items()}; rev_chars[0] = ""

config = {'truth_dim': 256, 'semantic_dim': 256, 'vocab_size': 20000}
model = NaturalResourceFoundationModel(config).to(device).eval()
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))

# 🌟 自动扫描所有缓存文件
cache_files = glob.glob("cache_*.pt")
global_max_len = -1; max_str_info = {}

for file in cache_files:
    data_all = torch.load(file, map_location='cpu', weights_only=False)
    for layer_name, layer in data_all.items():
        if layer['meta']['total_samples'] == 0 or layer['char_data'].shape[1] == 0: continue
        
        char_ids = np.round(layer['char_data'] * 16384.0).astype(np.int64)
        valid_lens = np.sum(char_ids > 0, axis=2) 
        local_max_len = np.max(valid_lens)
        if local_max_len > global_max_len:
            global_max_len = local_max_len
            idx_row, idx_col = np.unravel_index(np.argmax(valid_lens), valid_lens.shape)
            max_str_info = {'file': file, 'layer': layer_name, 'idx_row': idx_row, 'len': local_max_len, 
                            'raw_data': {k: layer[k][idx_row:idx_row+1] for k in ['cont_int','cont_frac_hi','cont_frac_lo','cont_norm','word_data','char_data']}}

if max_str_info:
    print(f"📍 来源: {max_str_info['file']} -> 图层: {max_str_info['layer']}")
    print(f"📏 最大字符数: {max_str_info['len']} 个字")
    
    b = [torch.from_numpy(max_str_info['raw_data'][k]).to(device) for k in ['cont_int','cont_frac_hi','cont_frac_lo','cont_norm','word_data','char_data']]
    with torch.no_grad():
        latent, _ = model(*b)
        _, _, dec_char = model.decode_physical(latent[:, :256], b[0].shape[1], b[4].shape[1], b[5].shape[1], b[5].shape[2])
        orig_char_ids = torch.round(b[5][0, 0] * 16384.0).long()
        orig_str = "".join([rev_chars.get(x.item(), "") for x in orig_char_ids if x != 0])
        dec_str = "".join([rev_chars.get(x.item(), "") for x in dec_char[0, 0] if x != 0])
        
    print(f"🖋️ 原始超长输入: {orig_str}")
    print(f"🧠 底座解码输出: {dec_str}")