import torch
emb = torch.load('./data/encoded_samples_20260324_1910.pt', map_location='cpu')
# 抽取前5个样本的Embedding
for i in range(5):
    v = emb[i]['embedding']
    print(f"样本 {i} | 均值: {v.mean():.4f} | 标准差: {v.std():.4f} | 前5位: {v[:5]}")