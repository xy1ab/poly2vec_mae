import faiss
import numpy as np
import time
# 1. 模拟数据准备
d = 1024          # 向量维度
n_data = 100000   # 数据库向量数量
n_query = 5       # 查询向量数量

# 生成随机数据 (实际使用时替换为你的 embedding)
xb = np.random.random((n_data, d)).astype('float32')
xq = np.random.random((n_query, d)).astype('float32')

# 推荐：进行 L2 归一化，这样聚类或搜索就是基于余弦相似度
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)

# 2. 配置索引参数
nlist = 100       # 聚类中心的数量（根据数据量调整，通常为 sqrt(N) 到 10*sqrt(N)）
m = 64            # 每个向量被分解成的子向量个数 (1024 必须能被 m 整除)
bits = 8          # 每个子向量量化后的位数（通常选 8）

# 3. 创建索引 (IVFPQ)
# IndexFlatL2 是基础量化器，用于划分 cell
quantizer = faiss.IndexFlatL2(d) 
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

# 4. 训练与添加数据
# IVFPQ 必须先训练才能添加数据
print("开始训练索引...")
index.train(xb) 
print(f"索引是否已训练: {index.is_trained}")

index.add(xb)
print(f"索引中的向量总数: {index.ntotal}")

# 5. 执行搜索
k = 4             # 返回最近邻的数量
index.nprobe = 10 # 搜索时检查的 cell 数量（值越大精度越高但速度越慢）
t_s = time.time()
distances, indices = index.search(xq, k)
t_e = time.time()
print(f"搜索耗时: {t_e - t_s:.4f} 秒")
# 6. 结果展示
print("\n查询结果 (Top 4 索引):")
print(indices)
print("\n查询结果 (距离):")
print(distances)