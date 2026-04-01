主要参数
总特征向量长度512维
单精度float32

0.1B

1. 语义特征提取主干：FT-Transformer（约占 99% 参数）

输入嵌入层 (Feature Tokenizer)：
将 105 个连续数值（标准化后）和离散 ID 映射为高维 Token。

Transformer 核心堆叠 (Backbone)：
网络层数 (Num Layers)：通常为 12 层 纯 Encoder 结构。
隐藏层维度 (Hidden Size)：768 维。
多头注意力 (Attention Heads)：12 头（每个头 64 维）。
前馈网络维度 (FFN Dim)：3072 维（通常是 Hidden Size 的 4 倍）。

输出投影层 (Projection)：
在最后一层，将 768 维的特征压缩映射到我们设定的 semantic_dim = 256 维，准备送入下游潜空间。

参数量粗算：单层 Transformer 大约 700 万参数，12 层加上 Embedding 层，总计大约在 8500 万 - 1 亿参数 之间。

2. 物理无损穿透护甲：单向 cINN（约占 1% 参数）

潜空间总维度 (Total Dim)：512 维（256 维物理硬账本 + 256 维语义特征）。

可逆层深度 (Num Layers)：6 层 SimpleCouplingLayer。

特征变换子网络 (t 网络)：每个 Coupling Layer 内部是一个极简的两层 MLP：Linear(256, 256) -> ReLU -> Linear(256, 256) -> Tanh。

参数量精算：
每层 t 网络的权重矩阵：256 × 256 × 2 = 131,072 个参数。加上偏置项，单层约 13.1 万参数。
6 层总计：131,072× 6 ≈ 78.6 万参数。

