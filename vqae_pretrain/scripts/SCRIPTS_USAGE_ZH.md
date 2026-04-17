建议先进入项目目录并激活环境：

```bash
source /home/xiaoyang/miniconda3/etc/profile.d/conda.sh
conda activate p2v
cd /home/xiaoyang/workspace/poly2vec_mae/vqae_pretrain
```

### 1 预训练启动

- 入口脚本：`scripts/run_pretrain.py`

- 具体功能：读取 `configs/pretrain_base.yaml` 与命令行覆盖参数，启动 `conv+ViT encoder + EMA VQ + attention+conv decoder` 的 VQAE 训练。训练输出统一写到 `<save_dir>/<run_name>/best` 与 `ckpt`。如果 `ckpt/train_state_a.pth` 或 `ckpt/train_state_b.pth` 已存在，使用同一条命令会自动续训。

- 当前实现说明：
  - encoder token 数由 `stem_strides` 的总下采样倍率决定，不再使用 patchify 作为 token 化入口；
  - 训练损失直接对完整频域图 `Mag/Cos/Sin` 计算，不再使用 patch-based loss；
  - 输入频域图会自动 padding 到 `latent_stride` 的整数倍。
  - decoder 内部先预测两个 raw 通道 `(mag_raw, phase_raw)`，再统一映射成受约束的三通道输出：
    - `mag = softplus(mag_raw)`
    - `cos = cos(phase_raw)`
    - `sin = sin(phase_raw)`
    这让重建结果天然满足 `mag >= 0` 与 `cos^2 + sin^2 = 1`。

- warmup 机制：
  - `warmup_epochs`：学习率 warmup，从训练开始即生效。
  - `vq_warmup_epochs`：VQ warmup。在这段时期内完全按连续 AE 训练，不启用 quantizer。
  - `vq_beta_warmup_epochs`：在 VQ warmup 结束后启动，将 `vq_beta` 从 `0` 线性增加到目标值。

- VQ 训练关键参数：
  - `codebook_size`：码本大小。
  - `code_dim`：量化空间维度。
  - `vq_beta`：commitment loss 权重。
  - `vq_decay`：EMA 更新衰减系数。配置文件中已附设置建议。
  - `vq_init_max_vectors`：用于初始化 codebook 的最大 latent 向量数。
  - `vq_kmeans_iters`：工程版 GPU K-means 的迭代次数。

- 日志中的核心 VQ 指标：
  - `VQ`：量化附加损失项。
  - `Perplexity`：码本使用多样性。
  - `ActiveCodes`：当前批次或当前评测窗口内实际被使用到的 code 数量。

- 调取示例：

```bash
python scripts/run_pretrain.py \
  --config configs/pretrain_base.yaml \
  --run_name 20260417_run1 \
  --gpu 0 \
  --epochs 200 \
  --batch_size 256 \
  --precision bf16 \
  --eval_every 5
```

- 续训示例：

```bash
python scripts/run_pretrain.py \
  --config configs/pretrain_base.yaml \
  --run_name 20260417_run1 \
  --epochs 240 \
  --gpu 0 \
  --eval_every 5
```

### 2 重建评估与可视化

- 入口脚本：`scripts/run_eval.py`

- 具体功能：加载 `vqae_best.pth`，对一个指定样本执行 `三角样本 -> 频域图 -> VQAE 重建 -> 频域结果 -> ICFT 可视化`，并输出一张 PNG。

- 主要输入：
  - `--model_dir`：包含 `vqae_best.pth` 与 `config.yaml` 的目录。
  - `--data_dir`：包含三角样本 shard 的目录。
  - `--row_index`：跨 shard 的全局样本序号。

- 输出内容：
  - 原始 `Mag/Cos/Sin`
  - 重建 `Mag/Cos/Sin`
  - 原始空间域 raster
  - 重建后的空间域 ICFT
  - 图中会额外标注 `Perplexity` 与 `ActiveCodes`

- 调取示例：

```bash
python scripts/run_eval.py \
  --model_dir ./outputs/ckpt/20260407_1200 \
  --data_dir ./data/processed/hangzhou \
  --row_index 0 \
  --save_dir ./outputs/eval
```

### 3 批量量化为离散索引

- 入口脚本：`scripts/run_quantize_batch.py`

- 具体功能：读取一批三角样本，经过 `encoder -> quantizer` 得到每个样本的二维离散索引网格，并保存为一个 `.pt` 文件。

- 输入：
  - `--model_dir`：包含 `vqae_best.pth` 与 `config.yaml`
  - `--input_dir`：包含三角样本 shard 的目录

- 输出 `.pt` 格式：
  - `metadata`
  - `samples`
    - `triangles`
    - `indices`

- `indices` 语义：
  - 每个样本保存为一个 `[H_lat, W_lat]` 的整数张量
  - 每个整数代表该 latent 位置所选中的 codebook 序号

- 调取示例：

```bash
python scripts/run_quantize_batch.py \
  --model_dir ./outputs/ckpt/20260407_1200 \
  --input_dir ./data/processed/hangzhou \
  --output_path ./outputs/quantized/hangzhou_indices.pt \
  --batch_size 64
```

### 4 从离散索引解码重建

- 入口脚本：`scripts/run_decode_indices.py`

- 具体功能：读取 `run_quantize_batch.py` 生成的量化结果，再结合 `quantizer.pth + decoder.pth + config.yaml`，将离散索引还原成频域重建结果。

- 输入：
  - `--indices_path`：量化输出 `.pt`
  - `--model_dir`：包含 `decoder.pth`、`quantizer.pth`、`config.yaml`

- 输出 `.pt` 格式：
  - `metadata`
  - `samples`
    - `triangles`
    - `indices`
    - `freq_real`
    - `freq_imag`

- 调取示例：

```bash
python scripts/run_decode_indices.py \
  --indices_path ./outputs/quantized/hangzhou_indices.pt \
  --model_dir ./outputs/exports/vqae_bundle \
  --output_path ./outputs/decoded/hangzhou_freq.pt
```

### 5 导出 VQAE 组件

- 入口脚本：`scripts/run_export.py`

- 具体功能：从完整的 `vqae_best.pth` 导出下游任务所需的组件文件。

- 输出目录内容：
  - `vqae_best.pth`
  - `encoder.pth`
  - `decoder.pth`
  - `quantizer.pth`
  - `config.yaml`

- 其中：
  - `decoder.pth` 包含 `post_vq_proj + decoder`
  - `quantizer.pth` 包含完整 quantizer 状态，包括 codebook、`cluster_size`、`embed_avg` 与初始化状态

- 调取示例：

```bash
python scripts/run_export.py \
  --vqae_ckpt_path ./outputs/ckpt/20260407_1200/best/vqae_best.pth \
  --config_path ./outputs/ckpt/20260407_1200/best/config.yaml \
  --output_dir ./outputs/exports/vqae_bundle \
  --precision fp32
```

### 6 码本使用统计

- 入口脚本：`scripts/run_codebook_stats.py`

- 具体功能：遍历一批三角样本，统计当前 VQAE 在整批数据上的码本使用情况。

- 输出 JSON 中的关键字段：
  - `active_codes`：至少被用过一次的 code 数量
  - `code_usage_ratio`：`active_codes / total_codes`
  - `avg_batch_perplexity`：按 batch 平均的 perplexity
  - `global_perplexity`：按全局 usage 统计得到的 perplexity

- 典型用途：
  - 判断 codebook 是否塌缩
  - 比较不同 `codebook_size / vq_beta / vq_decay` 的使用健康度

- 调取示例：

```bash
python scripts/run_codebook_stats.py \
  --model_dir ./outputs/ckpt/20260407_1200 \
  --input_dir ./data/processed/hangzhou \
  --output_path ./outputs/codebook_stats/hangzhou_stats.json \
  --batch_size 64
```

### 7 MUSA 入口说明

- 入口脚本：`scripts/run_pretrain_musa.py`

- 当前状态：仅保留结构对齐的占位版本，便于后续手动适配；第一阶段不保证可直接训练。

- 使用建议：优先以 CUDA 版本为主链路开发和验证，MUSA 后续再做针对性迁移。
