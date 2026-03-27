建议先进入项目目录并激活环境：

```bash
source /home/xiaoyang/miniconda3/etc/profile.d/conda.sh

conda activate p2v

cd /home/xiaoyang/workspace/poly2vec_mae_v1/mae_pretrain
```

### 1 预训练启动

- 入口脚本：`scripts/run_pretrain.py`
  
- 具体功能：读取预训练配置并启动 MAE 训练；当 `--gpu` 指定多卡时可自动拉起单机多卡 DDP；支持 `fp32/bf16/fp16`；支持用 `--viz_every` 控制 PNG 可视化输出间隔（按 epoch）。
  
- 配置说明：默认读取 `configs/pretrain_base.yaml`；命令行同名参数覆盖 YAML；未指定参数沿用 YAML。`--viz_every` 默认 `1`（每个 epoch 输出），例如 `10` 表示每 10 个 epoch 输出一次。
  
- 调取示例：
  

```bash
python scripts/run_pretrain.py \

  --config configs/pretrain_base.yaml \

  --gpu 0,1 \

  --epochs 150 \

  --batch_size 512 \

  --precision bf16 \

  --mask_ratio 0.75 \

  --viz_every 10
```

### 2 模型重建可视化评估

- 入口脚本：`scripts/run_eval.py`
  
- 具体功能：加载 `exports` 下的 `encoder_decoder.pth`（enc+dec），对样本做“CFT -> 掩码编码解码 -> 频域重建 -> ICFT可视化”，并输出训练同款布局 PNG。
  
- 配置说明：默认读取 `configs/eval_default.yaml`；关键参数包括 `model_dir`、`mask_ratio`、`precision`、`save_dir`；默认输出目录 `./outputs/viz`；命令行同名参数覆盖 YAML。
  
- 调取示例：
  

```bash
python scripts/run_eval.py \

  --config configs/eval_default.yaml \

  --model_dir ./outputs/exports/mae_20260325_1724 \

  --index 0 \

  --mask_ratio 0.75 \

  --save_dir ./outputs/viz
```

### 3 导出编码器权重

- 入口脚本：`scripts/run_export.py`
  
- 具体功能：将完整 MAE checkpoint 导出为仅 encoder 可用权重，便于下游任务调用。
  
- 配置说明：默认读取 `configs/export_default.yaml`；可由 CLI 覆盖 `mae_ckpt_path/config_path/output_path/precision`；缺失关键路径会报错。
  
- 调取示例：
  

```bash
python scripts/run_export.py \

  --config configs/export_default.yaml \

  --mae_ckpt_path ./outputs/ckpt/20260325_1724/mae_ckpt_150.pth \

  --config_path ./outputs/ckpt/20260325_1724/poly_mae_config.json \

  --output_path ./outputs/exports/encoder_epoch_150.pth \

  --precision bf16
```

### 4 构建三角剖分训练数据

- 入口脚本：`scripts/run_build_dataset.py`
  
- 具体功能：扫描输入目录中的矢量数据，按“每个文件一个任务”并行执行 polygon 三角剖分，并保存为单个 `.pt` 或多个分块 `.pt` 文件。
  
- 配置说明：不依赖 YAML，仅使用 CLI 参数；必须传 `--input_dirs`。  
  常用可选参数：`--output_path`（输出基路径）、`--num_workers`（文件级并行进程数，`<=0` 自动）、`--shard_size_mb`（分块大小，`<=0` 不分块）。
  
- 调取示例：
  

```bash
python scripts/run_build_dataset.py \

  --input_dirs /data/raw/vector_a /data/raw/vector_b \

  --output_path ./data/processed/polygon_triangles_normalized.pt \

  --num_workers 16 \

  --shard_size_mb 500
```

### 5 批量编码产物生成

- 入口脚本：`scripts/run_encode_batch.py`
  
- 具体功能：读取矢量数据并三角剖分，调用 encoder 批量提取 embedding，输出样本文件；支持数据增强与两种输出模式。
  
- 配置说明：不依赖 YAML；核心参数有 `--data_dir`、`--model_dir`、`--device`、`--precision`、`--augment_times`、`--batch_size`、`--output_mode`。  
  

  `output_mode` 说明：  

  `1` => `meta(4) + embedding(N) + triangles(T,3,2)`  

  `2` => `meta(4) + embedding(N)`

- 调取示例：

```bash
python scripts/run_encode_batch.py \

  --data_dir /data/raw \

  --model_dir ./outputs/exports/mae_20260325_1724 \

  --device cuda \

  --precision bf16 \

  --augment_times 10 \

  --batch_size 256 \

  --output_mode 2 \

  --output_path ./data/emb/encoded_samples.pt
```

### 6 运行时库引导

- 入口脚本：`scripts/runtime_bootstrap.py`
  
- 具体功能：在脚本早期阶段补全 CUDA 相关动态库搜索路径（如 `libcudnn.so`），用于提升不同服务器环境下的兼容性。
  
- 配置说明：该脚本由其他入口脚本自动调用，不作为独立业务脚本运行。
  
- 调取示例：
  

```bash
# 无需手动调用，run_pretrain.py / run_eval.py / run_export.py / run_build_dataset.py / run_encode_batch.py 会自动引入。
```
