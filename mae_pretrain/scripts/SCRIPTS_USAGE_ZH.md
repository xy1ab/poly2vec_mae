建议先进入项目目录并激活环境：

```bash
source /home/xiaoyang/miniconda3/etc/profile.d/conda.sh

conda activate p2v

cd /home/xiaoyang/workspace/poly2vec_mae/mae_pretrain
```

### 1 预训练启动

- 入口脚本：`scripts/run_pretrain.py`
  
- 具体功能：读取预训练配置并启动 MAE 训练；当 `--gpu` 指定多卡时可自动拉起单机多卡 DDP；支持 `fp32/bf16/fp16`；训练输出统一写入 `<save_dir>/<run_timestamp>/best` 与 `ckpt`；支持用 `--resume_dir` 从既有 run 目录续训。
  
- 配置说明：默认读取 `configs/pretrain_base.yaml`；命令行同名参数覆盖 YAML；未指定参数沿用 YAML。`--eval_every` 控制评测频率；只有评测 epoch 才会输出 val loss、PNG、并更新 `best/` 与 `ckpt/`。
  
- 调取示例：
  

```bash
python scripts/run_pretrain.py \

  --config configs/pretrain_base.yaml \

  --gpu 0,1 \

  --epochs 150 \

  --batch_size 512 \

  --precision bf16 \

  --mask_ratio 0.75 \

  --eval_every 10
```

- 续训示例：

```bash
python scripts/run_pretrain.py \
  --resume_dir ./outputs/ckpt/20260331_1911 \
  --epochs 200 \
  --gpu 0,1 \
  --num_workers 8 \
  --eval_every 5
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
  
- 具体功能：扫描输入目录中的矢量数据，按任务顺序逐个处理（`shp/geojs` 为每文件任务，`gdb` 为每图层任务）；每个任务内部按行分块并行进行 polygon 三角剖分（支持 MultiPolygon 与 donut），并对退化三角形进行过滤（面积过小/近共线）；结果保存为单个 `.pt` 或多个分块 `.pt` 文件。入口脚本已避免通过 `datasets` 包导入链触发额外模块加载；输出文件采用 `torch.save` 序列化（与训练侧 `torch.load` 直接兼容）。如果处理过程中反复出现 `BrokenProcessPool` 之类的原生崩溃信号，可先改用下方的 `run_polygon_diagnosis.py` 对问题样本做定位。
  
- 配置说明：不依赖 YAML，仅使用 CLI 参数；必须传 `--input_dirs`。  
  常用可选参数：`--file_type`（输入类型：`shp/gdb/geojs`，默认 `shp`）、`--layer`（仅 `gdb` 生效，默认 `all`，可指定单层名）、`--output_dir`（输出目录）、`--num_workers`（任务内并行进程数，`<=0` 自动，也兼容 `--num_worker`）、`--rows_per_chunk`（任务内每个分块处理的行数，默认 `2000`）、`--progress_every_chunks`（每合并 N 个分块打印一次摘要，默认 `10`，`<=0` 关闭）、`--shard_size_mb`（分块大小，`<=0` 不分块）、`--min_triangle_area`（最小三角面积阈值，默认 `1e-8`）、`--min_triangle_height`（最小高阈值，默认 `1e-5`）、`--log`（输出三角剖分日志）。  
  当 `file_type=shp` 时，分块命名为 `<shpfilename>_tri_part_<xxxx>.pt`（例如 `hangzhou_tri_part_0001.pt`）。  
  当使用 `--log` 时，日志默认保存到同一目录，命名为 `<shpfilename>_tri.triangulation_log.json`。
  
- 调取示例：
  

```bash
python scripts/run_build_dataset.py \

  --input_dirs /data/raw/vector_a /data/raw/vector_b \

  --output_dir ./data/processed \

  --num_workers 16 \

  --rows_per_chunk 2000 \

  --progress_every_chunks 10 \

  --shard_size_mb 500 \

  --min_triangle_area 1e-8 \

  --min_triangle_height 1e-5 \

  --log
```

```bash
python scripts/run_build_dataset.py \
  --input_dirs /data/boua \
  --file_type gdb \
  --layer BUILDING \
  --output_dir ./data/processed \
  --num_workers 8 \
  --log
```

### 5 图斑原生三角剖分崩溃诊断

- 入口脚本：`scripts/run_polygon_diagnosis.py`
  
- 具体功能：默认采用纯静态检查，不直接执行 `triangle.triangulate(...)`，而是复用当前 `build_dataset_triangle.py` 的修复、归一化与 triangle input 构造路径，评估每个 polygon 样本的潜在原生崩溃风险，并输出风险等级与原因标签（如 `invalid_geometry`、`shell_hole_touching`、`repair_changed_topology`、`triangle_vertices_too_many`）。如果确实需要慢速实探，也可以显式指定 `--mode probe`，按单样本隔离子进程执行真实三角剖分。
  
- 配置说明：不依赖 YAML；必须传 `--input_dirs`。这里可以传入 1 个或多个目录，每个目录都应包含同名 `.shp/.shx/.dbf` 等 shapefile 组件，并且目录下必须只有一个 `.shp` 文件。常用可选参数有 `--mode`（默认 `static`，仅做静态风险检查；`probe` 为慢速真实探测）、`--output_dir`（输出目录，默认 `./outputs/polygon_diagnosis`）、`--timeout_sec`（单样本探测超时，仅 `probe` 模式生效，默认 `20` 秒）、`--num_workers`（仅 `static` 模式生效的并行 worker 数，`<=0` 自动）、`--rows_per_chunk`（仅 `static` 模式生效的目标分块行数，默认 `2000`；如果分块太大导致单个 `.shp` 无法充分并行，脚本仍会自动再切细以利用多个 worker）、`--row_start` / `--row_end`（只诊断部分行，`row_end` 为开区间）。脚本会为每个输入目录各自输出到 `<output_dir>/<任务名>/`，主要包含 `summary.json`、`risk_samples.jsonl` 以及 1 张三联饼图总览：
  索引语义：

  `row_idx`：原始 `.shp` 的行号，也就是 `GeoDataFrame` 中的 geometry 序号；一个 `MultiPolygon` 行仍然只算 1 个 row。

  `sample_index`：将 `MultiPolygon` 按 part 展开之后的样本序号；如果某一行拆成多个 polygon part，它们会共享同一个 `row_idx`，但拥有不同的 `sample_index`。
  
  `summary.json`：汇总扫描范围、状态计数、图斑类型计数、风险标签计数，以及饼图涉及的数值统计。其中：
  `node_count_bucket_counts` 按 sample 统计 `max_triangle_vertices` 的 5 档分布；
  `min_edge_bucket_counts` 按 sample 统计归一化后 `min_normalized_edge_length` 的 5 档分布；
  `connectivity_bucket_counts` 按原始 polygon 行统计 5 类连通性分布；
  `risk_row_polygon_type_counts` 按“至少产出 1 条风险样本的原始行”统计。
  
  `risk_samples.jsonl`：`static` 模式下所有 `medium/high/critical` 风险样本；`probe` 模式下所有 `probe_status != ok` 的样本。每条记录都会附带 `polygon_type` 字段，类型定义为：
  
  `simple`：单 Polygon 且无孔洞。
  
  `multi`：MultiPolygon 且所有 part 都无孔洞。
  
  `donut`：单 Polygon 且只有一个孔洞。
  
  `porous`：单 Polygon 且孔洞数大于 1。
  
  `complex`：不属于前 4 类的复杂 polygon 图斑，主要指带孔洞的 MultiPolygon。

  `pie_overview.png`：单个 PNG，总共 3 个饼图横向排列，分别对应节点数分桶、归一化后最小边长分桶、连通性分桶；每个饼图的 legend 独立放在该饼图下方，并同时显示数量与百分比。
  
- 调取示例：

```bash
python scripts/run_polygon_diagnosis.py \
  --input_dirs /data/raw/testlook \
  --output_dir ./outputs/polygon_diagnosis \
  --mode static \
  --num_workers 8 \
  --rows_per_chunk 1000
```

```bash
python scripts/run_polygon_diagnosis.py \
  --input_dirs /data/raw/testlook /data/raw/hangzhou \
  --output_dir ./outputs/polygon_diagnosis \
  --mode static \
  --num_workers 16 \
  --rows_per_chunk 2000
```

```bash
python scripts/run_polygon_diagnosis.py \
  --input_dirs /data/raw/testlook \
  --output_dir ./outputs/polygon_diagnosis \
  --mode probe \
  --row_start 1000 \
  --row_end 1500 \
  --timeout_sec 30
```

### 6 单图斑可视化诊断

- 入口脚本：`scripts/run_viz_polygon.py`
  
- 具体功能：从单个 `.shp` 中提取一个指定的原始行或展开后的单样本图斑，输出可视化 PNG，并附带一个 JSON 元数据文件。PNG 会同时展示原始 geometry、展开后的 part、`buffer(0)` 修复结果、`_prepare_polygon_candidates()` 的候选图斑，以及归一化后的 candidate，便于直观看到 holes、超长边界、修复分裂等病态性。
  
- 配置说明：必须传 `--input_dir`，并且在 `--row_idx` 与 `--sample_index` 之间二选一。  
  `--row_idx` 适合看原始 shp 第几行；若该行是 `MultiPolygon`，PNG 中会把多个 part 一起画出来。  
  `--sample_index` 适合和 `run_polygon_diagnosis.py` 的输出对齐，直接复核某个高风险样本。  
  可选参数有 `--output_dir`（默认 `./outputs/polygon_viz`）和 `--dpi`。
  
- 调取示例：

```bash
python scripts/run_viz_polygon.py \
  --input_dir /data/raw/testlook \
  --sample_index 6485 \
  --output_dir ./outputs/polygon_viz
```

```bash
python scripts/run_viz_polygon.py \
  --input_dir /data/raw/testlook \
  --row_idx 6358 \
  --output_dir ./outputs/polygon_viz
```

### 7 批量编码产物生成

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

### 8 运行时库引导

- 入口脚本：`scripts/runtime_bootstrap.py`
  
- 具体功能：在脚本早期阶段补全 CUDA 相关动态库搜索路径（如 `libcudnn.so`），用于提升不同服务器环境下的兼容性。
  
- 配置说明：该脚本由其他入口脚本自动调用，不作为独立业务脚本运行。
  
- 调取示例：
  

```bash
# 无需手动调用，run_pretrain.py / run_eval.py / run_export.py / run_build_dataset.py / run_encode_batch.py 会自动引入。
```
