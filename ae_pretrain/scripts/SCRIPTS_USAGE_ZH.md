建议先进入项目目录并激活环境：

```bash
source /home/xiaoyang/miniconda3/etc/profile.d/conda.sh

conda activate p2v

cd /home/xiaoyang/workspace/poly2vec_mae/ae_pretrain
```

### 1 预训练启动

- 入口脚本：`scripts/run_pretrain.py`
  
- 具体功能：读取预训练配置并启动 AE 训练；当 `--gpu` 指定多卡时可自动拉起单机多卡 DDP；支持 `fp32/bf16/fp16`；训练输出统一写入 `<save_dir>/<run_timestamp>/best` 与 `ckpt`；支持用 `--resume_dir` 从既有 run 目录续训。
  
- 配置说明：默认读取 `configs/pretrain_base.yaml`；命令行同名参数覆盖 YAML；未指定参数沿用 YAML。`--eval_every` 控制评测频率；只有评测 epoch 才会输出 val loss、PNG、并更新 `best/` 与 `ckpt/`。当前关键参数会严格校验：`--val_ratio` 必须在 `(0,1)`，`--augment_times` 必须 `>= 1`；不再对非法值做静默钳制。AE 训练始终采用无掩码全图重建，不再暴露 `--mask_ratio` 与 `--loss_mode`。
  
- 调取示例：
  

```bash
python scripts/run_pretrain.py \

  --config configs/pretrain_base.yaml \

  --gpu 0,1 \

  --epochs 150 \

  --batch_size 512 \

  --precision bf16 \

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
  
- 具体功能：加载 `exports` 下的完整 AE 权重，对样本做“CFT -> AE 重建 -> 频域重建 -> ICFT 可视化”，并输出训练同款布局 PNG。
  
- 配置说明：不再依赖 YAML。脚本会直接从 `--model_dir` 中自动寻找模型配置和名称中包含 `ae`、`autoencoder`、`mae` 或 `encoder_decoder` 的 `.pth` 文件；从 `--data_dir` 中寻找三角剖分 shard。样本定位使用 `--row_index`，语义是“跨 shard 的全局样本序号”。数据目录中若存在且仅存在一个合法 `.manifest.json`，则优先按 manifest 中声明的 shard 顺序取样；若 manifest 缺失、非法或存在多个，则会打印 warning 并退回到当前目录下按文件名排序的 `*.pt`。当前只支持 `torch.save` 生成的 `.pt` shard；旧 pickle shard 不再兼容。
  
- 调取示例：
  

```bash
python scripts/run_eval.py \

  --model_dir ./outputs/exports/ae_20260407_1200 \

  --data_dir ./data/processed/hangzhou \

  --row_index 0 \

  --save_dir ./outputs/viz
```

### 3 导出编码器权重

- 入口脚本：`scripts/run_export.py`
  
- 具体功能：将完整 AE checkpoint 导出为仅 encoder 可用权重，便于下游任务调用。
  
- 配置说明：默认读取 `configs/export_default.yaml`；可由 CLI 覆盖 `ae_ckpt_path/config_path/output_path/precision`；缺失关键路径会报错。
  
- 调取示例：
  

```bash
python scripts/run_export.py \

  --config configs/export_default.yaml \

  --ae_ckpt_path ./outputs/ckpt/20260407_1200/best/ae_best.pth \

  --config_path ./outputs/ckpt/20260407_1200/best/poly_ae_config.json \

  --output_path ./outputs/exports/encoder_epoch_150.pth \

  --precision bf16
```

### 4 构建三角剖分训练数据

- 入口脚本：`scripts/run_build_dataset.py`
  
- 具体功能：扫描输入目录中的矢量数据，按任务顺序逐个处理（`shp/geojs` 为每文件任务，`gdb` 为每图层任务）；每个任务内部按行分块并行执行新的 row 级三角剖分流程。当前语义下，`1 个 shp-row` 严格对应 `1 个输出样本`：如果这一行是 `MultiPolygon`，会先整体去中心化并按整行 bbox 保长宽比归一化，再对各个 part 做严格过滤与逐 part 三角剖分，最后将所有保留 part 的三角形合并为同一条训练样本；不会再把一个 `MultiPolygon` 拆成多条训练样本。处理流程为：`读入 -> 整体去中心化+归一化 -> part 过滤 -> 判断是否进入隔离子进程 -> 逐 part 三角剖分后合并 -> 退化过滤 -> 增量写 shard`。入口脚本已避免通过 `datasets` 包导入链触发额外模块加载；输出文件采用 `torch.save` 序列化（与训练侧 `torch.load` 直接兼容）。如果处理过程中需要定位被过滤或被 dropped 的 row，可配合下方的 `run_polygon_diagnosis.py` 与 `run_viz_polygon_triangulation.py`。
  
- 配置说明：不依赖 YAML，仅使用 CLI 参数；必须传 `--input_dirs`。  
  常用可选参数：`--file_type`（输入类型：`shp/gdb/geojs`，默认 `shp`）、`--layer`（仅 `gdb` 生效，默认 `all`，可指定单层名）、`--output_dir`（输出目录）、`--num_workers`（任务内并行进程数，`<=0` 自动，也兼容 `--num_worker`）、`--rows_per_chunk`（任务内每个分块处理的行数，默认 `2000`）、`--progress_every_chunks`（每合并 N 个分块打印一次摘要，默认 `10`，`<=0` 关闭）、`--shard_size_mb`（分块大小，`>0` 时达到阈值就立即 flush 一个 `_part_xxxx.pt`，避免把全部结果长期堆在内存里）、`--norm_max`（归一化后的最大绝对坐标值，默认 `1.0`，即区间 `[-1,1]`；若设为 `0.8`，则区间为 `[-0.8,0.8]`）、`--min_triangle_area`（最小三角面积阈值，默认 `1e-8`）、`--min_triangle_height`（最小高阈值，默认 `1e-5`）、`--safe_mode`（`all | risky | off`，控制是否将整行处理链放入隔离子进程）、`--part_safe / --node_safe / --hole_safe / --edge_safe / --timeout_safe`（`safe_mode=risky` 时触发隔离的阈值与超时控制）、`--log`（输出 row 级三角剖分日志）。  
  当前 row 级严格过滤标准为：某个 part 只要 `part.is_valid == False` 或存在 `shell-hole-touching`，就会被直接过滤；不会再尝试 `buffer(0)`、`make_valid`、`split/shrink touching holes` 等修补。若某一行过滤后没有任何可用 part，或某个保留 part 三角剖分失败/超时/退化过滤后为空，则整行记为 `dropped`。  
  当 `file_type=shp` 时，分块命名为 `<shpfilename>_tri_part_<xxxx>.pt`（例如 `hangzhou_tri_part_0001.pt`）。当 `--shard_size_mb > 0` 时，会在同目录额外写出 `<shpfilename>_tri.manifest.json`，供训练和评估脚本优先按声明顺序读取 shard。
  当使用 `--log` 时，日志默认保存到同一目录，命名为 `<shpfilename>_tri.triangulation_log.json`。若存在 row 级异常或 chunk 级失败，还会额外写出 `<shpfilename>_tri.row_failures.json`，用于定位失败行。
  运行摘要中的 `triangulated / dropped / degenerated` 现在都按 `row` 统计：`triangulated + dropped = 总行数`，而 `degenerated` 是 `triangulated` 的子集，表示该行在处理过程中发生过 part 过滤或三角形过滤，但最终仍成功输出。  
  同时，终端摘要还会额外打印两组 row 画像统计：  
  `MultiPolygon rows`：`total / triangulated / dropped`；  
  `Hole rows`：`total / triangulated / dropped`。  
  对应的 JSON 日志里也会增加这些字段：`multipolygon_rows`、`triangulated_multipolygon_rows`、`dropped_multipolygon_rows`、`hole_rows`、`triangulated_hole_rows`、`dropped_hole_rows`。  
  此外，日志 JSON 里还会新增两组明细记录：  
  `multipolygon_row_records`：所有 `MultiPolygon` row 的记录；  
  `hole_row_records`：所有带孔 row 的记录。  
  每条记录会包含 `row_idx`、`geom_type`、`is_multipolygon`、`raw_part_count`、`has_holes`、`parts_with_holes`、`total_hole_count`、`max_part_hole_count`、`safe_mode`、`isolated`、`status`、`drop_reason`、`degenerated`、`filtered_part_count`、`filtered_triangle_count`、`kept_triangle_count`，便于后续针对多部件图斑和带孔图斑做定向排查。
  
- 调取示例：
  

```bash
python scripts/run_build_dataset.py \

  --input_dirs /data/raw/vector_a /data/raw/vector_b \

  --output_dir ./data/processed \

  --num_workers 16 \

  --rows_per_chunk 2000 \

  --progress_every_chunks 10 \

  --shard_size_mb 500 \

  --safe_mode risky \

  --part_safe 1 \

  --node_safe 2048 \

  --hole_safe 1 \

  --edge_safe 1e-5 \

  --timeout_safe 20 \

  --norm_max 1.0 \

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
  
- 具体功能：对一个或多个 shapefile 目录执行 row 级静态诊断，完全对齐当前 `run_build_dataset.py` 的处理语义，但不直接执行批量构建写盘。脚本会按 `shp-row` 进行整体去中心化+归一化、part 严格过滤，并依据 `safe_mode + *_safe` 阈值判断这一整行是否会进入隔离子进程，再在同一套 row 级规则下尝试判定该行最终会成功输出、发生退化过滤，还是会被 dropped。该脚本的目标不是给出抽象风险等级，而是帮助定位“哪些 row 会在当前构建规则下被丢弃，以及原因是什么”。
  
- 配置说明：不依赖 YAML；必须传 `--input_dirs`。这里可以传入 1 个或多个目录，每个目录都应包含同名 `.shp/.shx/.dbf` 等 shapefile 组件，并且目录下必须只有一个 `.shp` 文件。常用可选参数有 `--output_dir`（输出目录，默认 `./outputs/polygon_diagnosis`）、`--num_workers`（诊断并行 worker 数，`<=0` 自动）、`--rows_per_chunk`（诊断分块行数，默认 `2000`）、`--row_start` / `--row_end`（只诊断部分行，`row_end` 为开区间）、`--safe_mode`、`--part_safe`、`--node_safe`、`--hole_safe`、`--edge_safe`、`--timeout_safe`、`--norm_max`、`--min_triangle_area`、`--min_triangle_height`。脚本会为每个输入目录各自输出到 `<output_dir>/<任务名>/`，主要包含 `summary.json`、`risk_samples.jsonl` 以及 1 张三联饼图总览。  
  索引语义：  
  `row_idx`：原始 `.shp` 的行号，也就是 `GeoDataFrame` 中的 geometry 序号；当前诊断和构建都严格按 row 统计，不再有训练样本级的 `sample_index` 语义。
  
  `summary.json`：汇总扫描范围、状态计数、图斑类型计数、风险标签计数，以及饼图涉及的数值统计。其中：
  `node_count_bucket_counts` 按 row 统计总节点数的 5 档分布；
  `min_edge_bucket_counts` 按 row 统计归一化后最小边长的 5 档分布；
  `connectivity_bucket_counts` 按原始 polygon 行统计 5 类连通性分布；
  `risk_row_polygon_type_counts` 按“最终被 dropped 的原始行”统计。  
  同时会记录 row 级状态计数，例如 `triangulated_rows`、`dropped_rows`、`degenerated_rows`、`isolated_rows`。
  
  `risk_samples.jsonl`：只记录最终被 `dropped` 的 row；每条记录都会附带 `polygon_type`、过滤统计、是否会进入隔离子进程、drop 原因等字段。`polygon_type` 定义为：
  
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
  --safe_mode risky \
  --part_safe 1 \
  --node_safe 2048 \
  --hole_safe 1 \
  --edge_safe 1e-5 \
  --timeout_safe 20 \
  --norm_max 1.0 \
  --min_triangle_area 1e-8 \
  --min_triangle_height 1e-5 \
  --num_workers 8 \
  --rows_per_chunk 1000
```

```bash
python scripts/run_polygon_diagnosis.py \
  --input_dirs /data/raw/testlook /data/raw/hangzhou \
  --output_dir ./outputs/polygon_diagnosis \
  --safe_mode all \
  --num_workers 16 \
  --rows_per_chunk 2000
```

```bash
python scripts/run_polygon_diagnosis.py \
  --input_dirs /data/raw/testlook \
  --output_dir ./outputs/polygon_diagnosis \
  --row_start 1000 \
  --row_end 1500 \
  --safe_mode off \
  --norm_max 0.8
```

### 6 单图斑与三角剖分可视化诊断

- 入口脚本：`scripts/run_viz_polygon_triangulation.py`
  
- 具体功能：从单个 `.shp` 中提取指定原始行，并在新的 row 级处理语义下输出可视化 PNG 与 JSON 元数据文件。脚本主视角是“整行如何被处理”，但仍保留 `--part_index`，用于高亮这一行中的某个原始 part。PNG 会同时展示：原始整行 geometry、选中的 raw part、整行归一化后的 parts、过滤后的 parts，以及“原始三角剖分结果 / 退化过滤后的三角剖分结果”两张图；失败时也仍会输出 PNG 和 JSON，便于直观看到 part 过滤、整行隔离判定、逐 part 剖分、超时与退化过滤的结果。
  
- 配置说明：必须传 `--input_dir` 和 `--row_index`。  
  `--part_index` 表示该行中的第几个 part，按 1 开始计数，默认是 `1`；如果超过 part 总数，脚本会自动取最后一个 part。  
  `--timeout` 是 `--timeout_safe` 的别名，表示可视化阶段对单个保留 part 的受控三角剖分最长等待时间（秒）；超过该时间会被判定为失败并写入 PNG/JSON。  
  其他常用参数与构建脚本保持一致：`--safe_mode`、`--part_safe`、`--node_safe`、`--hole_safe`、`--edge_safe`、`--norm_max`、`--min_triangle_area`、`--min_triangle_height`。  
  当前 PNG 默认包含 8 个区域：`Raw Row Geometry`、`Selected Raw Part`、`Row-Normalized Parts`、`Filtered Parts`、`Raw Triangulation`、`Filtered Triangulation`、`Part Summary`、`Metadata`。被过滤的 part 会在过滤面板中以灰/红样式显示；最终剖分面板只展示保留下来的 part。
  
- 调取示例：

```bash
python scripts/run_viz_polygon_triangulation.py \
  --input_dir /data/raw/testlook \
  --row_index 6358 \
  --part_index 2 \
  --safe_mode risky \
  --part_safe 1 \
  --node_safe 2048 \
  --hole_safe 1 \
  --edge_safe 1e-5 \
  --timeout 20 \
  --norm_max 1.0 \
  --min_triangle_area 1e-8 \
  --min_triangle_height 1e-5 \
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

  --model_dir ./outputs/exports/ae_20260407_1200 \

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
