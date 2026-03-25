# poly2vec_mae

本项目采用 `configs + src + scripts` 架构，MAE 预训练服务下游任务。

## 目录结构

```text
poly2vec_mae/
├── configs/
│   ├── mae/
│   └── downstream/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
│   ├── checkpoints/
│   └── exports/
├── src/
│   ├── mae_core/
│   ├── downstream/
│   │   ├── task_recons/
│   │   └── task_classify/
│   └── utils/
│       ├── config/
│       ├── data/
│       ├── fourier/
│       ├── geometry/
│       ├── io/
│       └── viz/
├── scripts/
│   ├── mae/
│   └── downstream/
├── tests/
├── requirements.txt
└── README.md
```

## 环境安装

```bash
pip install -r requirements.txt
```

## MAE 预训练

默认读取 `configs/mae/pretrain.yaml`：

```bash
python scripts/mae/run_pretrain.py
```

指定配置：

```bash
python scripts/mae/run_pretrain.py --config configs/mae/pretrain.yaml
```

默认使用 `bf16` 混合精度训练与 `bf16` 权重导出（可在配置中改为 `fp32/fp16`）。

训练产物默认保存到：

- `outputs/checkpoints/<timestamp>/train_log.txt`
- `outputs/checkpoints/<timestamp>/recon_epoch_*.png`
- `outputs/checkpoints/<timestamp>/mae_ckpt_*.pth`
- `outputs/checkpoints/<timestamp>/poly_encoder_epoch_*.pth`

评估与下游可视化也建议保存到同一实验目录：

- `outputs/checkpoints/<timestamp>/cft_visualize_*.png`
- `outputs/checkpoints/<timestamp>/resnet_integration_proof_*.png`

训练结束后会自动导出交付包到：

- `outputs/exports/mae_<timestamp>/config.yaml`
- `outputs/exports/mae_<timestamp>/encoder_decoder.pth`
- `outputs/exports/mae_<timestamp>/encoder.pth`
- `outputs/exports/mae_<timestamp>/train_log.txt`

## MAE 评估可视化

```bash
python scripts/mae/run_eval.py
```

## 下游重建流水线示例

```bash
python scripts/downstream/run_pipeline_mae.py
```

可指定推理精度：

```bash
python scripts/downstream/run_pipeline_mae.py --precision bf16
```

## 编码器导出接口

- 主实现文件：`src/mae_core/model.py`
- 入口函数：
  - `export_encoder_from_mae_checkpoint(...)`
  - `load_pretrained_encoder(...)`

---

## 📖 Poly2Vec OCF (Occupancy Field) 隐式几何重建模块

利用预训练提取的 **384维Embedding**，通过 **FiLM 调制机制** 控制 **SIREN (正弦表示网络)**。模型不再输出离散像素，而是学习一个连续的空间占用场函数 $f(x, y | v) \to[0, 1]$，从而支持无限分辨率的完美多边形矢量提取与快速的空间布尔运算（图斑求交集）。

---

## 📂 核心项目架构

本模块遵循标准的配置分离架构，核心文件分布如下：

```text
poly2vec_mae/
├── configs/downstream/
│   └── recons.yaml                  # 统一配置文件
├── src/downstream/task_recons/
│   ├── model_siren.py               # SIREN 核心网络架构 
│   └── loader_ocf.py                # 数据加载
└── scripts/downstream/
    └── run_siren_train.py           # 训练主入口脚本
```

---

## 🛠️ 核心文件深度解析

### 1. `configs/downstream/recons.yaml` 
所有的实验参数均在此设置。
*   包含采样策略（如 `num_points: 1024`, `boundary_ratio: 0.7`）。
*   包含网络维度配置（`embed_dim: 384`, `hidden_dim: 256`, `num_layers: 5`）。

### 2. `src/downstream/task_recons/loader_ocf.py` 
负责从 `.pt` 提取数据。
*   **智能采样 (Smart Sampling)**
    *   **70%** 的坐标点生成在原始三角形边界附近。
    *   **30%** 的坐标点在全局 $[-1, 1]$ 平面随机生成。
*   **动态标签判定**：利用 `matplotlib.path` 底层 C 接口极速计算坐标点是否在多边形内（1.0 内部，0.0 外部）。

### 3. `src/downstream/task_recons/model_siren.py` 
将静态的 Embedding 转化为动态的几何函数。
*   **FiLM 调制**：将 384 维 Embedding 映射为缩放系数 ($\gamma$) 和偏移系数 ($\beta$)，逐层改变 SIREN 网络的状态。
*   **SIREN 正弦波层**：采用 $\sin(\omega_0 \cdot x)$ 激活函数，第一层 $\omega_0=30$。

### 4. `scripts/downstream/run_siren_train.py` 
*   **严谨切分**：自动按 **90% 训练 / 5% 验证 / 5% 测试** 划分数据。
*   **防数据泄露**：使用固定随机种子，并将测试集索引封存于 `siren_test_indices.pt`。
*   **实时评估**：每轮结束自动计算验证集 **mIoU (平均交并比)**。
*   **容灾存档**：自动跟踪最高 mIoU 并保存为 `siren_ocf_best.pth`。

---

## 🚀 快速启动

### 启动训练
在项目根目录 (`poly2vec_mae/`) 下执行以下命令：

```bash
# 1. 声明项目搜索路径 (极其重要，否则报 ModuleNotFoundError)
export PYTHONPATH=$PYTHONPATH:.

# 2. 后台挂起全量训练任务
nohup python scripts/downstream/run_siren_train.py > siren_train.log 2>&1 &

# 3. 实时监控训练进度与 Loss 表现
tail -f siren_train.log
```

---

## 📊 训练产出物 (Outputs)

在 `checkpoints_siren/` (或 YAML 指定目录) 获得以下资产：

1.  **`siren_ocf_best.pth`**: 验证集 mIoU 最高的最优模型权重。可直接用于后续的**无限分辨率可视化**与**图斑交集运算**。
2.  **`siren_test_indices.pt`**: 被严格隔离的测试集名单，用于撰写论文或评估报告时，证明模型的泛化能力。
