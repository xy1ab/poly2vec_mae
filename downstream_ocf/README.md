# Poly2Vec OCF (Occupancy Field) 隐式几何重建模块

本模块负责将 384 维的多边形几何嵌入（Embedding）通过 **SIREN (正弦表示网络)** 还原为高精度的连续空间占用场。

## 📂 项目结构
```text
downstream_ocf/
├── configs/
│   └── recons.yaml          # 训练超参数与路径配置
├── src/
│   ├── model_siren.py       # FiLM-SIREN 模型架构
│   └── loader_ocf.py        # 数据智能采样加载器
├── scripts/
│   ├── run_siren_train.py   # 训练主入口 (支持单卡/多卡)
│   └── run_siren_eval.py    # 性能评估脚本
├── data/                    # 存放 encoded_samples_xxx.pt 数据
└── outputs/                 # 存放模型权重 (.pth) 与可视化结果
```

## 🛠️ 环境准备
在项目根目录下运行：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 🚀 运行训练 (Training)

所有的训练脚本均支持从命令行直接覆盖 `configs/recons.yaml` 中的默认参数。

### 1. 声明环境变量 (每次打开新终端需执行)
```bash
export PYTHONPATH=$PYTHONPATH:.
```

### 2. 单卡训练 (Single-GPU)
```bash
python scripts/run_siren_train.py --batch_size 128 --epochs 100
```

### 3. 多卡并行训练 (Multi-GPU DDP)
利用 `torchrun` 启动。假设你有 4 张显卡：
```bash
torchrun --nproc_per_node=4 scripts/run_siren_train.py --batch_size 128
```

### 4. 常用命令行参数说明
| 参数 | 说明 | 示例 |
| :--- | :--- | :--- |
| `--config` | 指定 YAML 配置文件路径 | `--config configs/recons.yaml` |
| `--epochs` | 训练总轮数 | `--epochs 100` |
| `--batch_size` | 单卡批大小 | `--batch_size 256` |
| `--num_points` | 每次采样的坐标点数 | `--num_points 2048` |
| `--boundary_ratio`| 边界采样点的比例 (0.0~1.0) | `--boundary_ratio 0.8` |
| `--lr` | 学习率 | `--lr 5e-5` |

---

## 📊 模型评估 (Evaluation)

评估脚本会加载训练好的 `siren_ocf_best.pth`，并从封存的测试集中随机抽取样本，生成高分辨率对比图。

### 运行评估：
```bash
# 默认 256 分辨率测试
python scripts/run_siren_eval.py

# 1024x1024 无损超分测试 (查看边缘锐利度)
python scripts/run_siren_eval.py --resolution 1024 --num_samples 5
```

### 评估产出：
- **`outputs/vis_results_siren/`**: 包含模型预测概率场、二值化边界以及误差分布的对比图。
- **`.txt 报告`**: 自动记录每个样本的 IoU、连通分量数 (验证是否闭环) 以及 Hausdorff 距离 (验证是否无噪点)。

---

## 💡 核心技术点
1. **FiLM 调制**: 使用 384 维 Embedding 逐层控制 SIREN 的幅值和偏置，实现形状与位置的解耦。
2. **空间归一化**: 加载器自动根据 Meta 信息将“纳米级多边形”平移并放大至 `[-1, 1]` 空间，确保模型能够学习精细的直角特征。
3. **Smart Sampling**: 采用 70% 边界采样 + 30% 全局采样策略，针对性攻克几何边缘修复难题。
```
