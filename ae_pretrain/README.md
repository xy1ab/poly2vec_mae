# ae_pretrain

经典深度学习工程结构：

- `configs/`: 训练、评估、导出配置（平铺，不再细分子目录）
- `scripts/`: 启动脚本
- `data/`: 原始/处理后/嵌入数据
- `outputs/`: 训练产物（`ckpt/`）和导出包（`exports/`）
- `src/models`: 模型结构
- `src/datasets`: 数据集、增强、几何编解码
- `src/losses`: 损失函数
- `src/engine`: `trainer.py` / `evaluator.py` / `pipeline.py`
- `src/utils`: 基础工具

## 快速开始

```bash
pip install -r requirements.txt
python scripts/run_pretrain.py --config configs/pretrain_base.yaml
```

评估：

```bash
python scripts/run_eval.py --config configs/eval_default.yaml
```

导出 encoder：

```bash
python scripts/run_export.py --config configs/export_default.yaml
```

DDP（单机多卡）示例：

```bash
# 按需覆写 GPU 列表，脚本会自动按卡数拉起 DDP
python scripts/run_pretrain.py --config configs/pretrain_base.yaml --gpu 0,1

# 如果端口冲突，可指定 master_port
python scripts/run_pretrain.py --config configs/pretrain_base.yaml --gpu 0,1 --master_port 29600
```
