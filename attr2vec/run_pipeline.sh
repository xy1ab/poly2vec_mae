#!/bin/bash
# 注意：移除了 set -e，确保流程不会因单个脚本崩溃而中断

export OMP_NUM_THREADS=8
# 🌟 核心屏蔽：剔除被占用的 GPU 6，只让程序看到剩下的 7 张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7

LOG_FILE="NRE_Production_Final.log"

echo "======================================================" > "$LOG_FILE"
echo "🚀 [大一统底座] 0.1B 生产训练与全方位评估流水线启动 (7卡模式)" >> "$LOG_FILE"
echo "🕒 启动时间: $(date)" >> "$LOG_FILE"
echo "======================================================" >> "$LOG_FILE"

# --- 阶段一：模型训练 ---
echo -e "\n\n>>> 🟢 [阶段一] 开始 7 卡 DDP 分布式训练 (1000轮)..." >> "$LOG_FILE"
# 🌟 进程数对应修改为 7
torchrun --nproc_per_node=7 trainer.py >> "$LOG_FILE" 2>&1 || echo "⚠️ 训练过程遇到异常，但流水线继续..." >> "$LOG_FILE"

# --- 阶段二：评估流水线 ---
echo -e "\n\n======================================================" >> "$LOG_FILE"
echo "🏆 训练环节结束，正式启动全方位大测试集效果验证" >> "$LOG_FILE"
echo "======================================================" >> "$LOG_FILE"

scripts=(
    "verify_lossless.py"       # 1. 物理保真红线验证
    "semantic_probe.py"        # 2. 语义流形演算
    "semantic_probe_3lei.py"   # 3. 跨源语义对齐验证
    "text_hunter.py"           # 4. 极限长文本解码验证
    "extreme_hunter.py"        # 5. 极值张量追踪
    "biz_value_eval.py"        # 6. 业务价值消融实验
    "evaluator.py"             # 7. 全局图表生成
)

for script in "${scripts[@]}"; do
    echo -e "\n>>> 🔬 正在执行验证模块: $script ..." | tee -a "$LOG_FILE"
    python "$script" >> "$LOG_FILE" 2>&1
    
    # 检查上一条指令的退出状态码
    if [ $? -eq 0 ]; then
        echo "    ✅ $script 执行成功！" >> "$LOG_FILE"
    else
        echo "    ❌ $script 执行崩溃，已跳过并继续执行下一个！" >> "$LOG_FILE"
    fi
done

echo -e "\n\n======================================================" >> "$LOG_FILE"
echo "🕒 全局流水线结束时间: $(date)" >> "$LOG_FILE"
echo "======================================================" >> "$LOG_FILE"