#!/bin/bash
# ==============================================================
# 自然资源大一统底座 (NRE-0.1B) 端到端全自动流水线
# 包含：字典构建 -> 数据缓存化 -> 分布式训练 -> 多维评估审计
# ==============================================================

export OMP_NUM_THREADS=8
# 如果在南湖平台运行，需调整为南湖的 MUSA 可见卡环境变量配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

LOG_FILE="NRE_Global_Pipeline.log"
echo "🚀 [自然资源大一统底座] 端到端全流程启动 | 时间: $(date)" > "$LOG_FILE"

# ==========================================
# 阶段 1：全局词汇表构建 (Vocab Building)
# ==========================================
echo -e "\n>>> 📚 [阶段 1/4] 正在执行全域数据扫描，构建全局字典..." | tee -a "$LOG_FILE"
python vocab_builder.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ 字典构建失败！流水线强制终止。" | tee -a "$LOG_FILE"; exit 1
fi
echo "✅ 全局字典 (global_vocab_auto.json) 构建完毕！" | tee -a "$LOG_FILE"

# ==========================================
# 阶段 2：全量数据张量化与高速缓存 (Data Caching)
# ==========================================
echo -e "\n>>> 🗜️ [阶段 2/4] 正在抽取 GDB/CSV 属性，生成高速张量缓存 (.pt)..." | tee -a "$LOG_FILE"
python data_loader.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ 数据张量化失败！流水线强制终止。" | tee -a "$LOG_FILE"; exit 1
fi
echo "✅ 高速缓存池构建完毕！" | tee -a "$LOG_FILE"

# ==========================================
# 阶段 3：分布式底座训练 (DDP Training)
# ==========================================
echo -e "\n>>> 🧠 [阶段 3/4] 启动多卡 DDP 分布式训练..." | tee -a "$LOG_FILE"
# 注意：在南湖平台可能需要将 torchrun 替换为对应的 MPI 启动命令
torchrun --nproc_per_node=8 trainer.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️ 训练过程遇到警告或异常中断，但将尝试继续评估现有权重..." | tee -a "$LOG_FILE"
else
    echo "✅ 模型训练圆满收官！权重已保存至 best_model.pth" | tee -a "$LOG_FILE"
fi

# ==========================================
# 阶段 4：自动化评估审计流水线 (Evaluation)
# ==========================================
echo -e "\n>>> 🔬 [阶段 4/4] 启动全方位模型体检与图表生成..." | tee -a "$LOG_FILE"

# 评估脚本清单
scripts=(
    "verify_lossless.py"       # 物理无损审计
    "semantic_probe.py"        # 语义对齐探针
    "semantic_probe_3lei.py"   # 跨源流形验证
    "text_hunter.py"           # 文本极限猎手
    "extreme_hunter.py"        # 全局极值对账
    "biz_value_eval.py"        # 业务价值消融
    "evaluator.py"             # 综合可视化生成
)

for script in "${scripts[@]}"; do
    echo "    -> 正在执行: $script" | tee -a "$LOG_FILE"
    python "$script" >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        echo "       [✅ 通过] $script" >> "$LOG_FILE"
    else
        echo "       [❌ 失败] $script (已跳过，继续下一项)" >> "$LOG_FILE"
    fi
done

echo -e "\n🎉 流水线全部执行完毕！所有图表和日志已就绪。 | 结束时间: $(date)" | tee -a "$LOG_FILE"