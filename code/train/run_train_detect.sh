#!/bin/bash
# ============================================================
# Qwen3.6-27B LoRA 训练 — Detect 任务 (检测增强数据)
# ============================================================
# 训练数据: 检测增强数据 (~222K, 句子裁剪60%/80%)
# 参数: r=32, alpha=32, 1 epoch, QLoRA 4-bit
#
# 用法:
#   bash run_train_detect.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 路径 (请根据实际环境修改) ----
MODEL_PATH="${MODEL_PATH:-./models/Qwen3.6-27B}"
TRAIN_DATA="${TRAIN_DATA:-./data/augmented_detect/train_augmented.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-./output}"

# ---- 自动检测 GPU ----
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${NUM_GPUS}" -le 0 ] && NUM_GPUS=1

# ---- 训练参数 ----
MAX_LENGTH=2048
LORA_R=32
LORA_ALPHA=32
LORA_DROPOUT=0.05
LEARNING_RATE=2e-4
MIN_LR=2e-5
WARMUP=0.05
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
SEED=42
EPOCHS=1

if [ "${NUM_GPUS}" -ge 2 ]; then
    BATCH_SIZE=2
    GRAD_ACCUM=4
    LAUNCHER="torchrun --nproc_per_node ${NUM_GPUS} --master_port 29501"
else
    BATCH_SIZE=4
    GRAD_ACCUM=8
    MAX_LENGTH=1024
    LAUNCHER="python"
fi

OUTPUT_DIR="${OUTPUT_BASE}/qwen_detect_lora_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo ">>> Qwen3.6-27B LoRA | Task: detect"
echo "============================================================"
echo "模型: ${MODEL_PATH}"
echo "训练数据: ${TRAIN_DATA}"
echo "输出: ${OUTPUT_DIR}"
echo "GPU数: ${NUM_GPUS}"
echo "量化: 4-bit NF4 + 双重量化"
echo "序列长度: ${MAX_LENGTH}"
echo "LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "Epochs: ${EPOCHS}"
echo "LR: ${LEARNING_RATE}, Min LR: ${MIN_LR}"
echo "============================================================"

${LAUNCHER} "${SCRIPT_DIR}/train_qwen.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --train_path "${TRAIN_DATA}" \
    --task detect \
    --max_length ${MAX_LENGTH} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --use_4bit \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --use_double_quant \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine_with_min_lr \
    --min_lr ${MIN_LR} \
    --warmup_ratio ${WARMUP} \
    --weight_decay ${WEIGHT_DECAY} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 10 \
    --gradient_checkpointing \
    --log_dir "${OUTPUT_DIR}/logs" \
    --num_proc 8 \
    --seed ${SEED}

echo "============================================================"
echo "训练完成! LoRA adapter: ${OUTPUT_DIR}"
echo "============================================================"
