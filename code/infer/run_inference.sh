#!/bin/bash
# ============================================================
# 两阶段 4 卡并行推理
# ============================================================
# Stage 1: detect 4卡并行 → detect_results.jsonl
# Stage 2: source 4卡并行 → submit.jsonl
#
# 用法:
#   bash run_inference.sh \
#     --detect_lora /path/to/detect_lora \
#     --source_lora /path/to/source_lora \
#     --test_path /path/to/test_b.jsonl \
#     --detect_output /path/to/detect_results.jsonl \
#     --output /path/to/submit.jsonl
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-./models/Qwen3.6-27B}"
NUM_GPUS=4
BATCH_SIZE=4

# 默认参数
DETECT_LORA=""
SOURCE_LORA=""
TEST_PATH="./data/test_b.jsonl"
DETECT_OUTPUT="./output/detect_results.jsonl"
OUTPUT="./output/submit.jsonl"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --detect_lora)   DETECT_LORA=$2; shift 2;;
        --source_lora)   SOURCE_LORA=$2; shift 2;;
        --test_path)     TEST_PATH=$2; shift 2;;
        --detect_output) DETECT_OUTPUT=$2; shift 2;;
        --output)        OUTPUT=$2; shift 2;;
        --batch_size)    BATCH_SIZE=$2; shift 2;;
        *) shift;;
    esac
done

if [ -z "$DETECT_LORA" ] || [ -z "$SOURCE_LORA" ]; then
    echo "用法: bash run_inference.sh --detect_lora <path> --source_lora <path> [--test_path <path>] [--output <path>]"
    exit 1
fi

echo "============================================================"
echo "两阶段 ${NUM_GPUS} 卡并行推理"
echo "============================================================"
echo "模型: ${MODEL_PATH}"
echo "Detect LoRA: ${DETECT_LORA}"
echo "Source LoRA: ${SOURCE_LORA}"
echo "测试集: ${TEST_PATH}"
echo "Detect 输出: ${DETECT_OUTPUT}"
echo "最终输出: ${OUTPUT}"
echo "GPU数: ${NUM_GPUS}"
echo "Batch: ${BATCH_SIZE}"
echo "============================================================"

# ============================================================
# Stage 1: Detect 多卡并行
# ============================================================
echo ""
echo ">>> Stage 1: Detect ${NUM_GPUS}卡并行"
echo ""

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "启动 Detect Shard ${i}..."
    python "${SCRIPT_DIR}/infer_two_stage.py" \
        --stage detect \
        --model_path "${MODEL_PATH}" \
        --lora_path "${DETECT_LORA}" \
        --test_path "${TEST_PATH}" \
        --output "${DETECT_OUTPUT}" \
        --batch_size ${BATCH_SIZE} \
        --shard_id ${i} \
        --num_shards ${NUM_GPUS} \
        > "${SCRIPT_DIR}/detect_shard${i}.log" 2>&1 &
    PIDS+=($!)
done

echo "等待 Detect ${NUM_GPUS} 个分片完成..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait ${PIDS[$i]}; then
        echo "✓ Detect Shard ${i} 完成"
    else
        echo "✗ Detect Shard ${i} 失败"
        FAILED=1
    fi
done

if [ ${FAILED} -eq 1 ]; then
    echo "Detect 有分片失败, 检查日志:"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        echo "  cat ${SCRIPT_DIR}/detect_shard${i}.log"
    done
    exit 1
fi

# 合并 detect 分片
echo ""
echo "合并 Detect 分片..."
python3 -c "
import json
all_data = []
for i in range(${NUM_GPUS}):
    shard_file = '${DETECT_OUTPUT}'.replace('.jsonl', f'_shard{i}.jsonl')
    with open(shard_file) as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    print(f'  shard{i}: {len(all_data)} 条 (累计)')
with open('${DETECT_OUTPUT}', 'w') as f:
    for d in all_data:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
from collections import Counter
labels = Counter(d['label'] for d in all_data)
print(f'  合计: {len(all_data)} 条')
print(f'  Label: {dict(sorted(labels.items()))}')
print(f'  输出: ${DETECT_OUTPUT}')
"

# ============================================================
# Stage 2: Source 多卡并行
# ============================================================
echo ""
echo ">>> Stage 2: Source ${NUM_GPUS}卡并行"
echo ""

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "启动 Source Shard ${i}..."
    python "${SCRIPT_DIR}/infer_two_stage.py" \
        --stage source \
        --model_path "${MODEL_PATH}" \
        --lora_path "${SOURCE_LORA}" \
        --test_path "${TEST_PATH}" \
        --detect_results "${DETECT_OUTPUT}" \
        --output "${OUTPUT}" \
        --batch_size ${BATCH_SIZE} \
        --shard_id ${i} \
        --num_shards ${NUM_GPUS} \
        > "${SCRIPT_DIR}/source_shard${i}.log" 2>&1 &
    PIDS+=($!)
done

echo "等待 Source ${NUM_GPUS} 个分片完成..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait ${PIDS[$i]}; then
        echo "✓ Source Shard ${i} 完成"
    else
        echo "✗ Source Shard ${i} 失败"
        FAILED=1
    fi
done

if [ ${FAILED} -eq 1 ]; then
    echo "Source 有分片失败, 检查日志:"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        echo "  cat ${SCRIPT_DIR}/source_shard${i}.log"
    done
    exit 1
fi

# 合并 source 分片 → 最终 submit
echo ""
echo "合并 Source 分片..."
python3 -c "
import json
from collections import Counter

# 加载 detect 结果
with open('${DETECT_OUTPUT}') as f:
    detect = [json.loads(l) for l in f if l.strip()]

# 加载 source shard 结果
new_families = {}
for i in range(${NUM_GPUS}):
    shard_file = '${OUTPUT}'.replace('.jsonl', f'_shard{i}.jsonl')
    try:
        with open(shard_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    new_families[d['id']] = d['family']
        print(f'  source shard{i}: {len(new_families)} 条 (累计)')
    except FileNotFoundError:
        print(f'  source shard{i}: 不存在, 跳过')

# 合并
for item in detect:
    if item.get('label') == 1 and item['id'] in new_families:
        item['family'] = new_families[item['id']]

with open('${OUTPUT}', 'w') as f:
    for d in detect:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')

labels = Counter(d['label'] for d in detect)
families = Counter(d['family'] for d in detect if d.get('family', -1) != -1)
print(f'  合计: {len(detect)} 条')
print(f'  Label: {dict(sorted(labels.items()))}')
print(f'  Family: {dict(sorted(families.items()))}')
print(f'  输出: ${OUTPUT}')
"

echo ""
echo "============================================================"
echo "全部完成!"
echo "  Detect 结果: ${DETECT_OUTPUT}"
echo "  最终提交: ${OUTPUT}"
echo "============================================================"
