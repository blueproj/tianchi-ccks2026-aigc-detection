"""
Qwen3.6-27B 分阶段推理脚本
=========================
分两次独立运行, 彻底避免显存冲突:

Stage 1 (detect): 加载 base + detect LoRA → 推理全部样本 → 输出 detect_results.jsonl
Stage 2 (source): 加载 base + source LoRA → 读取 detect 结果, 对 label=1 推理 → 输出 submit.jsonl

用法:
  # Stage 1: detect 推理
  python infer_two_stage.py \
    --stage detect \
    --model_path /path/to/Qwen3.6-27B \
    --lora_path /path/to/detect_lora \
    --test_path /path/to/test_b.jsonl \
    --output /path/to/detect_results.jsonl \
    --batch_size 4

  # Stage 2: source 推理 (需等 stage 1 完成后单独运行)
  python infer_two_stage.py \
    --stage source \
    --model_path /path/to/Qwen3.6-27B \
    --lora_path /path/to/source_lora \
    --test_path /path/to/test_b.jsonl \
    --detect_results /path/to/detect_results.jsonl \
    --output /path/to/submit.jsonl \
    --batch_size 4
"""

import json
import os
import re
import gc
import argparse
import logging
import torch
from typing import List
from collections import Counter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ============================================================
# 常量
# ============================================================

DETECT_SYSTEM = (
    "You are a text classifier. Classify the text into one of three categories:\n"
    "0 - Human-written text\n"
    "1 - Machine-generated text\n"
    "2 - Human-machine collaboration\n"
    "Output only the number."
)

SOURCE_SYSTEM = (
    "You are a model attribution expert. The text is machine-generated.\n"
    "Identify the model family:\n"
    "0 - OpenAI (GPT series)\n"
    "1 - Alibaba (Qwen series)\n"
    "2 - DeepSeek\n"
    "3 - ByteDance (Doubao)\n"
    "4 - Moonshot (Kimi)\n"
    "5 - Google (Gemini)\n"
    "6 - Anthropic (Claude)\n"
    "7 - xAI (Grok)\n"
    "Output only the number."
)


# ============================================================
# 工具函数
# ============================================================

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_prompt(text, task, tokenizer):
    system = DETECT_SYSTEM if task == "detect" else SOURCE_SYSTEM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return prompt


def parse_label(text, num_classes):
    match = re.search(r"\d+", text.strip())
    if match:
        label = int(match.group())
        if 0 <= label < num_classes:
            return label
    return 0


# ============================================================
# 模型加载与释放
# ============================================================

def load_model(model_path, lora_path, use_4bit=True, logger=None, gpu_id=None):
    """加载模型。gpu_id 不为 None 时用单卡模式 (4卡并行)。"""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if logger:
        logger.info(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_4bit:
        if logger:
            logger.info(f"QLoRA 4-bit 模式加载 (gpu_id={gpu_id})")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        device_map = {"": 0} if gpu_id is not None else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        if logger:
            logger.info(f"BF16 模式加载 (gpu_id={gpu_id})")
        device_map = {"": 0} if gpu_id is not None else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

    if logger:
        logger.info(f"加载 LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    model.config.use_cache = True

    return model, tokenizer


def free_model(model):
    del model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 批量推理
# ============================================================

def batch_generate(model, tokenizer, prompts, batch_size=4, max_new_tokens=5, logger=None):
    results = []
    total = len(prompts)

    for i in range(0, total, batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        results.extend(decoded)

        if logger and (i // batch_size + 1) % 10 == 0:
            logger.info(f"  推理进度: {i + len(batch_prompts)}/{total}")

    return results


# ============================================================
# Stage 1: Detect 推理
# ============================================================

def run_detect(args, logger):
    logger.info("=" * 60)
    logger.info("Stage 1: 文本检测 (三分类)")
    logger.info("=" * 60)

    # 加载测试数据
    test_data = load_jsonl(args.test_path)
    logger.info(f"测试数据: {len(test_data)} 条")

    # 数据分片
    gpu_id = args.shard_id if args.num_shards > 1 else None
    if args.num_shards > 1:
        shard_size = len(test_data) // args.num_shards
        start = args.shard_id * shard_size
        end = (args.shard_id + 1) * shard_size if args.shard_id < args.num_shards - 1 else len(test_data)
        test_data = test_data[start:end]
        logger.info(f"分片 {args.shard_id}/{args.num_shards}: {len(test_data)} 条 [{start}:{end}]")

    # 加载模型
    use_4bit = not args.no_4bit
    model, tokenizer = load_model(args.model_path, args.lora_path, use_4bit, logger, gpu_id=gpu_id)

    # 构建 prompt
    prompts = [build_prompt(d["text"], "detect", tokenizer) for d in test_data]

    # 推理
    logger.info(f"开始 detect 推理 ({len(prompts)} 条)...")
    results = batch_generate(model, tokenizer, prompts,
                             batch_size=args.batch_size,
                             max_new_tokens=args.max_new_tokens,
                             logger=logger)

    # 解析标签
    labels = [parse_label(r, 3) for r in results]
    label_dist = Counter(labels)
    logger.info(f"检测标签分布: {dict(sorted(label_dist.items()))}")

    # 释放模型
    free_model(model)

    # 输出
    output_data = []
    for i, d in enumerate(test_data):
        output_data.append({"id": d["id"], "label": labels[i], "family": -1})

    # 分片模式: 输出到 shard 文件
    if args.num_shards > 1:
        out_path = args.output.replace(".jsonl", f"_shard{args.shard_id}.jsonl")
    else:
        out_path = args.output
    save_jsonl(output_data, out_path)
    logger.info(f"Detect 结果保存: {out_path} ({len(output_data)} 条)")
    logger.info("Stage 1 完成!")


# ============================================================
# Stage 2: Source 推理
# ============================================================

def run_source(args, logger):
    logger.info("=" * 60)
    logger.info("Stage 2: 模型溯源 (八分类, 仅 label=1)")
    logger.info("=" * 60)

    # 加载 detect 结果
    detect_data = load_jsonl(args.detect_results)
    logger.info(f"Detect 结果: {len(detect_data)} 条")

    # 加载原始测试集
    test_data = load_jsonl(args.test_path)
    test_text_map = {d["id"]: d["text"] for d in test_data}

    # 找出 label=1 的样本
    label1_items = [d for d in detect_data if d.get("label") == 1]
    logger.info(f"label=1 样本数: {len(label1_items)}")

    if len(label1_items) == 0:
        logger.warning("无 label=1 样本, 直接输出 detect 结果")
        save_jsonl(detect_data, args.output)
        return

    # 分片
    gpu_id = args.shard_id if args.num_shards > 1 else None
    if args.num_shards > 1:
        shard_size = len(label1_items) // args.num_shards
        start = args.shard_id * shard_size
        end = (args.shard_id + 1) * shard_size if args.shard_id < args.num_shards - 1 else len(label1_items)
        label1_items = label1_items[start:end]
        logger.info(f"分片 {args.shard_id}/{args.num_shards}: {len(label1_items)} 条 [{start}:{end}]")

    # 加载模型
    use_4bit = not args.no_4bit
    model, tokenizer = load_model(args.model_path, args.lora_path, use_4bit, logger, gpu_id=gpu_id)

    # 构建 prompt
    prompts = []
    for item in label1_items:
        text = test_text_map.get(item["id"], "")
        prompts.append(build_prompt(text, "source", tokenizer))

    # 推理
    logger.info(f"开始 source 推理 ({len(prompts)} 条)...")
    results = batch_generate(model, tokenizer, prompts,
                             batch_size=args.batch_size,
                             max_new_tokens=args.max_new_tokens,
                             logger=logger)

    # 解析 family
    families = [parse_label(r, 8) for r in results]
    family_dist = Counter(families)
    logger.info(f"溯源 family 分布: {dict(sorted(family_dist.items()))}")

    # 释放模型
    free_model(model)

    # 分片模式: 输出 shard 结果 (id -> family)
    if args.num_shards > 1:
        out_path = args.output.replace(".jsonl", f"_shard{args.shard_id}.jsonl")
        shard_data = [{"id": item["id"], "family": families[i]} for i, item in enumerate(label1_items)]
        save_jsonl(shard_data, out_path)
        logger.info(f"Source shard 结果保存: {out_path} ({len(shard_data)} 条)")
    else:
        # 单进程: 合并输出最终 submit
        family_map = {item["id"]: families[i] for i, item in enumerate(label1_items)}
        output_data = []
        for item in detect_data:
            new_item = dict(item)
            if item.get("label") == 1:
                new_item["family"] = family_map.get(item["id"], 0)
            else:
                new_item["family"] = -1
            output_data.append(new_item)
        save_jsonl(output_data, args.output)
        logger.info(f"最终结果保存: {args.output} ({len(output_data)} 条)")
    logger.info("Stage 2 完成!")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-27B 分阶段推理")
    parser.add_argument("--stage", type=str, required=True, choices=["detect", "source"],
                        help="detect: 三分类检测, source: 八分类溯源")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True,
                        help="detect 阶段: detect LoRA 路径; source 阶段: source LoRA 路径")
    parser.add_argument("--test_path", type=str, required=True, help="原始测试集")
    parser.add_argument("--output", type=str, required=True, help="输出路径")
    parser.add_argument("--detect_results", type=str, default=None,
                        help="source 阶段需要: detect 结果文件路径")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--no_4bit", action="store_true", default=False)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    logger = setup_logger()

    if args.stage == "detect":
        run_detect(args, logger)
    elif args.stage == "source":
        if not args.detect_results:
            logger.error("source 阶段需要 --detect_results 参数")
            return
        run_source(args, logger)


if __name__ == "__main__":
    main()
