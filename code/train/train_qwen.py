"""
Qwen3.6-27B LoRA SFT 训练脚本 (双LoRA: 检测 + 溯源)
=====================================================
支持两种任务:
  --task detect: 三分类文本检测 (label 0/1/2), 使用全量训练集
  --task source: 八分类模型溯源 (family 0-7), 仅使用 label=1 数据

架构: Qwen3.6-27B (base, 冻结) + QLoRA 4-bit
框架: transformers + peft + bitsandbytes
"""

import json
import os
import argparse
import logging
import random
import numpy as np
import torch
import transformers
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ============================================================
# 常量
# ============================================================
IGNORE_INDEX = -100

# 检测任务 system prompt
DETECT_SYSTEM = (
    "You are a text classifier. Classify the text into one of three categories:\n"
    "0 - Human-written text\n"
    "1 - Machine-generated text\n"
    "2 - Human-machine collaboration\n"
    "Output only the number."
)

# 溯源任务 system prompt
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

def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    # 多卡时只有 rank 0 打印到控制台, 其他 rank 只写文件
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    os.makedirs(log_dir, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{time_str}_rank{local_rank}.log")

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 只有 rank 0 输出到控制台
    if local_rank == 0:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


# ============================================================
# 数据加载与格式化
# ============================================================

def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def filter_by_task(data: List[dict], task: str) -> List[dict]:
    """按任务过滤数据。"""
    if task == "detect":
        # 检测任务: 使用全部数据
        return data
    elif task == "source":
        # 溯源任务: 仅使用 label=1 且有 family 的数据
        return [d for d in data if d.get("label") == 1 and "family" in d]
    else:
        raise ValueError(f"未知任务: {task}, 支持: detect / source")


def build_messages(text: str, target: str, task: str) -> List[Dict[str, str]]:
    """构造 chat 消息。"""
    system = DETECT_SYSTEM if task == "detect" else SOURCE_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
        {"role": "assistant", "content": target},
    ]


def tokenize_sample(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """将 chat 消息 tokenize, 只对 assistant 部分计算 loss。

    策略:
      1. 完整 tokenize (system + user + assistant)
      2. 单独 tokenize prompt (system + user + generation_prompt)
      3. prompt 部分的 labels 设为 IGNORE_INDEX
      4. 总长度超过 max_length 则截断 (保留 assistant 部分)
    """
    # 完整文本 (含 assistant 回复)
    try:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

    # prompt 文本 (不含 assistant 回复, 但含 generation_prompt)
    prompt_messages = messages[:2]  # system + user
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )

    # tokenize
    full_tokens = tokenizer(
        full_text, max_length=max_length, truncation=True,
        return_tensors="pt", add_special_tokens=False,
    )
    prompt_tokens = tokenizer(
        prompt_text, max_length=max_length, truncation=True,
        return_tensors="pt", add_special_tokens=False,
    )

    input_ids = full_tokens["input_ids"][0]
    labels = input_ids.clone()
    prompt_len = prompt_tokens["input_ids"].shape[1]
    labels[:prompt_len] = IGNORE_INDEX

    # 检查: response 部分是否有有效 token
    response_len = len(input_ids) - prompt_len
    if response_len <= 0:
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": full_tokens["attention_mask"][0],
    }


def prepare_dataset(
    data: List[dict],
    tokenizer,
    task: str,
    max_length: int,
    num_proc: int = 8,
) -> Dataset:
    """将数据列表转换为 tokenized Dataset。"""
    logger = logging.getLogger(__name__)

    # 构建目标 label
    texts = []
    targets = []
    for item in data:
        text = item["text"]
        if task == "detect":
            target = str(item["label"])
        else:
            target = str(item["family"])
        texts.append(text)
        targets.append(target)

    logger.info(f"任务: {task}, 数据量: {len(texts)}")
    target_dist = Counter(targets)
    logger.info(f"目标分布: {dict(sorted(target_dist.items()))}")

    dataset = Dataset.from_dict({
        "text": texts,
        "target": targets,
    })

    def tokenize_fn(examples):
        all_input_ids = []
        all_labels = []
        all_attn = []
        all_valid = []
        all_target = []

        for text, target in zip(examples["text"], examples["target"]):
            messages = build_messages(text, target, task)
            result = tokenize_sample(messages, tokenizer, max_length)

            if result is None:
                all_input_ids.append([])
                all_labels.append([])
                all_attn.append([])
                all_valid.append(False)
            else:
                all_input_ids.append(result["input_ids"].tolist())
                all_labels.append(result["labels"].tolist())
                all_attn.append(result["attention_mask"].tolist())
                all_valid.append(True)
            all_target.append(target)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attn,
            "valid": all_valid,
            "target": all_target,
        }

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=["text"],
        desc=f"Tokenizing ({task})",
    )

    # 过滤无效样本 + 按目标统计过滤数
    before = len(tokenized)
    filter_by_target: Counter = Counter()
    for sample in tokenized:
        if not sample["valid"]:
            filter_by_target[sample["target"]] += 1

    tokenized = tokenized.filter(lambda x: x["valid"], desc="Filtering invalid")
    tokenized = tokenized.remove_columns(["valid"])
    filtered = before - len(tokenized)
    if filtered > 0:
        logger.info(
            f"过滤 {filtered} 条无效样本 "
            f"(总长>{max_length} 或 response 为空)"
        )
        logger.info("各 target 过滤数:")
        for t in sorted(filter_by_target.keys()):
            if filter_by_target[t] > 0:
                logger.info(f"  {t:<10} {filter_by_target[t]:>5}")

    # 按 target 分组 token 长度分布
    target_lengths = {}
    for sample in tokenized:
        t = sample["target"]
        if t not in target_lengths:
            target_lengths[t] = []
        target_lengths[t].append(len(sample["input_ids"]))

    label_name = "label" if task == "detect" else "family"
    logger.info(f"各 {label_name} token 长度分布 (有效样本):")
    logger.info(f"  {label_name:<10} {'n':>6} {'p50':>6} {'p95':>6} {'max':>6}")
    for t in sorted(target_lengths.keys()):
        lens_t = np.array(target_lengths[t])
        p50 = int(np.percentile(lens_t, 50))
        p95 = int(np.percentile(lens_t, 95))
        mx = int(lens_t.max())
        logger.info(f"  {t:<10} {len(lens_t):>6} {p50:>6} {p95:>6} {mx:>6}")

    # 整体统计
    lengths = [len(x["input_ids"]) for x in tokenized]
    lens = np.array(lengths)
    logger.info(
        f"整体 Token 长度: mean={lens.mean():.0f}, median={np.median(lens):.0f}, "
        f"p95={np.percentile(lens, 95):.0f}, max={lens.max()}, "
        f"超过{max_length}: {(lens > max_length).sum()}"
    )

    # 移除 target 列 (训练不需要)
    tokenized = tokenized.remove_columns(["target"])

    logger.info(f"Tokenization 完成, 有效样本数: {len(tokenized)}")
    return tokenized


# ============================================================
# 模型加载
# ============================================================

def load_model_and_tokenizer(
    model_name_or_path: str,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    use_gradient_checkpointing: bool = True,
    use_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16",
    use_double_quant: bool = True,
) -> tuple:
    """加载 Qwen3.6-27B 并应用 QLoRA。"""
    logger = logging.getLogger(__name__)
    logger.info(f"加载 tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    compute_dtype = compute_dtype_map.get(bnb_4bit_compute_dtype, torch.bfloat16)

    if use_4bit:
        logger.info(f"QLoRA 模式: 4-bit {bnb_4bit_quant_type}, compute={bnb_4bit_compute_dtype}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
        # 多卡 DDP 时每张卡加载到对应 GPU, 单卡时 device_map=auto
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        device_map = {"": local_rank} if local_rank >= 0 else "auto"
        logger.info(f"device_map={device_map}, LOCAL_RANK={local_rank}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=use_gradient_checkpointing,
        )
    else:
        logger.info(f"BF16 模式: {model_name_or_path}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        except (ValueError, ImportError) as e:
            logger.warning(f"flash_attention_2 不可用 ({e}), 回退到 eager")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    # Qwen3.5 不支持在 from_pretrained 中传 use_cache, 需在 config 中设置
    model.config.use_cache = False

    # LoRA 配置
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {lora_config.target_modules}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================
# Callback
# ============================================================

class TrainingLogCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logger = logging.getLogger(__name__)
        if "loss" in logs:
            parts = [f"step={state.global_step}"]
            for key in ["loss", "grad_norm", "learning_rate", "epoch"]:
                if key in logs:
                    val = logs[key]
                    parts.append(f"{key}={val:.6f}" if isinstance(val, float) else f"{key}={val}")
            logger.info(" | ".join(parts))


# ============================================================
# 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.6-27B LoRA SFT (检测+溯源)")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True, help="训练 JSONL 路径")
    parser.add_argument("--task", type=str, required=True, choices=["detect", "source"],
                        help="detect: 三分类检测, source: 八分类溯源")

    parser.add_argument("--max_length", type=int, default=2048)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true", default=False)
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16")
    parser.add_argument("--use_double_quant", action="store_true", default=True)
    parser.add_argument("--no_double_quant", action="store_true", default=False)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr")
    parser.add_argument("--min_lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="./logs")

    args = parser.parse_args()
    if args.no_4bit:
        args.use_4bit = False
    if args.no_double_quant:
        args.use_double_quant = False
    return args


# ============================================================
# 主流程
# ============================================================

def main():
    args = parse_args()
    setup_seed(args.seed)
    logger = setup_logger(args.log_dir)

    logger.info("=" * 60)
    logger.info(f"Qwen3.6-27B LoRA SFT — Task: {args.task}")
    logger.info("=" * 60)
    logger.info(f"参数: {vars(args)}")

    # 1. 加载数据
    logger.info(f"加载数据: {args.train_path}")
    raw_data = load_jsonl(args.train_path)
    logger.info(f"原始数据量: {len(raw_data)}")

    data = filter_by_task(raw_data, args.task)
    logger.info(f"过滤后 ({args.task}): {len(data)} 条")

    # 2. 加载模型
    logger.info("加载模型...")
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        use_double_quant=args.use_double_quant,
    )

    # 3. 数据处理
    train_dataset = prepare_dataset(
        data=data,
        tokenizer=tokenizer,
        task=args.task,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )

    # 4. 样本预览
    logger.info("=" * 60)
    logger.info("前 5 条训练样本预览:")
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        total_tokens = len(input_ids)

        mask_boundary = 0
        for idx, l in enumerate(labels):
            if l != IGNORE_INDEX:
                mask_boundary = idx
                break

        num_label = sum(1 for l in labels if l != IGNORE_INDEX)
        prompt_text = tokenizer.decode(input_ids[:mask_boundary], skip_special_tokens=False)
        response_text = tokenizer.decode(input_ids[mask_boundary:], skip_special_tokens=False)

        logger.info(f"--- 样本 {i} ---")
        logger.info(f"  tokens={total_tokens} | mask={total_tokens - num_label} | loss={num_label}")
        logger.info(f"  [prompt尾50]: ...{prompt_text[-50:]}")
        logger.info(f"  [response头100]: {response_text[:100]}")
        logger.info(f"  [response尾80]: ...{response_text[-80:]}")
    logger.info("=" * 60)

    # 5. 训练配置
    total_steps = (
        len(train_dataset)
        // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    ) * args.num_train_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    logger.info(f"total_steps={total_steps}, warmup_steps={warmup_steps}")

    optim = "paged_adamw_32bit" if args.use_4bit else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs={"min_lr": args.min_lr},
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        optim=optim,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[TrainingLogCallback()],
    )

    # 6. 训练
    logger.info("开始训练...")
    logger.info(f"  模式 = {'QLoRA 4-bit' if args.use_4bit else 'BF16 LoRA'}")
    logger.info(f"  任务 = {args.task}")
    logger.info(f"  总样本数 = {len(train_dataset)}")
    logger.info(f"  有效batch = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Epochs = {args.num_train_epochs}")
    logger.info(f"  LR = {args.learning_rate}, Scheduler = {args.lr_scheduler_type}, Min LR = {args.min_lr}")
    logger.info(f"  Warmup = {warmup_steps} steps ({args.warmup_ratio*100:.1f}%)")
    logger.info(f"  Grad Clip = {args.max_grad_norm}")
    logger.info(f"  Weight Decay = {args.weight_decay}")
    logger.info(f"  max_length = {args.max_length}")

    trainer.train()

    # 7. 保存
    logger.info(f"保存 LoRA adapter: {args.output_dir}")
    trainer.save_state()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("=" * 60)
    logger.info("训练完毕!")
    logger.info(f"LoRA adapter: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
