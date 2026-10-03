"""
溯源任务数据增强脚本
==========================
仅对 label=1 文本做句子裁剪，每条生成:
  1. 原文 (保留全文, 与推理时长度一致)
  2. 中间 80% (去首尾各 10%, 即 crop_80)
  3. 60% 滑窗-头部 (0%-60%)
  4. 60% 滑窗-中部 (20%-80%)
  5. 60% 滑窗-尾部 (40%-100%)

共约 5× 扩增。多尺度 + 多位置, 削弱主题/格式依赖, 强化位置无关的家族指纹。

重要:
  - 保留全部原始文本特征 (含全角字符等家族 Unicode 指纹)
  - 仅 label=1, 标签安全 (family/model 不变)
  - 设最小阈值避免片段过短丢失家族信号

用法:
  python augment_source.py --input data/train.jsonl --output data/augmented_source/train_source.jsonl
"""

import json
import re
import os
import argparse
from collections import Counter


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# 句子分割
# ============================================================

def split_into_sentences(text):
    """按句子边界分割文本。"""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# ============================================================
# 裁剪函数
# ============================================================

MIN_SENTENCES = 6        # 少于 6 句不做裁剪
MIN_CHARS = 300          # 裁剪后少于 300 字符则丢弃


def crop_middle_80(sentences):
    """中间 80%: 去首尾各 10%。"""
    n = len(sentences)
    if n < MIN_SENTENCES:
        return None
    start = max(1, n // 10)
    end = n - n // 10
    if end - start < 3:
        return None
    return " ".join(sentences[start:end])


def crop_60_window(sentences, start_frac):
    """60% 滑窗: 从 start_frac 开始取 60%。"""
    n = len(sentences)
    if n < MIN_SENTENCES:
        return None
    start = int(n * start_frac)
    end = start + int(n * 0.6)
    # 边界保护
    start = max(0, min(start, n - 3))
    end = max(start + 3, min(end, n))
    if end - start < 3:
        return None
    return " ".join(sentences[start:end])


def gen_variants(text):
    """对单条文本生成所有裁剪变体, 返回 [(variant_text, aug_source), ...]。"""
    sentences = split_into_sentences(text)
    n = len(sentences)
    variants = []

    if n < MIN_SENTENCES:
        # 句子太少, 只保留原文, 不裁剪
        return variants

    # 1. 中间 80%
    v = crop_middle_80(sentences)
    if v and len(v) >= MIN_CHARS and v != text:
        variants.append((v, "crop_mid80"))

    # 2. 60% 滑窗 - 头部 (0%)
    v = crop_60_window(sentences, 0.0)
    if v and len(v) >= MIN_CHARS and v != text:
        variants.append((v, "crop_60_head"))

    # 3. 60% 滑窗 - 中部 (20%)
    v = crop_60_window(sentences, 0.2)
    if v and len(v) >= MIN_CHARS and v != text:
        variants.append((v, "crop_60_mid"))

    # 4. 60% 滑窗 - 尾部 (40%)
    v = crop_60_window(sentences, 0.4)
    if v and len(v) >= MIN_CHARS and v != text:
        variants.append((v, "crop_60_tail"))

    return variants


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="溯源任务数据增强 (多尺度裁剪)")
    parser.add_argument("--input", type=str, required=True, help="原始训练集路径 (含 label=1)")
    parser.add_argument("--output", type=str, required=True, help="增强后输出路径")
    args = parser.parse_args()

    print("=" * 60)
    print("溯源任务数据增强 (多尺度裁剪)")
    print("=" * 60)

    # 加载原始数据
    data = load_jsonl(args.input)
    print(f"原始数据: {len(data)} 条")

    # 仅保留 label=1 且有 family
    source_data = [d for d in data if d.get("label") == 1 and "family" in d]
    print(f"label=1 (有 family): {len(source_data)} 条")

    # 家族分布
    family_dist = Counter(d.get("family") for d in source_data)
    print(f"家族分布:")
    for f, c in sorted(family_dist.items()):
        print(f"  family={f}: {c}")

    # 生成增强
    all_data = []
    aug_counts = Counter()

    for item in source_data:
        # 原文
        orig = dict(item)
        orig["aug_source"] = "original"
        all_data.append(orig)
        aug_counts["original"] += 1

        # 裁剪变体
        variants = gen_variants(item["text"])
        for v_text, v_source in variants:
            new_item = dict(item)
            new_item["text"] = v_text
            new_item["aug_source"] = v_source
            new_item["id"] = f"{item['id']}_{v_source[:3]}"
            all_data.append(new_item)
            aug_counts[v_source] += 1

    print(f"\n增强前去重: {len(all_data)} 条")
    print("各增强来源:")
    for src, c in sorted(aug_counts.items()):
        print(f"  {src}: {c}")

    # 精确去重
    seen = set()
    deduped = []
    for item in all_data:
        if item["text"] not in seen:
            seen.add(item["text"])
            deduped.append(item)
    removed = len(all_data) - len(deduped)
    print(f"去重后: {len(deduped)} 条 (去除 {removed} 条重复)")

    # 最终分布
    final_aug = Counter(d["aug_source"] for d in deduped)
    print(f"\n最终增强来源分布:")
    for src, c in sorted(final_aug.items()):
        print(f"  {src}: {c} ({c/len(deduped)*100:.1f}%)")

    final_family = Counter(d.get("family") for d in deduped)
    print(f"\n最终家族分布:")
    for f, c in sorted(final_family.items()):
        print(f"  family={f}: {c}")

    # 保存
    save_jsonl(deduped, args.output)
    print(f"\n输出: {args.output}")

    # 保存统计报告
    report_path = args.output.replace(".jsonl", "_report.json")
    report = {
        "version": "source_v3",
        "input_count": len(data),
        "source_count": len(source_data),
        "before_dedup": len(all_data),
        "after_dedup": len(deduped),
        "removed_duplicates": removed,
        "aug_source_dist": dict(sorted(final_aug.items())),
        "family_dist": {str(k): v for k, v in sorted(final_family.items())},
        "methods": {
            "original": "原文",
            "crop_mid80": "中间80% (去首尾各10%)",
            "crop_60_head": "60%滑窗-头部 (0%-60%)",
            "crop_60_mid": "60%滑窗-中部 (20%-80%)",
            "crop_60_tail": "60%滑窗-尾部 (40%-100%)",
        },
        "min_sentences": MIN_SENTENCES,
        "min_chars": MIN_CHARS,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"统计报告: {report_path}")

    print(f"\n{'='*60}")
    print("溯源任务数据增强 完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
