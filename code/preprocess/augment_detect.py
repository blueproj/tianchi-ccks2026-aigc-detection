"""
检测任务数据增强脚本
================
增强方法:
  1. 句子裁剪60% (label 0/1 + label=2 modify-human)
  2. 句子裁剪80% (全部 label 0/1/2)
  3. 合并 + 精确去重

改进点:
  - 80%温和裁剪
  - label=2 machine-modify-human 参与裁剪
  - label=2 continue-human/mix 参与保守80%裁剪

用法:
  python augment_detect.py --input data/train.jsonl --output data/augmented_detect/train_augmented.jsonl
"""

import json
import re
import os
import argparse
from collections import Counter


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ============================================================
# 1. 句子分割
# ============================================================

def split_into_sentences(text):
    """按句子边界分割文本。"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# ============================================================
# 2. 句子裁剪 60% (去首尾各20%)
# ============================================================

def crop_60(text, min_sentences=5):
    """去掉首尾各 20% 句子，保留中间 60%。"""
    sentences = split_into_sentences(text)
    n = len(sentences)
    if n < min_sentences:
        return None

    start = n // 5  # 去掉前 20%
    end = n - n // 5  # 去掉后 20%

    if end - start < 3:
        return None

    cropped = ' '.join(sentences[start:end])
    return cropped.strip() if cropped != text else None


# ============================================================
# 3. 句子裁剪 80% (去首尾各 max(1, n//10))
# ============================================================

def crop_80(text, min_sentences=5):
    """去掉首尾各 max(1, n//10) 句，保留中间 ~80%。"""
    sentences = split_into_sentences(text)
    n = len(sentences)
    if n < min_sentences:
        return None

    trim = max(1, n // 10)
    start = trim
    end = n - trim

    if end - start < 3:
        return None

    cropped = ' '.join(sentences[start:end])
    return cropped.strip() if cropped != text else None


# ============================================================
# 4. 判断 label=2 的 method 类型
# ============================================================

def is_modify_human(item):
    """machine-modify-human: 机器润色人类文本"""
    method = item.get('method', '')
    return 'modify' in method

def is_continue_human(item):
    """machine-continue-human: 人类前缀+机器续写"""
    method = item.get('method', '')
    return 'continue' in method

def is_mix_machine(item):
    """human-mix-machine: 人类与机器段落交替"""
    method = item.get('method', '')
    return 'mix' in method


# ============================================================
# 5. 增强函数
# ============================================================

def augment_crop_60(data):
    """句子裁剪60%: label 0/1 + label=2 modify-human"""
    results = []
    for item in data:
        label = item.get('label')
        should_crop = False

        if label in (0, 1):
            should_crop = True
        elif label == 2 and is_modify_human(item):
            should_crop = True

        if not should_crop:
            continue

        cropped = crop_60(item['text'])
        if cropped and len(cropped) > 50:
            new_item = dict(item)
            new_item['text'] = cropped
            new_item['aug_source'] = 'crop_60'
            new_item['id'] = f"{item['id']}_c6"
            results.append(new_item)
    return results


def augment_crop_80(data):
    """句子裁剪80%: 全部 label 0/1/2"""
    results = []
    for item in data:
        label = item.get('label')

        # label=2 的 continue-human 和 mix-machine 只做80%保守裁剪
        # label=2 的 modify-human 也做80%
        # label 0/1 都做
        if label not in (0, 1, 2):
            continue

        cropped = crop_80(item['text'])
        if cropped and len(cropped) > 50:
            new_item = dict(item)
            new_item['text'] = cropped
            new_item['aug_source'] = 'crop_80'
            new_item['id'] = f"{item['id']}_c8"
            results.append(new_item)
    return results


# ============================================================
# 6. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='检测任务数据增强')
    parser.add_argument('--input', type=str, required=True, help='原始训练集路径')
    parser.add_argument('--output', type=str, required=True, help='增强后输出路径')
    args = parser.parse_args()

    print('=' * 60)
    print('检测任务数据增强')
    print('=' * 60)

    # 加载原始数据
    data = load_jsonl(args.input)
    print(f'原始数据: {len(data)} 条')

    label_dist = Counter(d.get('label') for d in data)
    print(f'Label 分布: {dict(sorted(label_dist.items()))}')

    # 统计 label=2 的 method 分布
    l2_methods = Counter(d.get('method', 'N/A') for d in data if d.get('label') == 2)
    print(f'Label=2 Method 分布:')
    for m, c in l2_methods.most_common():
        print(f'  {m}: {c}')

    # 1. 原始数据
    original = []
    for item in data:
        new_item = dict(item)
        new_item['aug_source'] = 'original'
        original.append(new_item)
    print(f'\n原始数据: {len(original)} 条')

    # 2. 句子裁剪 60%
    print('\n--- 句子裁剪 60% (label 0/1 + modify-human) ---')
    cropped_60 = augment_crop_60(data)
    print(f'  生成 {len(cropped_60)} 条')

    # 分label统计
    c60_label = Counter(d.get('label') for d in cropped_60)
    for l, c in sorted(c60_label.items()):
        print(f'    label={l}: {c}')

    # 3. 句子裁剪 80%
    print('\n--- 句子裁剪 80% (全部 label 0/1/2) ---')
    cropped_80 = augment_crop_80(data)
    print(f'  生成 {len(cropped_80)} 条')

    # 分label统计
    c80_label = Counter(d.get('label') for d in cropped_80)
    for l, c in sorted(c80_label.items()):
        method_dist = Counter(d.get('method', 'N/A') for d in cropped_80 if d.get('label') == l)
        method_str = ', '.join(f'{m}={mc}' for m, mc in method_dist.most_common())
        print(f'    label={l}: {c} ({method_str})')

    # 4. 合并
    all_data = original + cropped_60 + cropped_80
    print(f'\n合并前: {len(all_data)} 条')

    # 5. 精确去重
    seen = set()
    deduped = []
    for item in all_data:
        if item['text'] not in seen:
            seen.add(item['text'])
            deduped.append(item)
    removed = len(all_data) - len(deduped)
    print(f'去重后: {len(deduped)} 条 (去除 {removed} 条重复)')

    # 统计
    aug_dist = Counter(d['aug_source'] for d in deduped)
    print(f'\n增强来源分布:')
    for src, count in sorted(aug_dist.items()):
        print(f'  {src}: {count} ({count/len(deduped)*100:.1f}%)')

    label_dist_final = Counter(d.get('label') for d in deduped)
    print(f'\n最终 Label 分布:')
    for l, c in sorted(label_dist_final.items()):
        print(f'  label={l}: {c}')

    source_count = sum(1 for d in deduped if d.get('label') == 1)
    print(f'\nSource 训练样本 (label=1): {source_count}')

    # 保存
    save_jsonl(deduped, args.output)
    print(f'\n输出: {args.output}')

    # 保存统计报告
    report_path = args.output.replace('.jsonl', '_report.json')
    report = {
        'original_count': len(data),
        'crop_60_count': len(cropped_60),
        'crop_80_count': len(cropped_80),
        'before_dedup': len(all_data),
        'after_dedup': len(deduped),
        'removed_duplicates': removed,
        'aug_source_dist': dict(sorted(aug_dist.items())),
        'label_dist': {str(k): v for k, v in sorted(label_dist_final.items())},
        'source_train_count': source_count,
        'crop_60_by_label': {str(k): v for k, v in sorted(c60_label.items())},
        'crop_80_by_label': {str(k): v for k, v in sorted(c80_label.items())},
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'统计报告: {report_path}')

    print(f'\n{"="*60}')
    print('检测任务数据增强 完成!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
