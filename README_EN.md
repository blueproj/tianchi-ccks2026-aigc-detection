# CCKS2026 LLM-Generated Text Detection and Source Tracing

> 🏆 **2nd Place**

中文 | [English](./README_EN.md)

## Competition Overview

**Competition:** CCKS2026 - Task 6: LLM-Generated Text Detection and Source Tracing

**Platform:** [Tianchi (Alibaba Cloud)](https://tianchi.aliyun.com/competition/entrance/532474)

**Organizer:** Institute of Information Engineering, Chinese Academy of Sciences (CCKS2026)

Given a piece of text, the system must complete two tasks:

1. **Text Detection (3-class)**: classify the text as human-written (0), machine-generated (1), or human-machine collaboration (2)
2. **Source Tracing (8-class)**: for text classified as machine-generated, identify its model family (OpenAI / Alibaba / DeepSeek / ByteDance / Moonshot / Google / Anthropic / xAI)

**Evaluation Metric:**

```
FinalScore = 0.8 × F1_detect + 0.2 × F1_source
```

- F1_detect: 3-class Macro-F1 (all test samples)
- F1_source: 8-class Macro-F1 (only the ground-truth label=1 subset)

## Solution Overview

### Core Approach

A **decoupled two-stage architecture** based on Qwen3.6-27B with two independently trained QLoRA adapters: Stage 1 performs 3-class detection on all test samples; Stage 2 performs 8-class model-family attribution only on samples detected as machine-generated; the two stages are merged into the final submission file.

```
┌──────────────────────────────────────────────────────────┐
│                    Test Text (15,000)                     │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │  Stage 1: Detect LoRA   │
          │  Qwen3.6-27B + QLoRA    │
          │  3-class: 0/1/2         │
          └────────────────────────┘
                       │
              ┌────────┴────────┐
              │  label=0 → family=-1  │
              │  label=2 → family=-1  │
              │  label=1 ↓             │
              └────────┬────────┘
                       │
          ┌────────────┴────────────┐
          │  Stage 2: Source LoRA   │
          │  Qwen3.6-27B + QLoRA    │
          │  8-class: 0-7           │
          └────────────┬────────────┘
                       │
              ┌────────┴────────┐
              │   submit.jsonl         │
              │   {id, label, family} │
              └─────────────────┘
```

### Task Analysis

**Theoretical basis for text detection:**

| Dimension | Human Text | Machine Text | Collaborative Text |
|-----------|-----------|--------------|-------------------|
| Syntax | volatile, colloquial | regular, uniform sentence length | in between |
| Lexical richness | high | low (low TTR) | in between |
| Perplexity | high, high variance | low, low variance | locally over-fluent |

- machine-modify-human (41% of collaboration class) is only machine-polished and retains most human features, making it the hardest category to detect
- LLMs capture contextual stylistic features better than traditional encoders, especially for mixed-style collaborative text

**Theoretical basis for model-family attribution:**

- Differences in training data and RLHF strategies across model families form distinguishable **style fingerprints** (lexical preferences, punctuation habits, syntactic patterns)
- Family style features are **position-independent**, hence multi-position sliding-window augmentation is used to force the model to learn position-invariant family fingerprints rather than topic cues

### Key Techniques

- **Base Model:** Qwen3.6-27B
- **Fine-tuning:** QLoRA (4-bit NF4 quantization + double quantization)
- **Dual-LoRA Architecture:** detect / source trained independently; detection uses full data, source tracing uses only the label=1 subset
- **Data Augmentation:** multi-scale sentence cropping (detection: 60%/80% crops; source: 80% + 60% sliding windows ×3 positions)
- **LoRA Config:** r=32, alpha=32, dropout=0.05, target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
- **Training:** 1 epoch, batch_size=2×4GPU, gradient_accumulation=4, lr=2e-4 (cosine_with_min_lr, min_lr=2e-5)
- **Inference:** greedy decoding (do_sample=False), 4-GPU data-shard parallelism

## Results

| Metric | Score |
|--------|-------|
| Detection F1 (F1_detect) | 0.9893 |
| Source F1 (F1_source) | 0.7318 |
| **Final Score (FinalScore)** | **0.9378** |
| **Leaderboard B Rank** | **2nd Place** |

## File Structure

```
├── code/                              # Source code
│   ├── preprocess/                    # Data augmentation
│   │   ├── augment_detect.py          # Detection augmentation (60%/80% sentence cropping)
│   │   └── augment_source.py          # Source augmentation (multi-scale sliding windows)
│   ├── train/                         # Training code
│   │   ├── train_qwen.py              # QLoRA training script (detect/source tasks)
│   │   ├── run_train_detect.sh        # Detect LoRA training launcher
│   │   └── run_train_source.sh        # Source LoRA training launcher
│   ├── infer/                         # Inference code
│   │   ├── infer_two_stage.py         # Two-stage inference script (detect → source)
│   │   └── run_inference.sh           # 4-GPU parallel inference launcher
│   └── requirements.txt               # Python dependencies
├── results/                           # Submission results
│   └── submit.jsonl                   # Best submission file (Leaderboard B 0.9378)
├── README.md                          # Chinese README
└── README_EN.md                       # English README
```

## Environment Setup

### Hardware

| Component | Configuration |
|-----------|---------------|
| Python | >= 3.10 |
| CUDA | >= 12.1 |
| Training | 4 × NVIDIA RTX 4090 48GB |
| Inference | 4 × NVIDIA RTX 4090 48GB |

### Installation

```bash
cd code
pip install -r requirements.txt
```

### Base Model

This solution uses Qwen3.6-27B as the base model. Download it and place it in the `models/Qwen3.6-27B` directory:

```bash
huggingface-cli download Qwen/Qwen3.6-27B --local-dir ./models/Qwen3.6-27B
```

### Single-GPU Reproduction

With a single GPU (≥24GB VRAM), the training scripts automatically detect the GPU count and switch to single-GPU settings (batch_size=4, gradient_accumulation=8, max_length=1024); for inference, set `--num_shards 1 --shard_id 0` to run serially. Single-GPU training takes ~3-4× longer with no impact on final model quality.

## Reproduction

### 1. Data Preparation

Download the official training set `train.jsonl` (98,800 samples) and test set B `test_b.jsonl` (15,000 samples) from the Tianchi competition page and place them in the `data/` directory.

Run data augmentation:

```bash
cd code
python preprocess/augment_detect.py --input ../data/train.jsonl --output ../data/augmented_detect/train_augmented.jsonl   # detection augmentation (222K)
python preprocess/augment_source.py --input ../data/train.jsonl --output ../data/augmented_source/train_source.jsonl     # source augmentation (118K)
```

### 2. Training

```bash
cd code
bash train/run_train_detect.sh   # train detect LoRA (best: checkpoint-4000)
bash train/run_train_source.sh   # train source LoRA (best: checkpoint-2000)
```

Training parameters:

- 1 epoch, per_device_batch_size=2, gradient_accumulation_steps=4 (4 GPUs)
- learning_rate=2e-4, cosine_with_min_lr scheduler (min_lr=2e-5)
- QLoRA 4-bit NF4 quantization + double quantization, gradient_checkpointing
- checkpoint saved every 200 steps, keeping at most 10

### 3. Inference

```bash
cd code
bash infer/run_inference.sh \
    --detect_lora /path/to/detect_lora/checkpoint-4000 \
    --source_lora /path/to/source_lora/checkpoint-2000 \
    --test_path ../data/test_b.jsonl \
    --detect_output ../output/detect_results.jsonl \
    --output ../results/submit.jsonl
```

Stage 1 runs 3-class detection on all 15,000 samples; Stage 2 runs 8-class attribution only on the label=1 subset; results are merged into `submit.jsonl` (one `{"id": int, "label": int, "family": int}` per line).

## Technical Highlights

1. **Decoupled Dual-LoRA**: two adapters trained independently — detection on full data, source tracing on the label=1 subset only — avoiding cross-task interference
2. **Multi-scale Sentence Cropping**: crops at different retention ratios (60%/80%) increase length and position diversity, improving robustness to Leaderboard B distribution shift
3. **Position-invariant Family Fingerprints**: source augmentation uses head/middle/tail sliding windows to weaken document-level topic dependency and strengthen family style features
4. **Overfitting-prevention Config**: LoRA r=32 low-rank adaptation + 1 epoch training limits model capacity and avoids overfitting training-set-specific patterns
5. **Efficient Training**: QLoRA 4-bit quantization + gradient checkpointing enables 27B fine-tuning on 4 × RTX 4090
6. **Shard-parallel Inference**: 4 GPUs each load the full model and process 1/4 of the data, maximizing throughput

## License

This project is for educational and reference purposes only. Competition data is owned by the organizer.
