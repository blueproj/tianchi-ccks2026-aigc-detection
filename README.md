# CCKS2026 大模型生成文本检测及溯源

> 🏆 **第2名**

[English](./README_EN.md) | 中文

## 比赛简介

**比赛名称：** CCKS2026-任务六：大模型生成文本检测及溯源

**比赛平台：** [天池（阿里云）](https://tianchi.aliyun.com/competition/entrance/532474)

**主办方：** 中国科学院信息工程研究所（CCKS2026）

给定一段文本，系统需完成两步任务：

1. **文本检测（三分类）**：判断文本为纯人类撰写(0)、纯机器生成(1)或人机协作(2)
2. **来源溯源（八分类）**：对判定为纯机器生成的文本，进一步识别其模型家族（OpenAI / Alibaba / DeepSeek / ByteDance / Moonshot / Google / Anthropic / xAI）

**评测指标：**

```
FinalScore = 0.8 × F1_detect + 0.2 × F1_source
```

- F1_detect：三分类 Macro-F1（全部测试样本）
- F1_source：八分类 Macro-F1（仅真实 label=1 子集）

## 方案概述

### 核心思路

采用 **检测-溯源解耦的两阶段架构**，基于 Qwen3.6-27B 分别训练两个独立的 QLoRA adapter：Stage 1 对全部测试样本做三分类检测，Stage 2 仅对判定为纯机器生成的样本做八分类家族溯源，最后合并两阶段结果生成提交文件。

```
┌──────────────────────────────────────────────────────────┐
│                     测试文本 (15000条)                      │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │  Stage 1: Detect LoRA   │
          │  Qwen3.6-27B + QLoRA    │
          │  三分类: 0/1/2           │
          └────────────┬────────────┘
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
          │  八分类: 0-7             │
          └────────────┬────────────┘
                       │
              ┌────────┴────────┐
              │   submit.jsonl         │
              │   {id, label, family} │
              └─────────────────┘
```

### 赛题分析

**文本检测的理论依据：**

| 维度 | 人类文本 | 机器文本 | 协作文本 |
|------|---------|---------|---------|
| 句法结构 | 波动大、口语化 | 规整、句长均匀 | 介于两者之间 |
| 词汇丰富度 | 高 | 低（TTR 低） | 介于两者之间 |
| 困惑度 | 高且波动大 | 低且波动小 | 局部过度流畅 |

- 人机协作中 machine-modify-human（占协作类 41%）仅被机器润色，保留大部分人类特征，是最难检测的类别
- 大语言模型相比传统编码器能更好地捕获上下文级风格特征，对混合风格的协作文本识别更具优势

**模型家族溯源的理论依据：**

- 不同模型家族在训练数据与 RLHF 策略上的差异形成可区分的**风格指纹**（词汇偏好、标点习惯、句法模式）
- 家族风格特征具有**位置无关性**，因此采用多位置滑窗裁剪增强，迫使模型学习位置无关的家族指纹而非主题线索

### 关键技术

- **基座模型：** Qwen3.6-27B
- **微调方法：** QLoRA（4-bit NF4 量化 + 双重量化）
- **双 LoRA 架构：** detect / source 独立训练，检测用全量数据、溯源仅用 label=1 子集
- **数据增强：** 多尺度句子裁剪（检测 60%/80% 裁剪；溯源 80% + 60% 滑窗 ×3 位置）
- **LoRA 配置：** r=32, alpha=32, dropout=0.05, target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
- **训练参数：** 1 epoch, batch_size=2×4GPU, gradient_accumulation=4, lr=2e-4 (cosine_with_min_lr, min_lr=2e-5)
- **推理方式：** 贪心解码（do_sample=False），4 GPU 数据分片并行

## 成绩

| 指标 | 得分 |
|------|------|
| 检测 F1 (F1_detect) | 0.9893 |
| 溯源 F1 (F1_source) | 0.7318 |
| **总分 (FinalScore)** | **0.9378** |
| **B 榜排名** | **第2名** |

## 文件结构

```
├── code/                              # 方案代码
│   ├── preprocess/                    # 数据增强
│   │   ├── augment_detect.py          # 检测任务数据增强（句子裁剪 60%/80%）
│   │   └── augment_source.py          # 溯源任务数据增强（多尺度多位置滑窗）
│   ├── train/                         # 训练代码
│   │   ├── train_qwen.py              # QLoRA 训练脚本（detect/source 双任务）
│   │   ├── run_train_detect.sh        # 检测 LoRA 训练启动脚本
│   │   └── run_train_source.sh        # 溯源 LoRA 训练启动脚本
│   ├── infer/                         # 推理代码
│   │   ├── infer_two_stage.py         # 两阶段推理脚本（detect → source）
│   │   └── run_inference.sh           # 4 卡并行推理启动脚本
│   └── requirements.txt               # Python 依赖
├── results/                           # 提交结果
│   └── submit.jsonl                   # 最优成绩提交文件（B榜 0.9378）
├── README.md                          # 中文说明
└── README_EN.md                       # English README
```

## 环境配置

### 运行环境

| 项目 | 配置 |
|------|------|
| Python | >= 3.10 |
| CUDA | >= 12.1 |
| 训练环境 | 4 × NVIDIA RTX 4090 48GB |
| 推理环境 | 4 × NVIDIA RTX 4090 48GB |

### 安装依赖

```bash
cd code
pip install -r requirements.txt
```

### 基座模型

本方案使用 Qwen3.6-27B 作为基座模型，请下载后放置到 `models/Qwen3.6-27B` 目录下：

```bash
huggingface-cli download Qwen/Qwen3.6-27B --local-dir ./models/Qwen3.6-27B
```

### 单卡复现说明

如仅有单张 GPU（≥24GB VRAM），训练脚本会自动检测 GPU 数量并切换为单卡配置（batch_size=4, gradient_accumulation=8, max_length=1024）；推理时设置 `--num_shards 1 --shard_id 0` 串行执行即可。单卡训练时间约增加 3-4 倍，最终模型效果不受影响。

## 复现流程

### 1. 数据准备

从天池比赛页下载官方训练集 `train.jsonl`（98,800 条）与测试集 B `test_b.jsonl`（15,000 条），放置到 `data/` 目录下。

运行数据增强：

```bash
cd code
python preprocess/augment_detect.py --input ../data/train.jsonl --output ../data/augmented_detect/train_augmented.jsonl   # 检测任务增强（222K）
python preprocess/augment_source.py --input ../data/train.jsonl --output ../data/augmented_source/train_source.jsonl     # 溯源任务增强（118K）
```

### 2. 模型训练

```bash
cd code
bash train/run_train_detect.sh   # 训练检测 LoRA（最优 checkpoint-4000）
bash train/run_train_source.sh   # 训练溯源 LoRA（最优 checkpoint-2000）
```

训练参数：

- 1 epoch, per_device_batch_size=2, gradient_accumulation_steps=4（4 卡）
- learning_rate=2e-4, cosine_with_min_lr 调度（min_lr=2e-5）
- QLoRA 4-bit NF4 量化 + 双重量化, gradient_checkpointing
- 每 200 步保存 checkpoint，最多保留 10 个

### 3. 模型推理

```bash
cd code
bash infer/run_inference.sh \
    --detect_lora /path/to/detect_lora/checkpoint-4000 \
    --source_lora /path/to/source_lora/checkpoint-2000 \
    --test_path ../data/test_b.jsonl \
    --detect_output ../output/detect_results.jsonl \
    --output ../results/submit.jsonl
```

Stage 1 对全部 15,000 条样本做三分类检测，Stage 2 仅对 label=1 子集做八分类溯源，合并后输出 `submit.jsonl`（每行 `{"id": int, "label": int, "family": int}`）。

## 技术亮点

1. **检测-溯源解耦双 LoRA**：两个任务独立训练 adapter，检测用全量数据、溯源仅用 label=1 子集，避免任务间干扰
2. **多尺度句子裁剪增强**：不同保留比例（60%/80%）的裁剪增加文本长度与位置多样性，提升对 B 榜分布偏移的鲁棒性
3. **位置无关家族指纹**：溯源增强采用头部/中部/尾部三种滑窗位置，削弱文档级主题依赖，强化家族风格特征
4. **防过拟合参数配置**：LoRA r=32 低秩适配 + 1 epoch 训练，限制模型容量，避免过拟合训练集特定模式
5. **高效训练**：QLoRA 4-bit 量化 + 梯度检查点，在 4 × RTX 4090 上完成 27B 模型微调
6. **分片并行推理**：4 GPU 各自独立加载完整模型处理 1/4 数据，最大化推理吞吐量

## License

本项目仅供学习和参考使用。比赛数据版权归主办方所有。
