# 基于语音自监督表征的低资源 ASR 项目规划文档

> 用途：本文件用于指导 5 天内完成课程大作业，实现一个以语音自监督表征为核心的低资源英文 ASR 系统，并形成可复现实验与 ICASSP 风格论文报告。

---

## 0. 项目结论与范围锁定

### 0.1 最终选择

本项目选择：

```text
方向一：基于语音自监督表征的低资源 ASR
```

推荐题目：

```text
Low-resource English ASR with Self-supervised Speech Representations
```

中文题目：

```text
基于语音自监督表征的低资源英文语音识别系统设计与消融分析
```

### 0.2 五天内不做的内容

为控制工程风险，本项目不做以下内容：

```text
1. 不做 TTS。
2. 不从零训练大型 ASR 模型。
3. 不使用 large 级别 SSL 模型作为主实验。
4. 不把离散单元、K-means、codebook size、BPE 作为主线。
5. 不使用已经在 LibriSpeech ASR 上监督微调好的模型作为主结果。
6. 不追求 leaderboard 级别 WER，只追求完整、可解释、可复现。
```

### 0.3 最终方案

稳妥版 + 稍微增强版：

```text
数据：LibriSpeech train-clean-100 的低资源子集
任务：英文 ASR
主模型：facebook/wav2vec2-base + CTC
Baseline：frozen wav2vec2-base + CTC head
主实验：fine-tuned wav2vec2-base + CTC
消融 1：1h vs 10h 训练数据规模
消融 2：frozen vs fine-tune
增强实验：wav2vec2-base vs WavLM-base 或 HuBERT-base
指标：WER, CER, training time, inference RTF, parameter count
分析：错误案例分析，训练成本与识别性能关系
```

---

## 1. 开源数据、模型与代码使用原则

### 1.1 学术诚信边界

以下做法是合规的：

```text
1. 使用公开数据集 LibriSpeech。
2. 使用 Hugging Face / torchaudio / OpenSLR 下载或加载数据。
3. 使用开源代码库作为训练框架或参考实现。
4. 使用自监督预训练模型作为 encoder 初始化。
5. 自己进行低资源子集构造、模型微调、评测和结果分析。
6. 在论文和 README 中明确说明数据、模型和代码来源。
```

以下做法不合规或高风险：

```text
1. 直接下载别人已经 fine-tuned 的 ASR 模型，跑 test 后当作自己训练结果。
2. 复制开源项目 README、论文文字或实验结果作为自己的报告内容。
3. 不说明使用了哪个开源代码库、哪个预训练 checkpoint、哪个数据划分。
4. 用 test-clean 调参，然后仍将 test-clean 作为最终测试集。
5. 把公开 leaderboard 数字写成自己实验结果。
```

论文中建议写明：

```text
Our implementation is based on the Hugging Face Transformers CTC speech recognition example. We modify the data sampling, training configuration, evaluation script, and ablation settings for low-resource ASR experiments.
```

---

## 2. 推荐开源资源

### 2.1 数据集

#### 主数据集：LibriSpeech ASR Corpus

推荐使用：

```text
train-clean-100
validation / dev-clean
test / test-clean
```

用途：

| 数据划分 | 用途 |
|---|---|
| train-clean-100 | 构造 1h / 10h 低资源训练子集 |
| dev-clean / validation | 验证、选择 checkpoint |
| test-clean / test | 最终测试 |

官方来源：

```text
https://www.openslr.org/12
```

Hugging Face 数据集页：

```text
https://huggingface.co/datasets/openslr/librispeech_asr
```

工程建议：

```text
首选：torchaudio.datasets.LIBRISPEECH 或 OpenSLR 官方下载
可选：Hugging Face datasets 加载
```

说明：

- 如果 Hugging Face `openslr/librispeech_asr` 加载遇到版本兼容问题，立即切换到 `torchaudio.datasets.LIBRISPEECH` 或 OpenSLR 手动下载。
- 不要在数据下载问题上消耗太久。

---

### 2.2 自监督预训练模型

#### 主模型：facebook/wav2vec2-base

用途：主实验和 baseline。

```text
https://huggingface.co/facebook/wav2vec2-base
```

使用理由：

```text
1. 自监督预训练语音模型，符合课程要求。
2. Hugging Face 支持完善。
3. 训练和微调资料最多，5 天内最稳。
4. base 模型算力要求可控。
```

注意：

```text
使用 facebook/wav2vec2-base，而不是 facebook/wav2vec2-base-960h。
```

区别：

| 模型 | 是否推荐作为主模型 | 原因 |
|---|---|---|
| facebook/wav2vec2-base | 推荐 | 自监督预训练模型，需要自己 fine-tune |
| facebook/wav2vec2-base-960h | 不推荐作为主结果 | 已经在 ASR 数据上监督微调，容易削弱作业独立性 |

---

#### 增强模型 A：microsoft/wavlm-base

用途：可选的不同 SSL 模型对比。

```text
https://huggingface.co/microsoft/wavlm-base
```

推荐程度：高于 HuBERT，作为唯一增强模型即可。

---

#### 增强模型 B：facebook/hubert-base-ls960

用途：可选的不同 SSL 模型对比。

```text
https://huggingface.co/facebook/hubert-base-ls960
```

注意：

```text
HuBERT base 是预训练模型，本身没有 ASR tokenizer；需要用 CTC 训练脚本构造字符级 vocabulary 并 fine-tune。
```

---

### 2.3 代码库

#### 主代码库：Hugging Face Transformers

推荐使用官方 CTC 语音识别脚本作为主工程基础：

```text
https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py
```

代码库：

```bash
git clone https://github.com/huggingface/transformers.git
```

建议基于以下目录改造：

```text
transformers/examples/pytorch/speech-recognition/
```

主要改造点：

```text
1. 低资源子集构造：1h / 10h。
2. 文本规范化。
3. frozen vs fine-tune 开关。
4. WER / CER 同时输出。
5. prediction 保存。
6. RTF 和训练成本统计。
7. metrics.csv 汇总。
```

---

#### 备用代码库：SpeechBrain

用途：参考 ASR recipe，不作为主线。

```text
https://github.com/speechbrain/speechbrain
```

---

#### 备用代码库：S3PRL

用途：如果后续想做更像“表征分析”的 frozen feature 实验，可参考 S3PRL upstream/downstream 设计。

```text
https://github.com/s3prl/s3prl
```

五天内不建议主用 S3PRL。

---

### 2.4 评测工具

推荐：

```text
jiwer
```

用途：

```text
WER, CER, substitution/deletion/insertion 相关分析
```

安装：

```bash
pip install jiwer
```

代码仓库：

```text
https://github.com/jitsi/jiwer
```

---

## 3. Startup：第一天必须完成的最小闭环

### 3.1 环境创建

建议环境：

```bash
conda create -n ssl_asr python=3.10 -y
conda activate ssl_asr

pip install torch torchaudio
pip install transformers datasets evaluate jiwer accelerate soundfile librosa pandas numpy tqdm tensorboard
```

如果使用 CUDA 版本 PyTorch，请按机器 CUDA 版本从 PyTorch 官网选择安装命令。

---

### 3.2 创建项目仓库

建议项目结构：

```text
ssl_low_resource_asr/
├── README.md
├── requirements.txt
├── configs/
│   ├── wav2vec2_frozen_10h.yaml
│   ├── wav2vec2_finetune_1h.yaml
│   ├── wav2vec2_finetune_10h.yaml
│   └── wavlm_finetune_10h.yaml
├── scripts/
│   ├── prepare_librispeech_subsets.py
│   ├── train_wav2vec2_frozen_10h.sh
│   ├── train_wav2vec2_finetune_1h.sh
│   ├── train_wav2vec2_finetune_10h.sh
│   ├── train_wavlm_finetune_10h.sh
│   ├── eval_all.sh
│   └── collect_results.py
├── src/
│   ├── text_normalize.py
│   ├── metrics.py
│   ├── error_analysis.py
│   ├── measure_rtf.py
│   └── utils.py
├── data/
│   ├── raw/
│   ├── manifests/
│   └── vocab/
├── exp/
│   ├── wav2vec2_frozen_10h/
│   ├── wav2vec2_finetune_1h/
│   ├── wav2vec2_finetune_10h/
│   └── wavlm_finetune_10h/
├── results/
│   ├── metrics.csv
│   ├── predictions/
│   ├── error_cases.md
│   └── figures/
└── paper/
    ├── main.tex
    ├── refs.bib
    └── figures/
```

---

### 3.3 数据准备

优先使用 torchaudio：

```python
import torchaudio

train_set = torchaudio.datasets.LIBRISPEECH(
    root="./data/raw",
    url="train-clean-100",
    download=True,
)

dev_set = torchaudio.datasets.LIBRISPEECH(
    root="./data/raw",
    url="dev-clean",
    download=True,
)

test_set = torchaudio.datasets.LIBRISPEECH(
    root="./data/raw",
    url="test-clean",
    download=True,
)
```

需要生成 manifest 文件：

```jsonl
{"audio_path": ".../xxx.flac", "text": "...", "duration": 7.34, "speaker_id": "...", "chapter_id": "...", "utt_id": "..."}
```

低资源子集构造策略：

```text
1. 固定随机种子 seed=42。
2. 从 train-clean-100 中随机打乱 utterance。
3. 按累计 duration 抽取到 1h 和 10h。
4. dev/test 不参与训练抽样。
5. 保存 manifest，保证可复现。
```

输出：

```text
data/manifests/train_1h.jsonl
data/manifests/train_10h.jsonl
data/manifests/dev_clean.jsonl
data/manifests/test_clean.jsonl
```

---

### 3.4 文本规范化

统一规则：

```text
1. 转小写。
2. 删除标点符号。
3. 多空格压缩成单空格。
4. 去除首尾空格。
5. 保留英文空格作为 CTC token。
```

建议字符级 vocabulary：

```text
a b c ... z
'
|
[UNK]
[PAD]
```

其中 `|` 表示空格。

---

### 3.5 第一天 debug 目标

第一天必须跑通：

```text
1. 数据 manifest 可生成。
2. wav2vec2-base 能加载。
3. CTC 训练脚本能在 100～200 条样本上跑完 1 个 epoch。
4. dev 上能输出 prediction。
5. 能计算 WER 和 CER。
```

debug 命令可以先用小样本：

```bash
python run_speech_recognition_ctc.py \
  --model_name_or_path facebook/wav2vec2-base \
  --dataset_name <your_local_or_hf_dataset> \
  --output_dir ./exp/debug_wav2vec2 \
  --overwrite_output_dir \
  --max_train_samples 200 \
  --max_eval_samples 100 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-4 \
  --num_train_epochs 1 \
  --fp16 \
  --do_train \
  --do_eval
```

如果使用自定义 local manifest，需要 Codex 将官方脚本的数据加载部分改造成读取 JSONL manifest。

---

## 4. 系统 Pipeline

### 4.1 主 pipeline

```text
Raw waveform, 16kHz
        ↓
Text normalization + audio loading
        ↓
Pretrained SSL encoder
facebook/wav2vec2-base
        ↓
Frame-level continuous speech representations
        ↓
Linear CTC classification head
        ↓
CTC loss training
        ↓
Greedy decoding
        ↓
Predicted transcription
        ↓
WER / CER / error analysis
```

---

### 4.2 模型结构

主模型：

```text
Wav2Vec2 encoder + dropout + linear projection + CTC loss
```

输入：

```text
16kHz waveform
```

输出：

```text
character-level posterior sequence
```

解码：

```text
argmax over frame-level logits
remove repeated tokens
remove blank token
replace | with whitespace
```

---

### 4.3 Baseline 与主模型区别

| 系统 | Encoder | CTC Head | 训练参数 | 目的 |
|---|---|---|---|---|
| Baseline | frozen wav2vec2-base | trainable | 少 | 验证固定 SSL 表征是否足够 |
| Main | fine-tuned wav2vec2-base | trainable | 多 | 验证端到端微调是否提升低资源 ASR |

---

## 5. 实验规划

### 5.1 总体实验矩阵

#### P0：必须完成

| 实验编号 | 名称 | 模型 | 数据 | 训练策略 | 指标 |
|---|---|---|---:|---|---|
| E1 | Frozen SSL baseline | wav2vec2-base | 10h | freeze encoder, train CTC head | WER, CER |
| E2 | Fine-tuned main model | wav2vec2-base | 10h | fine-tune encoder + CTC | WER, CER |
| E3 | Low-resource scale | wav2vec2-base | 1h | fine-tune encoder + CTC | WER, CER |

#### P1：稍微增强版

| 实验编号 | 名称 | 模型 | 数据 | 训练策略 | 指标 |
|---|---|---|---:|---|---|
| E4 | SSL model comparison | WavLM-base 或 HuBERT-base | 10h | fine-tune encoder + CTC | WER, CER |

建议只选择一个增强模型：

```text
优先 WavLM-base；如果 WavLM 训练脚本不顺利，再换 HuBERT-base。
```

---

### 5.2 训练超参数建议

#### 8GB GPU 保守配置

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
fp16: true
gradient_checkpointing: true
learning_rate: 3e-4
num_train_epochs: 3
freeze_feature_encoder: true
max_duration_in_seconds: 15
save_total_limit: 2
```

#### 12GB～16GB GPU 推荐配置

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
fp16: true
gradient_checkpointing: true
learning_rate: 3e-4
num_train_epochs: 5
freeze_feature_encoder: true
max_duration_in_seconds: 20
save_total_limit: 2
```

#### 24GB GPU 稳定配置

```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
fp16: true
gradient_checkpointing: true
learning_rate: 3e-4
num_train_epochs: 5
freeze_feature_encoder: true
max_duration_in_seconds: 20
save_total_limit: 3
```

说明：

```text
freeze_feature_encoder 只冻结最底层 CNN feature encoder，不等于冻结整个 wav2vec2 encoder。
如果要做 frozen SSL baseline，需要冻结整个 wav2vec2 encoder，只训练 CTC head。
```

---

### 5.3 Frozen baseline 的实现要求

Frozen baseline 应冻结除 CTC head 之外的参数。

检查代码：

```python
for name, param in model.named_parameters():
    if "lm_head" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

训练前打印：

```text
total parameters
trainable parameters
trainable ratio
```

这也是论文中“模型复杂度和训练成本分析”的数据来源。

---

### 5.4 Fine-tune 主模型的实现要求

Fine-tune 主模型允许训练 Transformer encoder 和 CTC head。

推荐仍然冻结底层 feature encoder：

```python
model.freeze_feature_encoder()
```

原因：

```text
1. 更省显存。
2. 训练更稳定。
3. 低资源场景下可以降低过拟合风险。
```

---

### 5.5 评价指标

必须报告：

| 指标 | 含义 | 工具 |
|---|---|---|
| WER | Word Error Rate | jiwer |
| CER | Character Error Rate | jiwer |

建议报告：

| 指标 | 含义 | 计算方式 |
|---|---|---|
| Training time | 训练耗时 | 训练日志记录 |
| RTF | Real-time factor | inference time / audio duration |
| Params | 参数量 | PyTorch 统计 |
| Trainable params | 可训练参数量 | PyTorch 统计 |

RTF 计算：

```text
RTF = total_inference_wall_time / total_audio_duration
```

---

## 6. 结果表格模板

### 6.1 主结果表

| Method | Train Data | Encoder | Training Strategy | WER ↓ | CER ↓ |
|---|---:|---|---|---:|---:|
| wav2vec2-base + CTC | 10h | wav2vec2-base | frozen encoder | TBD | TBD |
| wav2vec2-base + CTC | 10h | wav2vec2-base | fine-tune | TBD | TBD |
| wav2vec2-base + CTC | 1h | wav2vec2-base | fine-tune | TBD | TBD |
| WavLM-base + CTC | 10h | WavLM-base | fine-tune | TBD | TBD |

---

### 6.2 训练成本表

| Method | Train Data | Params | Trainable Params | GPU | Train Time | RTF | WER ↓ |
|---|---:|---:|---:|---|---:|---:|---:|
| frozen wav2vec2 | 10h | TBD | TBD | TBD | TBD | TBD | TBD |
| fine-tuned wav2vec2 | 10h | TBD | TBD | TBD | TBD | TBD | TBD |
| fine-tuned wav2vec2 | 1h | TBD | TBD | TBD | TBD | TBD | TBD |
| fine-tuned WavLM | 10h | TBD | TBD | TBD | TBD | TBD | TBD |

---

### 6.3 错误案例分析表

| ID | Reference | Prediction | Error Type | Analysis |
|---|---|---|---|---|
| TBD | TBD | TBD | substitution | 发音相近词混淆 |
| TBD | TBD | TBD | deletion | 短功能词漏识别 |
| TBD | TBD | TBD | insertion | CTC 解码产生冗余词 |
| TBD | TBD | TBD | long utterance | 长句上下文不足 |
| TBD | TBD | TBD | rare word | 低资源数据覆盖不足 |

---

## 7. 五天执行计划

### Day 1：Startup 与最小闭环

目标：跑通 debug 训练和 WER/CER。

任务：

```text
1. 建 repo。
2. 配环境。
3. 下载或加载 LibriSpeech train-clean-100 / dev-clean / test-clean。
4. 生成 1h / 10h manifest。
5. 跑通 wav2vec2-base CTC debug 训练。
6. dev 上输出 prediction。
7. 计算 WER / CER。
```

验收标准：

```text
1. exp/debug_wav2vec2/ 中有训练日志。
2. results/predictions/debug_dev_predictions.jsonl 存在。
3. results/metrics.csv 中有至少一行 WER/CER。
```

---

### Day 2：完成 P0 的 E1/E2

目标：完成 frozen baseline 和 10h fine-tune 主模型。

任务：

```text
1. 训练 E1：wav2vec2-base frozen + CTC, 10h。
2. 训练 E2：wav2vec2-base fine-tune + CTC, 10h。
3. 保存 checkpoint、日志、prediction、metrics。
4. 统计参数量、可训练参数量、训练时间。
```

验收标准：

```text
1. E1 有 WER/CER。
2. E2 有 WER/CER。
3. E2 相比 E1 最好有下降；若无下降，也要保留结果并分析可能原因。
```

---

### Day 3：完成低资源规模消融 E3

目标：完成 1h vs 10h 的核心消融。

任务：

```text
1. 训练 E3：wav2vec2-base fine-tune + CTC, 1h。
2. 和 E2 比较 WER/CER。
3. 整理低资源数据规模影响图表。
4. 开始写实验设置和方法部分。
```

验收标准：

```text
1. E3 有 WER/CER。
2. 主结果表至少有 E1/E2/E3 三行。
3. 能写出“数据规模影响”的结论。
```

---

### Day 4：增强实验 E4 与错误分析

目标：补一个不同 SSL 模型对比，并完成错误分析。

任务：

```text
1. 尝试训练 E4：WavLM-base 或 HuBERT-base, 10h。
2. 如果 E4 失败，不继续死磕，改为补充 RTF/错误案例分析。
3. 保存 5～10 条典型错误案例。
4. 完成论文初稿中的 Results and Analysis。
```

验收标准：

```text
1. 有 E4 结果，或有完整错误分析和训练成本分析。
2. paper/main.tex 主体结构完整。
3. results/metrics.csv 完整。
```

---

### Day 5：收尾与提交

目标：不再做新实验，只做复现、排版、打包。

任务：

```text
1. 整理 README。
2. 检查所有脚本路径。
3. 补充 requirements.txt。
4. 整理 checkpoints 和 results。
5. 编译 ICASSP 格式 PDF。
6. 打包提交材料。
```

Day 5 禁止：

```text
1. 切换主模型。
2. 新开大型实验。
3. 改成 TTS。
4. 大规模重构代码。
5. 尝试离散单元主线。
```

---

## 8. Codex 工作拆分建议

### 8.1 Codex 任务 1：创建项目骨架

Prompt：

```text
Create a Python project skeleton for a low-resource ASR project using self-supervised speech representations. The project should include folders for configs, scripts, src, data/manifests, results, exp, and paper. Add placeholder README.md, requirements.txt, and basic Python modules for text normalization, metrics, error analysis, RTF measurement, and utility functions.
```

验收：

```text
项目目录完整，无语法错误。
```

---

### 8.2 Codex 任务 2：实现 LibriSpeech manifest 生成

Prompt：

```text
Implement scripts/prepare_librispeech_subsets.py. It should use torchaudio.datasets.LIBRISPEECH to download or load train-clean-100, dev-clean, and test-clean. It should compute duration for each utterance, normalize the transcript, and write JSONL manifests for train_1h.jsonl, train_10h.jsonl, dev_clean.jsonl, and test_clean.jsonl. Use a fixed random seed and sample training utterances until the accumulated duration reaches the target hours. Include CLI arguments for data root, output dir, seed, and target hours.
```

验收：

```text
python scripts/prepare_librispeech_subsets.py --data_root ./data/raw --manifest_dir ./data/manifests
```

输出：

```text
data/manifests/train_1h.jsonl
data/manifests/train_10h.jsonl
data/manifests/dev_clean.jsonl
data/manifests/test_clean.jsonl
```

---

### 8.3 Codex 任务 3：实现文本规范化

Prompt：

```text
Implement src/text_normalize.py for English ASR. It should lowercase text, remove punctuation except apostrophe if configured, collapse multiple spaces, strip leading/trailing spaces, and provide a function normalize_text(text: str) -> str. Add simple unit tests or a __main__ demo.
```

验收：

```text
"THIS, IS A TEST." -> "this is a test"
```

---

### 8.4 Codex 任务 4：实现 WER/CER 和 prediction 保存

Prompt：

```text
Implement src/metrics.py using jiwer to compute WER and CER from lists of references and predictions. Implement a function save_predictions(path, records) that writes JSONL records with id, reference, prediction, audio_path, and duration. Implement a command-line script scripts/collect_results.py that reads prediction JSONL files, computes WER/CER, and appends results to results/metrics.csv.
```

验收：

```text
python scripts/collect_results.py --pred results/predictions/example.jsonl --method debug --out results/metrics.csv
```

---

### 8.5 Codex 任务 5：改造 Hugging Face CTC 训练脚本

Prompt：

```text
Adapt Hugging Face Transformers examples/pytorch/speech-recognition/run_speech_recognition_ctc.py for local JSONL manifests. The script should load audio_path and text from JSONL, resample audio to 16kHz if necessary, build or load a character-level CTC vocabulary, train Wav2Vec2ForCTC or WavLMForCTC, support full fine-tuning and fully frozen encoder modes, save predictions on validation/test sets, and compute WER/CER. Keep CLI arguments for model_name_or_path, train_manifest, eval_manifest, test_manifest, output_dir, freeze_encoder, per_device_train_batch_size, gradient_accumulation_steps, fp16, num_train_epochs, and learning_rate.
```

验收：

```bash
bash scripts/train_wav2vec2_finetune_1h.sh
```

必须产生：

```text
exp/<experiment_name>/
results/predictions/<experiment_name>_test.jsonl
results/metrics.csv
```

---

### 8.6 Codex 任务 6：实现错误分析

Prompt：

```text
Implement src/error_analysis.py and scripts/error_analysis.py. The script should read a prediction JSONL file and output a markdown table containing examples of substitution, deletion, insertion, long-utterance errors, and rare-word errors where possible. Use edit distance alignment at word level. Save the result to results/error_cases.md.
```

验收：

```bash
python scripts/error_analysis.py --pred results/predictions/wav2vec2_finetune_10h_test.jsonl --out results/error_cases.md
```

---

### 8.7 Codex 任务 7：实现 RTF 统计

Prompt：

```text
Implement src/measure_rtf.py to measure inference real-time factor for a trained ASR model on a manifest. It should load the model, run inference on N utterances, measure wall-clock inference time, sum audio duration, and output RTF = inference_time / audio_duration. Append the result to results/metrics.csv or save a separate JSON file.
```

验收：

```bash
python src/measure_rtf.py --model_dir exp/wav2vec2_finetune_10h --manifest data/manifests/test_clean.jsonl --num_samples 100
```

---

## 9. 论文结构规划

建议 ICASSP 风格论文结构：

```text
Abstract
1. Introduction
2. Related Work
   2.1 Self-supervised Speech Representation Learning
   2.2 Low-resource ASR
   2.3 CTC-based ASR
3. Method
   3.1 Problem Formulation
   3.2 SSL Encoder for Speech Representation
   3.3 CTC-based Recognition Model
   3.4 Training and Decoding
4. Experiments
   4.1 Dataset and Preprocessing
   4.2 Experimental Setup
   4.3 Evaluation Metrics
   4.4 Baselines and Ablations
5. Results and Analysis
   5.1 Main Results
   5.2 Frozen vs Fine-tuned Representations
   5.3 Effect of Labeled Data Scale
   5.4 Performance-Cost Trade-off
   5.5 Error Analysis
6. Conclusion
References
```

---

## 10. 报告中可写的核心观点

### 10.1 方法动机

```text
低资源 ASR 的主要困难是标注语音数据不足。自监督语音模型可以从大规模无标注语音中学习通用表征，因此适合作为低资源 ASR 的声学 encoder。
```

### 10.2 Frozen vs fine-tune

```text
Frozen SSL encoder 可以降低训练成本，但由于 encoder 不能适配目标转写任务，WER 可能较高。Fine-tuning 允许表征适配 CTC 识别目标，通常可以获得更低 WER/CER，但训练成本更高。
```

### 10.3 数据规模影响

```text
随着标注训练数据从 1h 增加到 10h，模型能看到更多说话人、音素上下文和词汇覆盖，WER/CER 预期下降。该实验直接体现 SSL 表征在低资源条件下的迁移效果。
```

### 10.4 不同 SSL 模型比较

```text
Wav2Vec2、HuBERT 和 WavLM 的预训练目标不同，因此迁移到低资源 ASR 后可能表现不同。WavLM 在建模 spoken content 的同时也考虑 speaker-related 信息，因此可作为增强对比模型。
```

### 10.5 错误分析

```text
典型错误可能集中在短功能词删除、发音相近词替换、长句识别错误和低频词识别错误。这些现象反映了 CTC 解码和低资源训练数据覆盖不足的局限。
```

---

## 11. 最小成功标准

五天结束时，最低必须交付：

```text
1. 可运行代码仓库。
2. LibriSpeech 低资源数据构造脚本。
3. wav2vec2-base frozen + CTC baseline。
4. wav2vec2-base fine-tune + CTC 主模型。
5. 1h vs 10h 低资源规模消融。
6. WER / CER 结果表。
7. 至少 5 条错误案例分析。
8. README 复现说明。
9. ICASSP 模板 PDF 报告。
```

若时间允许，额外交付：

```text
1. WavLM-base 或 HuBERT-base 对比实验。
2. RTF 统计。
3. 参数量和可训练参数量统计。
4. 训练成本表。
```

---

## 12. 风险与降级方案

### 12.1 如果 GPU 显存不足

采取：

```text
1. batch size 降为 1。
2. gradient_accumulation_steps 增加到 16 或 32。
3. 开启 fp16。
4. 开启 gradient checkpointing。
5. 限制最大音频长度，例如 15s。
6. 先只做 1h 和 10h。
7. 放弃 WavLM/HuBERT 增强实验。
```

---

### 12.2 如果 fine-tune 太慢

采取：

```text
1. 减少 epoch。
2. 使用 max_steps 代替完整 epoch。
3. 只在 dev 子集上快速验证。
4. 保留 frozen baseline 和 1h/10h 消融。
5. 不做不同 SSL 模型对比。
```

---

### 12.3 如果 Hugging Face 数据加载失败

采取：

```text
1. 使用 torchaudio.datasets.LIBRISPEECH。
2. 或手动从 OpenSLR 下载 tar.gz。
3. 生成本地 JSONL manifest。
4. 训练脚本统一读取 manifest，避免依赖数据集库特定版本。
```

---

### 12.4 如果 WavLM/HuBERT 训练失败

采取：

```text
1. 放弃 E4。
2. 把增强部分换成训练成本分析。
3. 把论文重点放在 frozen vs fine-tune 和 1h vs 10h。
```

---

## 13. 最终提交材料清单

```text
ssl_low_resource_asr.zip
├── README.md
├── requirements.txt
├── scripts/
├── src/
├── configs/
├── results/
│   ├── metrics.csv
│   ├── predictions/
│   └── error_cases.md
├── paper/
│   ├── main.tex
│   ├── refs.bib
│   └── main.pdf
└── LICENSE 或 NOTICE，如需要
```

README 至少包含：

```text
1. 项目简介。
2. 使用的数据集。
3. 使用的预训练模型。
4. 环境安装。
5. 数据准备命令。
6. 训练命令。
7. 评测命令。
8. 实验结果表。
9. 开源代码和模型引用说明。
```

---

## 14. 推荐参考链接

数据：

```text
LibriSpeech OpenSLR:
https://www.openslr.org/12

Hugging Face LibriSpeech:
https://huggingface.co/datasets/openslr/librispeech_asr

Torchaudio LIBRISPEECH:
https://docs.pytorch.org/audio/stable/generated/torchaudio.datasets.LIBRISPEECH.html
```

模型：

```text
Wav2Vec2 Base:
https://huggingface.co/facebook/wav2vec2-base

HuBERT Base:
https://huggingface.co/facebook/hubert-base-ls960

WavLM Base:
https://huggingface.co/microsoft/wavlm-base
```

代码：

```text
Hugging Face CTC ASR example:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py

Hugging Face speech recognition README:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/README.md

SpeechBrain:
https://github.com/speechbrain/speechbrain

S3PRL:
https://github.com/s3prl/s3prl

JiWER:
https://github.com/jitsi/jiwer
```

论文：

```text
wav2vec 2.0:
https://arxiv.org/abs/2006.11477

HuBERT:
https://arxiv.org/abs/2106.07447

WavLM:
https://arxiv.org/abs/2110.13900

LibriSpeech:
https://www.danielpovey.com/files/2015_icassp_librispeech.pdf
```

---

## 15. 当前最推荐执行命令顺序

```bash
# 1. 创建环境
conda create -n ssl_asr python=3.10 -y
conda activate ssl_asr
pip install torch torchaudio transformers datasets evaluate jiwer accelerate soundfile librosa pandas numpy tqdm tensorboard

# 2. 生成数据 manifest
python scripts/prepare_librispeech_subsets.py \
  --data_root ./data/raw \
  --manifest_dir ./data/manifests \
  --seed 42 \
  --target_hours 1 10

# 3. debug 训练
bash scripts/train_debug.sh

# 4. frozen baseline
bash scripts/train_wav2vec2_frozen_10h.sh

# 5. 10h fine-tune 主模型
bash scripts/train_wav2vec2_finetune_10h.sh

# 6. 1h fine-tune 消融
bash scripts/train_wav2vec2_finetune_1h.sh

# 7. 可选：WavLM 10h fine-tune
bash scripts/train_wavlm_finetune_10h.sh

# 8. 汇总结果
bash scripts/eval_all.sh
python scripts/error_analysis.py --pred results/predictions/wav2vec2_finetune_10h_test.jsonl --out results/error_cases.md
```

---

## 16. 最重要的执行原则

```text
第一优先级：先跑通完整 pipeline。
第二优先级：保证 E1/E2/E3 有结果。
第三优先级：写清楚方法和实验设置。
第四优先级：补 WavLM/HuBERT 增强实验。
第五优先级：美化代码和报告。
```

不要为了追求复杂创新导致没有完整结果。课程大作业的关键是：

```text
明确任务 + 使用 SSL 表征 + 有 baseline + 有消融 + 有量化指标 + 可复现 + 报告分析清楚
```
