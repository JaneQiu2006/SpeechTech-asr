# Low-resource ASR with self-supervised speech representations

本项目在 LibriSpeech 上研究低资源条件下的自监督语音表征，包括 Wav2Vec2/WavLM
全量与部分微调、冻结层 probing、下游 CTC head、连续/离散表示及数据规模消融。
训练使用字符级 CTC；dev-clean 用于模型选择，test-clean 仅用于最终评测。

## 1. 环境

完整实验以 Linux、Python 3.10、单张 24 GB RTX 3090 为基准：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

检查代码与环境：

```bash
python -m pytest -q
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

训练脚本首次运行会从 Hugging Face 下载
`facebook/wav2vec2-base`、`microsoft/wavlm-base` 等预训练权重。

## 2. 数据准备

将解压后的 LibriSpeech 放在：

```text
data/
├── train-clean-100/
├── dev-clean/
└── test-clean/
```

生成固定随机种子（42）的 manifest、30 类字符词表以及过滤后累计时长的 1h/3h
子集：

```bash
python scripts/prepare_librispeech_subsets.py --local_only
python scripts/prepare_librispeech_subsets.py --local_only \
  --splits train-clean-100 --target_hours 1 3 \
  --max_train_duration_in_seconds 15 --subset_suffix effective_15s
```

若本地数据不完整，去掉 `--local_only`，脚本将自动下载。关键输出为：

```text
data/manifests/train_1h_effective_15s.jsonl   # 约 1.001 h
data/manifests/train_3h_effective_15s.jsonl   # 约 3.001 h
data/manifests/train_10h.jsonl                # 训练时过滤后约 6.392 h
data/manifests/dev_clean.jsonl
data/manifests/test_clean.jsonl
data/vocab/vocab.json
```

所有正式训练限制音频长度为 15 秒，因此结果中报告有效时长，而非 manifest
名称中的名义时长。若音频不在项目的 `data/` 下，启动时设置
`DATA_ROOT=/absolute/path/to/data`。

## 3. 复现实验

以下命令均从项目根目录执行。脚本会做 CUDA/cuDNN 预检、写入独立日志，并拒绝
覆盖已有正式产物；中断后应先检查对应的 `exp/`、`logs/` 和 prediction 文件。

### 3.1 主实验 E1–E9

```bash
# E2、修复后的 E3、E4
bash scripts/train_e2_e4_rtx3090.sh e2

# 公平冻结基线、E5、部分微调、masking 和在线 frozen BiLSTM
bash scripts/train_new_experiments_rtx3090.sh \
  e1-30 e5 e6a e6b e8 e9
```

E7（HuBERT）按最终实验方案跳过。E9 是保留的数值塌缩实验，不应作为 frozen
encoder 的性能上限。

### 3.2 表征与离散单元 E10–E12

```bash
bash scripts/train_representation_extension_rtx3090.sh e10 e11 e12a
bash scripts/test_e5_e12_linux.sh
```

该阶段提取 Wav2Vec2 隐层特征，训练冻结表征的 BiLSTM-CTC head，并仅用训练帧
拟合 K-means。缓存位于 `features/`，codebook 位于 `artifacts/kmeans/`。

评测 E1–E4：

```bash
python scripts/evaluate_e1_e5.py \
  --experiments E1 E2 E3 E4 \
  --manifest data/manifests/test_clean.jsonl \
  --metrics_path results/test_metrics.csv \
  --predictions_dir results/predictions
```

### 3.3 深挖实验 E13–E23

```bash
bash scripts/train_deep_dive_rtx3090.sh all e22 e23
```

该队列完成完整 layer-wise probing、层融合、head 容量、top-k 部分微调、更大
codebook、离散 embedding、数据规模交互、错误分析、masking 小网格和第二随机
种子。每个训练实验保存 dev/test prediction、`summary.json` 和
`completion.json`。

### 3.4 严格对照、第二种子与统计分析 E24–E25

```bash
bash scripts/run_all_followup_experiments_rtx3090.sh
```

该脚本完成 30 类词表下的连续/centroid/embedding 对照、top-5 第二种子，并
运行 10,000 次配对 bootstrap、错误分析、master table 构建和作图。若训练已
完成而只需重建分析：

```bash
bash scripts/run_all_followup_experiments_rtx3090.sh --from analysis
```

## 4. 输出与结果核验

主要产物：

```text
exp/                                  模型、配置、checkpoint
features/                             FP16 隐层特征缓存
artifacts/kmeans/                     K-means codebook 与统计
results/predictions/                  逐句预测
results/master_metrics.csv            统一复算后的主结果表
results/followup_paired_bootstrap.csv 配对 bootstrap 结果
results/deep_dive_error_analysis.md   错误分析
results/figures/                      论文图
logs/                                 训练与评测日志
```

可单独重建汇总与图：

```bash
python scripts/build_master_results.py --output results/master_metrics.csv
python scripts/plot_deep_dive_results.py
```

在相同数据、seed 和配置下，`test-clean` 的关键核验值如下（四舍五入）：

| 设置 | Test WER |
|---|---:|
| Wav2Vec2 full fine-tuning，6.392h（E2） | 11.44% |
| Wav2Vec2 + masking，6.392h（E8） | 11.35% |
| top-5 partial fine-tuning（E16B） | 12.62% |
| frozen layer 8 + 2-layer BiLSTM（E13） | 19.51% |
| layer 8 centroid units，K=1000（E24D） | 33.08% |
| 1h full fine-tuning / frozen layer 8 | 40.51% / 31.63% |

E8 与 E2 的差异无统计显著性；小幅数值差异不应视为稳定提升。
