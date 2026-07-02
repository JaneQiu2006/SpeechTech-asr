# SSL 表征层与离散语音单元扩展实验计划

> 用途：在 E1–E5 已完成、E6a/E6b 预计可在当前服务器回收前完成的基础上，
> 规划额外租用一天 RTX 3090 Ti 时最值得执行的实验。本文是
> `doc/ssl_asr_remaining_gpu_training_plan.md` 的扩展，不替代其中已经确定的
> E6a、E6b、E8 和 E9 配置。

---

## 0. 决策背景

### 0.1 当前有效结果

E1–E5 已完成 dev 和完整 test-clean 评估。正式 test-clean 共 2620 条、
5.403 小时。

| ID | 系统 | 有效训练时长 | Test WER | Test CER | RTF |
|---|---|---:|---:|---:|---:|
| E1 | Frozen Wav2Vec2 + linear CTC | 6.392 h | 99.92% | 85.46% | 0.00389 |
| E2 | Full fine-tuned Wav2Vec2 + CTC | 6.392 h | **11.44%** | **3.36%** | 0.00403 |
| E3 | Full fine-tuned Wav2Vec2 + CTC | 1.001 h | 40.51% | 11.65% | 0.00397 |
| E4 | Full fine-tuned WavLM + CTC | 6.392 h | 17.50% | 5.08% | 0.00886 |
| E5 | Full fine-tuned Wav2Vec2 + CTC | 3.001 h | 17.12% | 4.88% | 0.00384 |

当前已经能够回答：

```text
1. 增加标注数据能够显著改善识别性能，但收益逐渐减小。
2. 当前 frozen Wav2Vec2 + linear CTC 配置无法得到有效识别结果。
3. 在相同数据与训练框架下，Wav2Vec2 优于 WavLM。
4. WavLM 在本地 RTX 4060 上的 RTF 约为 Wav2Vec2 的 2.2 倍。
```

当前仍不能回答：

```text
1. Wav2Vec2 的哪个隐藏层最适合低资源 ASR？
2. E1 失败是表征本身不足，还是 linear CTC head 太弱？
3. 连续表示量化为离散单元后，WER、模型复杂度和信息率如何变化？
4. codebook size 在识别精度与压缩率之间产生什么权衡？
5. masking 是否能改善 1h 极低资源条件下的泛化？
```

### 0.2 E6–E9 的状态和取舍

| ID | 实验 | 决策 |
|---|---|---|
| E6a | Wav2Vec2 top-6-layer fine-tuning | 当前服务器回收前完成 |
| E6b | Wav2Vec2 top-3-layer fine-tuning | 当前服务器回收前完成 |
| E7 | HuBERT-base full fine-tuning | 跳过 |
| E8 | Wav2Vec2 feature masking regularization | 额外一天执行 |
| E9 | Frozen Wav2Vec2 + BiLSTM-CTC | 额外一天执行 |

跳过 E7 是合理的资源分配：E2 与 E4 已提供两种 SSL encoder 的直接比较，
第三个 encoder 的边际信息小于“隐藏层”和“连续/离散表示”的对照。但最终
报告只能表述为 Wav2Vec2 与 WavLM 的比较，不能将结论泛化到所有 SSL 模型。

原剩余训练计划提出“不从零搭建离散单元 ASR 主线”，其前提是只剩约
9–10 小时 GPU 时间。额外增加完整一天后，该约束不再成立，但仍应采用
最小、可控的离散量化方案，避免扩展成新的大模型工程。

---

## 1. 新增实验总表

### 1.1 核心实验

| 优先级 | ID | 实验 | 主要研究问题 | 预计时间 |
|---|---|---|---|---:|
| P0 | E10a | Frozen Wav2Vec2 layer 6 + BiLSTM-CTC | 中层表示是否适合 ASR | 1–2 h |
| P0 | E10b | Frozen Wav2Vec2 layer 9 + BiLSTM-CTC | 高层表示是否优于中层 | 1–2 h |
| P0 | E10c | Frozen Wav2Vec2 layer 12 + BiLSTM-CTC | 最终层连续表示对照 | 1–2 h |
| P0 | E11a | 最佳层离散量化，K=50 | 高压缩、小 codebook | 1–2 h |
| P0 | E11b | 最佳层离散量化，K=100 | 中等压缩 | 1–2 h |
| P0 | E11c | 最佳层离散量化，K=200 | 大 codebook、低量化损失 | 1–2 h |
| P1 | E12a | E3 1h + time masking | 正则化能否改善极低资源泛化 | 1–2 h |

E10/E11 的时间估算建立在“冻结 encoder 特征只提取一次并缓存”的前提上。
如果每个 downstream 实验都重复执行 Wav2Vec2 前向，预计无法在一天内完成。

### 1.2 可选实验

| 优先级 | ID | 实验 | 启动条件 |
|---|---|---|---|
| P1 | E10d | 所有隐藏层的 learned scalar mixture + BiLSTM-CTC | E10a–c 已完成且剩余 ≥3 h |
| P2 | E12b | E3 1h + time/feature masking | E12a 有效且 E8 未发生 collapse |
| P2 | E13 | 最佳新增配置更换随机种子复跑 | 剩余 ≥2 h |

---

## 2. E10：不同连续表征层

### 2.1 研究问题

```text
冻结 Wav2Vec2 后，不同 Transformer 层包含的语音识别信息是否存在明显差异？
更强的 BiLSTM head 能否利用 E1 的 linear head 无法利用的连续表征？
```

E9 已经证明“冻结 encoder + 更强 head”可以被实现，但仅观察最终层不足以
判断冻结表示整体是否无效。不同层编码的声学、音系和语义信息并不相同，
因此必须在完全相同的 downstream head 下比较多个层。

### 2.2 实验定义

| 配置项 | E10a | E10b | E10c |
|---|---:|---:|---:|
| SSL encoder | Wav2Vec2-base | Wav2Vec2-base | Wav2Vec2-base |
| encoder 状态 | frozen/eval | frozen/eval | frozen/eval |
| 输出层 | Transformer 6 | Transformer 9 | Transformer 12 |
| downstream | 2-layer BiLSTM + linear CTC | 相同 | 相同 |
| hidden size | 256 | 256 | 256 |
| CTC vocabulary | 项目现有 30 字符 | 相同 | 相同 |
| 训练数据 | 6.392h effective train set | 相同 | 相同 |
| dev 数据 | 现有 ≤15s dev 子集 | 相同 | 相同 |

层编号必须在实现中固定语义：

```text
hidden_states[0]  ：CNN projection 后、Transformer 前
hidden_states[1]  ：Transformer block 1 输出
...
hidden_states[12] ：Transformer block 12 输出
```

文档、配置和日志统一使用 Transformer block 的 1-based 编号。

### 2.3 公平性要求

1. 三组实验使用同一份缓存、同一个 tokenizer 和完全相同的 BiLSTM head。
2. encoder 设为 `eval()`，避免冻结参数后 dropout 仍随训练随机变化。
3. 不在 dev/test 特征上拟合任何归一化参数。
4. 模型选择只使用 dev WER，test-clean 仅在配置确定后评估一次。
5. E9 可作为工程参考，但 E10c 应在统一缓存管线中重新训练，以保证与
   E10a/E10b 的输入处理完全一致。

### 2.4 可选 learned layer mixture

如果剩余时间充足，增加 E10d：

\[
h_t=\sum_{l=0}^{12}\operatorname{softmax}(\alpha)_l h_{t,l}
\]

只训练 13 个 scalar 权重和相同的 BiLSTM-CTC head。该实验可以回答：

```text
手动选择单层是否足够，还是组合多个层能够获得更好的识别表示？
```

---

## 3. E11：连续表示与离散单元

### 3.1 研究问题

```text
将最佳连续隐藏表示量化为有限 codebook 后，识别性能下降多少？
更大的 codebook 是否能用更高 bitrate 换取更低 WER？
离散表示相对 FP16 连续表示能压缩多少？
```

### 3.2 离散单元构造

1. 从 E10 的 dev 最佳层提取训练集帧级隐藏表示。
2. 仅从训练集随机抽取最多 500,000 帧拟合 MiniBatch K-means。
3. 固定随机种子 42。
4. 分别训练 K=50、100、200 三个 codebook。
5. 对每帧选择最近的聚类中心：

\[
z_t=\arg\min_k\|h_t-c_k\|_2^2
\]

6. downstream 输入使用对应的 768 维聚类中心 \(c_{z_t}\)，而不是额外学习
   一个 embedding。

这种设计使连续和离散系统使用相同维度、相同 BiLSTM 和相同 CTC head。
两者的主要差别仅为是否经过 K-means 量化，避免把 embedding 参数量变化
误解释为离散表示本身的影响。

### 3.3 不做相邻重复折叠

训练 CTC 时保留原始约 50 Hz 的帧级离散序列，不执行相邻重复单元折叠。
原因是折叠可能使输入序列短于字符标签序列，导致 CTC 对齐不可行。

相邻重复折叠只用于额外统计：

```text
raw token rate
deduplicated token rate
raw bitrate
deduplicated bitrate
```

### 3.4 Codebook 与信息压缩指标

每个 K 必须报告：

| 指标 | 定义 |
|---|---|
| nominal codebook size | K |
| used units | test 中实际出现的聚类数 |
| utilization | used units / K |
| perplexity | \(2^{H(U)}\) |
| token rate | 单元总数 / 音频秒数 |
| entropy | \(-\sum_u p(u)\log_2p(u)\) |
| empirical bitrate | token rate × entropy |
| fixed-width bitrate | token rate × \(\lceil\log_2K\rceil\) |
| WER/CER | 相同 CTC 解码规则 |
| end-to-end RTF | encoder + quantization + downstream |

连续 FP16 表示的理论信息率：

\[
R_{\text{continuous}}
=f\times d\times16
\]

其中 Wav2Vec2 帧率约为 50 Hz，维度 \(d=768\)，因此：

\[
R_{\text{continuous}}\approx50\times768\times16
=614400\text{ bit/s}
\]

离散表示固定宽度上限示例：

| K | 每单元位数 | 50 Hz 理论 bitrate |
|---:|---:|---:|
| 50 | 6 bit | 300 bit/s |
| 100 | 7 bit | 350 bit/s |
| 200 | 8 bit | 400 bit/s |

实际报告必须使用从音频长度和单元序列测得的真实 token rate，不能直接把
50 Hz 近似值写成最终结果。

### 3.5 结果解释

| 观察 | 可支持的结论 |
|---|---|
| K=50 WER 明显恶化 | 高压缩丢失了识别所需的音系细节 |
| K 增大时 WER 持续下降 | codebook 容量与识别精度存在清晰权衡 |
| K=100/200 接近连续表示 | SSL 表征存在较大离散化冗余 |
| 三个 K 均失败 | 当前 K-means 单元或 downstream 不足，不能证明所有离散表示无效 |
| codebook 利用率很低 | K 过大或聚类分布不均，应同时解释有效 codebook size |

负结果仍然有价值，但结论必须限定为：

```text
Wav2Vec2 指定层 + 当前 K-means + 当前 BiLSTM-CTC 设置
```

不能泛化为“离散语音单元不适合 ASR”。

---

## 4. E12：1h 条件下的正则化

### 4.1 背景

E3 repaired 使用：

```text
mask_time_prob = 0.0
mask_feature_prob = 0.0
```

其 test WER 为 40.51%，明显弱于 E5/E2，因此 1h 条件更适合观察正则化是否
改善低资源泛化。

E2 默认使用 Wav2Vec2 time masking；E8 在此基础上增加 feature masking。
因此 E8 主要回答 6.392h 条件下 feature masking 的增益，而 E12a 应先回答
1h 条件下启用标准 time masking 是否有益。

### 4.2 E12a 配置

除 masking 外与 E3 repaired 完全一致：

```yaml
experiment: wav2vec2_finetune_1h_time_mask
train_manifest: data/manifests/train_1h_effective_15s.jsonl
max_steps: 1500
learning_rate: 1.0e-4
encoder_learning_rate: 2.0e-5
encoder_freeze_steps: 100
warmup_ratio: 0.1
mask_time_prob: 0.05
mask_feature_prob: 0.0
```

### 4.3 动态停止条件

```text
1. 连续 8 次评估 non-empty ratio < 0.01：停止。
2. step 800 仍接近全 blank：停止。
3. dev WER 明显差于 E3 且无继续下降趋势：停止。
4. E12a 有效且时间充足时，才考虑 time+feature masking 的 E12b。
```

---

## 5. 特征缓存与实现要求

### 5.1 缓存内容

建议新增独立脚本，不修改 E1–E9 的正式产物：

```text
scripts/extract_ssl_features.py
scripts/train_kmeans_units.py
scripts/train_cached_ctc_head.py
scripts/evaluate_discrete_ctc.py
```

缓存目录建议：

```text
features/wav2vec2_base/
├── train_layer6/
├── train_layer9/
├── train_layer12/
├── dev_layer6/
├── dev_layer9/
└── dev_layer12/

artifacts/kmeans/
├── wav2vec2_layer<best>_k50.joblib
├── wav2vec2_layer<best>_k100.joblib
└── wav2vec2_layer<best>_k200.joblib
```

每条缓存记录至少包含：

```text
utterance ID
真实帧数
隐藏表示或离散 index
reference
audio duration
source checkpoint
layer index
dtype
```

缓存隐藏表示可使用 FP16 降低磁盘占用，训练 downstream 时转回 FP32。

### 5.2 数据隔离

```text
训练集：允许拟合 K-means、归一化统计和 downstream 参数。
dev：只用于选择层、K 和 checkpoint。
test-clean：只在最终选择完成后评估，禁止用于选择 K。
```

K-means 严禁混入 dev/test 帧。否则 codebook 已间接观察测试分布，削弱实验
严谨性。

### 5.3 RTF 测量

需要区分：

```text
cached-head RTF：只测 downstream，主要用于工程调试。
end-to-end RTF：encoder + hidden layer extraction + K-means assignment + head。
```

论文只把 end-to-end RTF 与 E1–E9 比较。缓存训练速度不能作为部署推理速度。

### 5.4 与现有 CTC 指标的关系

`results/test_metrics.csv` 中已有：

```text
ctc_vocab_size
ctc_token_rate
ctc_token_entropy_bits
ctc_label_bitrate_bps
```

这些是解码后字符标签指标，不是离散声学表示指标。E11 必须使用独立字段：

```text
unit_codebook_size
unit_used_types
unit_token_rate
unit_entropy_bits
unit_bitrate_bps
unit_dedup_token_rate
unit_dedup_bitrate_bps
```

两类指标不能合并或互相替代。

---

## 6. 额外一天的执行队列

### 6.1 推荐顺序

```text
阶段 A：已有配置
E8 → E9

阶段 B：连续表征层
一次性提取 layer 6/9/12
→ E10a → E10b → E10c
→ 按 dev WER 选最佳层

阶段 C：离散单元
在最佳层训练 K=50/100/200 codebook
→ E11a → E11b → E11c

阶段 D：低资源泛化
E12a

阶段 E：仍有余量
E10d 或最佳新增配置的第二随机种子
```

### 6.2 时间预算

| 阶段 | 预计时间 |
|---|---:|
| E8 | 2–3 h |
| E9 | 3–4 h |
| 特征提取与 K-means | 1–2 h |
| E10a–c | 3–5 h |
| E11a–c | 3–5 h |
| E12a | 1–2 h |
| 同步、异常和重启缓冲 | 3 h |
| 合计 | 16–24 h |

如果实际 E9 比预计更慢，优先保证：

```text
E8、E9、E10a–c、E11b（K=100）
```

随后按剩余时间补 K=50 和 K=200。K=100 是最优先的离散实验，因为它处于
压缩率和容量的中间点，也便于与已有离散语音单元工作比较。

### 6.3 停止与降级策略

| 情况 | 动作 |
|---|---|
| E9 FP32 超时 | 保留最佳 checkpoint，不延长训练；优先启动缓存实验 |
| 特征缓存超过预计磁盘 | 只缓存 layer 6/9/12 的 FP16，按 split 分块 |
| 三层快速验证均全 blank | 先检查 CTC 长度、padding 和 head，再继续；禁止直接跑完整队列 |
| K-means 内存不足 | 使用 MiniBatch K-means，限制采样为 500k 帧 |
| K=100 明显失败 | 先诊断 codebook utilization 和 CTC 输入长度，再决定是否运行 K=50/200 |
| 剩余时间少于 2 h | 放弃 E12/E13，优先同步模型、codebook、日志和配置 |

---

## 7. 服务器租用前的验收清单

不要在核心代码尚未通过本地小样本验证时开始计费。租用前必须完成：

```text
[ ] hidden_states 层编号测试。
[ ] 20 条音频的 layer 6/9/12 特征提取。
[ ] padding 帧不会进入 K-means。
[ ] 缓存特征与在线 encoder 输出数值一致。
[ ] K=10 的最小 K-means smoke test。
[ ] 连续特征和 centroid 特征均能执行一次 CTC forward/backward。
[ ] 保存后可重新加载 BiLSTM head 和 codebook。
[ ] 离散序列长度满足 CTC 对齐要求。
[ ] train/dev/test 隔离断言。
[ ] 所有正式输出目录拒绝覆盖。
```

服务器回收前必须同步：

```text
exp/<experiment>/
results/predictions/*_dev.jsonl
logs/*.log
artifacts/kmeans/*
最终 YAML/命令行配置
trainer_state.json
特征归一化参数
codebook utilization 统计
```

训练特征缓存体积较大，可以不全部下载，但 codebook、配置、日志和模型必须
下载。若本地需要重建 test 离散单元，应确保 codebook 和特征处理参数完整。

---

## 8. 最终结果组织

### 8.1 表征层对比

| ID | SSL layer | Downstream | Trainable Params | Dev WER | Test WER | RTF |
|---|---:|---|---:|---:|---:|---:|
| E10a | 6 | BiLSTM-CTC | TBD | TBD | TBD | TBD |
| E10b | 9 | BiLSTM-CTC | TBD | TBD | TBD | TBD |
| E10c | 12 | BiLSTM-CTC | TBD | TBD | TBD | TBD |
| E10d | learned mixture | BiLSTM-CTC | TBD | TBD | TBD | TBD |

### 8.2 连续与离散表示

| Representation | Layer | K | WER | CER | Bitrate | RTF |
|---|---:|---:|---:|---:|---:|---:|
| Continuous FP16 | best | N/A | TBD | TBD | TBD | TBD |
| Discrete K-means | best | 50 | TBD | TBD | TBD | TBD |
| Discrete K-means | best | 100 | TBD | TBD | TBD | TBD |
| Discrete K-means | best | 200 | TBD | TBD | TBD | TBD |

建议绘制：

```text
1. WER vs codebook size
2. WER vs empirical bitrate
3. codebook utilization vs K
4. 不同 layer 的 WER
5. trainable parameters vs WER（E1/E6/E2）
```

### 8.3 论文主线

新增实验完成后，结果章节可以组织为：

```text
5.1 Effect of Labeled Data Scale
    E3, E5, E2

5.2 Frozen, Partial, and Full Fine-tuning
    E1, E9, E6b, E6a, E2

5.3 Layer-wise Analysis of SSL Representations
    E10a, E10b, E10c

5.4 Continuous versus Discrete Speech Representations
    best E10, E11a, E11b, E11c

5.5 Regularization under Limited Labeled Data
    E3, E8, E12

5.6 Accuracy, Complexity, and Information Compression
    WER, parameters, RTF, token rate, bitrate
```

---

## 9. 最终建议

额外一天最有价值的目标不是继续堆叠 SSL encoder，而是形成以下闭环：

```text
数据规模：
1h → 3h → 6.392h

参数更新范围：
frozen → top 3 → top 6 → full fine-tune

连续表征层：
layer 6 → layer 9 → layer 12

离散表示容量：
K=50 → K=100 → K=200

系统权衡：
WER/CER ↔ trainable parameters ↔ RTF ↔ bitrate
```

最低成功标准：

```text
完成 E8、E9、E10a–c 和至少一个 K=100 离散实验。
```

理想完成标准：

```text
完成 E8、E9、E10a–c、E11a–c 和 E12a，
并对最佳新增配置至少补一个不同随机种子。
```

---

## 10. 参考文献

1. Baevski, A., Zhou, H., Mohamed, A., & Auli, M.  
   *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
   Representations.*  
   https://arxiv.org/abs/2006.11477

2. Hsu, W.-N., et al.  
   *HuBERT: Self-Supervised Speech Representation Learning by Masked
   Prediction of Hidden Units.*  
   https://arxiv.org/abs/2106.07447

3. Yang, S.-w., et al.  
   *SUPERB: Speech processing Universal PERformance Benchmark.*  
   https://arxiv.org/abs/2105.01051

4. Lakhotia, K., et al.  
   *Generative Spoken Language Modeling from Raw Audio.*  
   https://arxiv.org/abs/2102.01192

5. Pasad, A., Chien, C.-M., Settle, S., & Livescu, K.  
   *What Do Self-Supervised Speech Models Know About Words?*  
   https://aclanthology.org/2024.tacl-1.21/

---

## 11. 已实现代码与执行命令

### 11.1 实现状态

核心 E10/E11/E12a 流水线已经实现：

| 文件 | 用途 |
|---|---|
| `src/ssl_asr/representations.py` | 连续/离散共用的缓存数据集、BiLSTM-CTC 与 unit 指标 |
| `scripts/extract_ssl_features.py` | 一次提取并缓存 layer 6/9/12 |
| `scripts/train_cached_ctc_head.py` | 训练连续或 centroid-quantized BiLSTM-CTC |
| `scripts/train_kmeans_units.py` | 仅用训练帧拟合 K-means 并生成 unit cache |
| `scripts/select_best_representation_layer.py` | 根据 dev WER 选择最佳连续层 |
| `scripts/evaluate_representation_ctc.py` | 连续/离散系统完整 test-clean 与端到端 RTF |
| `scripts/train_representation_extension_rtx3090.sh` | 额外一天服务器队列 |
| `scripts/test_representation_extension.ps1` | 本地 RTX 4060 测试入口 |

E10d learned layer mixture、E12b 和 E13 仍属于时间充足时的可选实验，当前
队列不会自动执行。

### 11.2 服务器完整队列

服务器已按项目 RTX 3090 环境部署时，安装更新后的服务器依赖：

```bash
pip install -r requirements-rtx3090.txt
```

不要在服务器上改用通用 `requirements.txt` 覆盖已经固定的 CUDA PyTorch。

默认执行 E8、E9、E10、E11 和 E12a：

```bash
bash scripts/train_representation_extension_rtx3090.sh
```

如果 E8/E9 已经完成，只运行新增表征实验：

```bash
bash scripts/train_representation_extension_rtx3090.sh e10 e11 e12a
```

也可以分阶段运行：

```bash
bash scripts/train_representation_extension_rtx3090.sh e10
bash scripts/train_representation_extension_rtx3090.sh e11
bash scripts/train_representation_extension_rtx3090.sh e12a
```

E11 会读取：

```text
artifacts/kmeans/best_layer.txt
```

如果该文件不存在，会从三个已完成的 E10 `summary.json` 中重新选择；E10
未全部完成时，E11 会拒绝启动。

### 11.3 本地 test-clean

同步正式模型、codebook 和配置后，在项目根目录运行：

```powershell
.\scripts\test_representation_extension.ps1
```

现有 `scripts/sync_remote_artifacts.ps1` 已支持选择性同步 E10/E11 的
`model.pt`、`completion.json`、`summary.json`、`centers.npy`、
`kmeans.joblib` 和 `best_layer.txt`，不会下载体积较大的训练特征缓存与
离散 unit cache。

显存不足时：

```powershell
.\scripts\test_representation_extension.ps1 -BatchSize 1
```

该脚本会：

```text
1. 测试已存在的 E6a、E6b、E8、E9、E12a。
2. 测试已完成的 E10 layer 6/9/12。
3. 根据 best_layer.txt 测试 K=50/100/200 的 E11。
4. 已有完整 test 指标时跳过。
```

输出：

```text
results/test_metrics.csv
results/representation_test_metrics.csv
results/predictions/*_test.jsonl
```

### 11.4 Debug 产物管理

正式队列不运行 startup validation，也不创建 debug experiment。所有原子写入
临时文件使用 `.tmp` 后缀；队列退出时只清理由本扩展流水线在
`features/wav2vec2_base` 和 `artifacts/kmeans` 下产生的残留 `.tmp` 文件。

以下文件属于正式可复现实验产物，不应作为 debug 删除：

```text
features/wav2vec2_base/*
artifacts/kmeans/*/centers.npy
artifacts/kmeans/*/kmeans.joblib
artifacts/kmeans/*/summary.json
exp/wav2vec2_layer*_bilstm_ctc/*
exp/wav2vec2_discrete_k*_bilstm_ctc/*
logs/*_rtx3090.log
```
