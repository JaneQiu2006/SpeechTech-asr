# SSL 低资源 ASR 深挖实验方案（Codex 执行版）

> 用途：在已有 E1–E12 实验结果基础上，继续围绕“自监督语音表征如何被有效用于低资源 ASR”设计后续深挖实验。本文档面向 Codex，重点给出可实现的实验编号、代码改动、脚本入口、验收标准、停止规则和结果组织方式。
>
> 当前项目主线：基于 `facebook/wav2vec2-base` / `microsoft/wavlm-base` 的低资源英文 ASR，字符级 CTC 解码，数据来自 LibriSpeech 低资源子集。已有完整 test-clean 评估与 E1–E12 结果。

---

## 0. 当前已知结论与深挖目标

### 0.1 已有强信号

当前 E1–E12 已经支持以下事实：

1. **数据规模有效，但收益递减。**
   - E3：1.001 h full fine-tuned Wav2Vec2，test WER 40.51%。
   - E5：3.001 h full fine-tuned Wav2Vec2，test WER 17.12%。
   - E2：6.392 h full fine-tuned Wav2Vec2，test WER 11.44%。

2. **top-6 partial fine-tuning 接近 full fine-tuning。**
   - E2 full fine-tune：90.194M trainable params，test WER 11.44%。
   - E6a top-6 fine-tune：47.667M trainable params，test WER 12.14%。
   - E6b top-3 fine-tune：26.403M trainable params，test WER 22.41%。

3. **frozen 表征不是无效，而是强依赖层选择和 downstream head。**
   - E1 frozen Wav2Vec2 + linear CTC：test WER 99.92%。
   - E10 layer 9 + BiLSTM-CTC：test WER 20.51%。
   - E10 layer 6：30.03%；layer 12：46.18%。

4. **离散单元形成清晰 accuracy–bitrate trade-off。**
   - E11 K=50：WER 71.56%，bitrate 274.93 bit/s。
   - E11 K=100：WER 61.65%，bitrate 324.43 bit/s。
   - E11 K=200：WER 52.60%，bitrate 374.31 bit/s。
   - codebook 使用充分，性能下降不是 codebook collapse。

5. **masking regularization 当前不是主线。**
   - E8 与 E2 差距很小，paired bootstrap CI 包含 0。
   - E12a 与 E3 差距很小，paired bootstrap CI 包含 0。

### 0.2 后续实验的核心问题

后续不再优先“堆更多 SSL encoder”，而是围绕以下四个问题深挖：

```text
Q1. Wav2Vec2 的哪个 hidden layer 最适合 frozen ASR？
Q2. frozen 表征需要多强的 downstream head 才能被有效利用？
Q3. top-k partial fine-tuning 的性能拐点在哪里？
Q4. 离散化损失来自 codebook size，还是 downstream 表示方式？
```

---

## 1. 总体实验队列

### 1.1 实验优先级表

| 优先级 | ID | 实验 | 是否训练 | 主要用途 |
|---:|---|---|---|---|
| P0 | E13 | 完整 Wav2Vec2 layer-wise probing | 是 | 补齐 layer 1–12 的 frozen 表征曲线 |
| P0 | E15 | frozen best layer 的 downstream head 容量消融 | 是 | 解释 E1 失败与 E10 成功 |
| P1 | E14 | learned scalar mixture 层融合 | 是 | 判断手工选 layer 是否足够 |
| P1 | E16 | top-k partial fine-tuning 网格 | 是 | 找到 top-3 与 top-6 之间的拐点 |
| P1 | E17 | 更大 K 的离散单元 | 是 | 扩展 WER–bitrate 曲线 |
| P1 | E21 | 错误类型系统分析 | 否 | 提升论文分析深度 |
| P2 | E18 | centroid 输入 vs learned unit embedding | 是 | 深挖离散单元建模方式 |
| P2 | E20 | 数据规模 × frozen layer 9 + BiLSTM | 是 | 连接数据规模与 frozen 表征主线 |
| P3 | E22 | masking 小网格 | 是 | 仅用于训练稳定性讨论，不优先 |
| P3 | E23 | 关键配置第二随机种子 | 是 | 验证稳定性，时间充足再做 |

### 1.2 推荐执行组合

#### 只剩 6–8 小时 GPU

```text
1. E13：补 layer 7/8/10/11，先不用全层。
2. E15：Linear / MLP / 1-layer BiLSTM / 2-layer BiLSTM head 对比。
3. E16b：top-5 partial fine-tuning。
4. E21：本地错误分析。
```

最低成功标准：

```text
E13 局部层扫描 + E15 head 容量消融 + E21 错误分析。
```

#### 有 12–16 小时 GPU

```text
1. E13：完整 layer-wise probing。
2. E14：learned scalar mixture。
3. E15：head 容量消融。
4. E16a/E16b：top-4/top-5 partial fine-tuning。
5. E17a：K=500 离散单元。
6. E21：错误类型分析。
```

#### 额外有完整一天 GPU

```text
1. E13：完整 layer-wise probing。
2. E14：learned scalar mixture。
3. E15a–E15e：完整 head 容量消融。
4. E16a/E16b/E16c：top-4/top-5/top-8 partial fine-tuning。
5. E17a/E17b：K=500/K=1000。
6. E18b：learned unit embedding。
7. E20a/E20b：1h/3h frozen best layer + BiLSTM。
8. E21：错误类型分析。
9. E23：对 E10 best 或 E16 top-5 追加一个 seed。
```

---

## 2. 统一实验约束

### 2.1 数据与评估约束

所有新增实验必须遵守：

```text
1. 训练数据仍报告 effective hours，而不是名义 1h/3h/10h。
2. dev 只用于选择 checkpoint、hidden layer、K 或 head 类型。
3. test-clean 只在配置确定后最终评估一次。
4. 不使用 test-clean 选择 layer、K、head 或 training steps。
5. prediction 必须保存为 JSONL，字段至少包含：id, reference, prediction, duration, audio_path。
6. 所有实验必须记录 WER, CER, substitution, deletion, insertion, exact match。
7. 所有训练实验必须记录 total params, trainable params, trainable ratio, RTF。
```

### 2.2 命名规范

新增实验输出目录统一使用：

```text
exp/deep_dive/<experiment_id>_<short_name>/
logs/deep_dive/<experiment_id>_<short_name>.log
results/predictions/deep_dive/<experiment_id>_<short_name>_dev.jsonl
results/predictions/deep_dive/<experiment_id>_<short_name>_test.jsonl
results/deep_dive_metrics.csv
results/deep_dive_error_analysis.md
```

示例：

```text
exp/deep_dive/e13_layer8_bilstm_ctc/
exp/deep_dive/e15_mlp_layer9_ctc/
exp/deep_dive/e16_top5_finetune/
exp/deep_dive/e17_k500_centroid_bilstm_ctc/
```

### 2.3 原子写入与断点恢复

Codex 实现时必须保证：

```text
1. 所有 summary.json 先写入 .tmp，再 rename。
2. 已完成实验检测 completion.json，默认跳过。
3. 若命令传入 --overwrite 才允许覆盖。
4. 中断后重新运行队列，不应重复训练已完成实验。
5. 日志中必须打印 git commit hash 或当前代码修改摘要。
```

---

## 3. E13：完整 Wav2Vec2 layer-wise probing

### 3.1 研究问题

```text
Wav2Vec2 哪一层最适合 frozen ASR？
E10 中 layer 9 最优是否只是 6/9/12 三点采样造成的偶然现象？
```

### 3.2 实验矩阵

已有：

| ID | Layer | Downstream | Test WER |
|---|---:|---|---:|
| E10a | 6 | 2-layer BiLSTM-CTC | 30.03% |
| E10b | 9 | 2-layer BiLSTM-CTC | 20.51% |
| E10c | 12 | 2-layer BiLSTM-CTC | 46.18% |

新增：

| ID | Layer | Downstream | 说明 |
|---|---:|---|---|
| E13a | 1 | 2-layer BiLSTM-CTC | 低层声学特征 |
| E13b | 2 | 2-layer BiLSTM-CTC | 低层声学特征 |
| E13c | 3 | 2-layer BiLSTM-CTC | 低中层 |
| E13d | 4 | 2-layer BiLSTM-CTC | 低中层 |
| E13e | 5 | 2-layer BiLSTM-CTC | 接近已有 layer 6 |
| E13f | 7 | 2-layer BiLSTM-CTC | layer 6 与 9 之间 |
| E13g | 8 | 2-layer BiLSTM-CTC | layer 9 附近 |
| E13h | 10 | 2-layer BiLSTM-CTC | layer 9 附近 |
| E13i | 11 | 2-layer BiLSTM-CTC | 接近最终层 |

若时间有限，优先运行：

```text
layer 7, layer 8, layer 10, layer 11
```

### 3.3 实现要求

扩展现有脚本：

```text
scripts/extract_ssl_features.py
scripts/train_cached_ctc_head.py
scripts/evaluate_representation_ctc.py
```

需要新增参数：

```bash
--layers 1 2 3 4 5 7 8 10 11
--experiment_prefix e13_layerwise
--skip_existing
```

缓存目录：

```text
features/wav2vec2_base/train_layer<L>/
features/wav2vec2_base/dev_layer<L>/
features/wav2vec2_base/test_layer<L>/
```

### 3.4 推荐命令

```bash
# 1. 提取所有新增层特征
python scripts/extract_ssl_features.py \
  --model_name_or_path facebook/wav2vec2-base \
  --train_manifest data/manifests/train_10h_effective_15s.jsonl \
  --dev_manifest data/manifests/dev_clean_effective_15s.jsonl \
  --test_manifest data/manifests/test_clean.jsonl \
  --layers 1 2 3 4 5 7 8 10 11 \
  --output_dir features/wav2vec2_base \
  --dtype fp16 \
  --skip_existing

# 2. 训练每层 BiLSTM-CTC head
bash scripts/train_deep_dive_layerwise.sh

# 3. 根据 dev WER 选择最佳层
python scripts/select_best_representation_layer.py \
  --summary_glob 'exp/deep_dive/e13_layer*_bilstm_ctc/summary.json' \
  --out artifacts/deep_dive/best_layer_e13.txt
```

### 3.5 Codex 任务

```text
Modify scripts/extract_ssl_features.py and scripts/train_cached_ctc_head.py so they support arbitrary Wav2Vec2 transformer layer indices 1-12, skip already extracted features, and write one summary.json per layer. Add scripts/train_deep_dive_layerwise.sh to run E13 for layers 1,2,3,4,5,7,8,10,11. The script must not overwrite existing E10 layer 6/9/12 artifacts. It should append dev/test metrics to results/deep_dive_metrics.csv.
```

### 3.6 验收标准

```text
[ ] 每个 layer 都有 summary.json。
[ ] 每个 layer 都保存 dev prediction。
[ ] dev WER 曲线可由 CSV 绘制。
[ ] 只有 dev 最优层和关键层需要完整 test。
[ ] results/figures/layerwise_wer_curve.png 生成成功。
```

---

## 4. E14：learned scalar mixture 层融合

### 4.1 研究问题

```text
手工选择单层是否足够？
不同层是否存在互补信息？
```

### 4.2 方法

对所有 hidden states 做可学习加权：

```math
h_t = \sum_{l=0}^{12} \mathrm{softmax}(\alpha)_l h_{t,l}
```

其中：

```text
hidden_states[0]  = CNN projection 后、Transformer 前
hidden_states[1]  = Transformer block 1 输出
...
hidden_states[12] = Transformer block 12 输出
```

### 4.3 实验设置

| 项 | 配置 |
|---|---|
| ID | E14 |
| Encoder | frozen Wav2Vec2-base |
| Input | all hidden states, learned scalar mixture |
| Downstream | 2-layer BiLSTM-CTC |
| Trainable | scalar weights + BiLSTM head |
| Data | 6.392h effective train set |
| 对照 | E10 layer 9 / E13 best single layer |

### 4.4 实现建议

新增脚本：

```text
scripts/train_layer_mixture_ctc_head.py
```

或扩展：

```text
scripts/train_cached_ctc_head.py --representation_type layer_mixture
```

缓存策略：

```text
优先不重复保存 13 份完整特征。
可选方案 A：训练时从各层 cache 读取并加权。
可选方案 B：预先保存每条 utterance 的 [num_layers, T, D] FP16 tensor。
```

如果磁盘不足，使用方案 A。

### 4.5 输出要求

`summary.json` 必须包含：

```json
{
  "experiment_id": "E14",
  "representation": "learned_layer_mixture",
  "dev_wer": 0.0,
  "test_wer": 0.0,
  "layer_weights": {
    "0": 0.0,
    "1": 0.0,
    "...": 0.0,
    "12": 0.0
  }
}
```

### 4.6 Codex 任务

```text
Implement learned scalar mixture for cached Wav2Vec2 hidden states. Add a module LayerScalarMixture with 13 learnable scalar parameters initialized to zeros, softmax-normalized at forward time. Add a training script that loads cached layer features for the same utterance, computes the weighted representation, and feeds it to the existing BiLSTM-CTC head. Save learned layer weights to summary.json and export a bar plot of layer weights.
```

### 4.7 验收标准

```text
[ ] E14 dev/test WER 可计算。
[ ] learned layer weights 非 NaN，sum≈1。
[ ] results/figures/e14_layer_mixture_weights.png 生成成功。
[ ] 与 E13 best single layer 可并排比较。
```

---

## 5. E15：downstream head 容量消融

### 5.1 研究问题

```text
E1 失败是 frozen 表征无效，还是 linear CTC head 太弱？
序列建模 head 对 frozen SSL feature 是否必要？
```

### 5.2 固定条件

```text
Encoder: frozen facebook/wav2vec2-base
Layer: E13 dev 最优层；若 E13 未完成，则使用 layer 9
Train data: 6.392h effective train set
Evaluation: dev for checkpoint selection, test-clean for final evaluation
```

### 5.3 实验矩阵

| ID | Head | 配置 | 目的 |
|---|---|---|---|
| E15a | Linear CTC | `768 -> vocab_size` | 对齐 E1，但使用 best middle layer |
| E15b | MLP CTC | `768 -> 512 -> vocab_size`, GELU, dropout | 检查非线性映射是否足够 |
| E15c | 1-layer BiLSTM CTC | hidden size 256 | 检查序列上下文是否关键 |
| E15d | 2-layer BiLSTM CTC | 沿用 E10 | 对照 E10 best |
| E15e | small Transformer CTC | 2 layers, d_model 256/384, 4 heads | 可选，比较另一类序列 head |

### 5.4 训练参数建议

| Head | learning rate | max steps | dropout |
|---|---:|---:|---:|
| Linear | 1e-3 | 3000 | 0.1 |
| MLP | 1e-3 | 3000 | 0.1 |
| 1-layer BiLSTM | 8e-4 | 3000 | 0.1 |
| 2-layer BiLSTM | 沿用 E10 | 沿用 E10 | 沿用 E10 |
| small Transformer | 5e-4 | 3000 | 0.1 |

### 5.5 需要实现的 head 类型

建议在 `src/ssl_asr/representations.py` 中抽象：

```python
class LinearCTCHead(nn.Module):
    ...

class MLPCTCHead(nn.Module):
    ...

class BiLSTMCTCHead(nn.Module):
    ...

class TransformerCTCHead(nn.Module):
    ...
```

`train_cached_ctc_head.py` 增加参数：

```bash
--head_type linear|mlp|bilstm|transformer
--num_lstm_layers 1|2
--hidden_size 256
--mlp_hidden_size 512
--num_transformer_layers 2
--num_attention_heads 4
```

### 5.6 推荐命令

```bash
bash scripts/train_deep_dive_head_capacity.sh \
  --layer $(cat artifacts/deep_dive/best_layer_e13.txt || echo 9)
```

### 5.7 Codex 任务

```text
Refactor the cached representation CTC training code to support multiple downstream heads: linear, MLP, 1-layer BiLSTM, 2-layer BiLSTM, and small Transformer. Add scripts/train_deep_dive_head_capacity.sh to run E15a-E15d on the selected best Wav2Vec2 layer. Save trainable parameter counts and RTF for every head. Make E15e optional behind a command-line flag.
```

### 5.8 验收标准

```text
[ ] E15a–E15d 均有 dev WER。
[ ] 至少 E15a/E15b/E15c/E15d 有 trainable params。
[ ] 每个 head 的 prediction 保存成功。
[ ] 输出表格：head type vs WER vs trainable params vs RTF。
```

---

## 6. E16：top-k partial fine-tuning 网格

### 6.1 研究问题

```text
top-3 明显不够，top-6 接近 full fine-tune。
性能拐点究竟在 top-4、top-5 还是 top-6？
```

### 6.2 实验矩阵

已有：

| ID | 可训练范围 | Test WER |
|---|---|---:|
| E6b | top-3 + CTC | 22.41% |
| E6a | top-6 + CTC | 12.14% |
| E2 | full fine-tune | 11.44% |

新增：

| ID | 可训练范围 | `freeze_transformer_layers` | 目的 |
|---|---|---:|---|
| E16a | top-4 + CTC | 8 | 连接 top-3 与 top-6 |
| E16b | top-5 + CTC | 7 | 最重要补点 |
| E16c | top-8 + CTC | 4 | 判断 top-6 以上是否继续接近 full |
| E16d | top-2 + CTC | 10 | 可选，极限低参数 |

### 6.3 实现要求

复用现有 full fine-tuning 脚本，并确保支持：

```bash
--freeze_transformer_layers N
```

冻结逻辑：

```python
for i, layer in enumerate(model.wav2vec2.encoder.layers):
    if i < args.freeze_transformer_layers:
        for p in layer.parameters():
            p.requires_grad = False
```

训练前必须打印：

```text
total parameters
trainable parameters
trainable ratio
frozen layer indices
trainable layer indices
```

### 6.4 推荐训练参数

| 项 | 配置 |
|---|---|
| model | `facebook/wav2vec2-base` |
| data | 6.392h effective train set |
| learning rate | `1e-4` |
| warmup ratio | `0.1` |
| weight decay | `0.005` |
| max steps | 与 E6a/E6b 对齐 |
| save/eval steps | 与 E6a/E6b 对齐 |
| best model | dev WER |

### 6.5 推荐命令

```bash
bash scripts/train_partial_ft_grid.sh top5
bash scripts/train_partial_ft_grid.sh top4
bash scripts/train_partial_ft_grid.sh top8
```

### 6.6 Codex 任务

```text
Add or verify --freeze_transformer_layers support in the CTC fine-tuning script. Implement scripts/train_partial_ft_grid.sh so it can run E16a top-4, E16b top-5, and E16c top-8 partial fine-tuning. Each run must write frozen/trainable layer indices, total params, trainable params, dev/test WER, CER, and prediction files. The script should skip completed runs unless --overwrite is passed.
```

### 6.7 验收标准

```text
[ ] E16b top-5 至少完成。
[ ] results/deep_dive_metrics.csv 有 E16a/E16b/E16c 行。
[ ] results/figures/trainable_params_vs_wer.png 生成成功。
[ ] 表格中包含 E1/E10/E6b/E16a/E16b/E6a/E16c/E2。
```

---

## 7. E17：更大 codebook size 的离散单元

### 7.1 研究问题

```text
K=200 仍远弱于 continuous layer 9。
增大 codebook size 是否继续显著降低 WER？
```

### 7.2 实验矩阵

已有：

| ID | K | WER | Bitrate |
|---|---:|---:|---:|
| E11a | 50 | 71.56% | 274.93 bit/s |
| E11b | 100 | 61.65% | 324.43 bit/s |
| E11c | 200 | 52.60% | 374.31 bit/s |

新增：

| ID | K | 输入层 | 输入形式 |
|---|---:|---|---|
| E17a | 500 | E13 best layer 或 layer 9 | centroid vector |
| E17b | 1000 | E13 best layer 或 layer 9 | centroid vector |

优先只做：

```text
E17a: K=500
```

### 7.3 实现要求

复用：

```text
scripts/train_kmeans_units.py
scripts/train_cached_ctc_head.py
scripts/evaluate_representation_ctc.py
```

K-means 约束：

```text
1. 只用训练集帧拟合 K-means。
2. 最多随机抽取 500,000 帧。
3. 固定 seed=42。
4. 严禁使用 dev/test 帧拟合 codebook。
5. 训练 CTC 时保留原始约 50 Hz 帧级序列，不做相邻重复折叠。
```

### 7.4 需要记录的 unit 指标

每个 K 必须输出：

```text
unit_codebook_size
unit_used_types
unit_utilization
unit_perplexity
unit_token_rate
unit_entropy_bits
unit_bitrate_bps
unit_fixed_width_bitrate_bps
unit_dedup_token_rate
unit_dedup_bitrate_bps
WER
CER
RTF
```

### 7.5 推荐命令

```bash
# K=500
python scripts/train_kmeans_units.py \
  --feature_dir features/wav2vec2_base/train_layer$(cat artifacts/deep_dive/best_layer_e13.txt || echo 9) \
  --k 500 \
  --max_train_frames 500000 \
  --seed 42 \
  --output_dir artifacts/kmeans/deep_dive_layer_best_k500

python scripts/train_cached_ctc_head.py \
  --representation_type discrete_centroid \
  --kmeans_dir artifacts/kmeans/deep_dive_layer_best_k500 \
  --head_type bilstm \
  --output_dir exp/deep_dive/e17_k500_centroid_bilstm_ctc
```

### 7.6 Codex 任务

```text
Extend the discrete unit pipeline to support K=500 and K=1000. Add scripts/train_deep_dive_discrete_k_grid.sh. The script should read the best layer from artifacts/deep_dive/best_layer_e13.txt if available, otherwise fall back to layer 9. It should train K-means only on training frames, generate unit caches, train the same BiLSTM-CTC head, evaluate dev/test, and append unit bitrate metrics to results/deep_dive_metrics.csv.
```

### 7.7 验收标准

```text
[ ] E17a K=500 至少完成。
[ ] codebook utilization、perplexity、bitrate 均成功计算。
[ ] WER vs K 图包含 K=50/100/200/500。
[ ] WER vs empirical bitrate 图生成成功。
```

---

## 8. E18：centroid 输入 vs learned unit embedding

### 8.1 研究问题

```text
E11/E17 使用 centroid vector 作为输入。
如果改成 unit ID + learned embedding，会不会更适合 CTC downstream training？
```

### 8.2 实验矩阵

| ID | K | 输入形式 | Embedding dim | Head |
|---|---:|---|---:|---|
| E18a | 200 | centroid vector | 768 | 已有 E11c 对照 |
| E18b | 200 | learned embedding | 256 | BiLSTM-CTC |
| E18c | 500 | learned embedding | 256 | 如果 E17a 有效再做 |
| E18d | 200 | learned embedding | 768 | 可选，控制维度 |

### 8.3 注意事项

这组实验不再是“纯粹连续 vs 离散”的公平比较，而是“离散 token 的 downstream 建模方式”比较。论文中应单独说明：

```text
Centroid input preserves the geometry of the SSL representation, while learned embedding treats units as symbolic tokens and learns a task-specific representation.
```

### 8.4 实现要求

新增或扩展模型：

```python
class DiscreteEmbeddingBiLSTMCTC(nn.Module):
    def __init__(self, num_units, embedding_dim, hidden_size, vocab_size):
        ...
```

`train_cached_ctc_head.py` 增加参数：

```bash
--discrete_input_type centroid|embedding
--unit_embedding_dim 256
```

### 8.5 Codex 任务

```text
Implement discrete unit embedding input for the cached CTC pipeline. Add --discrete_input_type centroid|embedding. For embedding mode, load unit IDs rather than centroid vectors, learn nn.Embedding(K, dim), feed embeddings to the existing BiLSTM-CTC head, and report both embedding parameters and total trainable parameters. Add E18b for K=200 and optionally E18c for K=500.
```

### 8.6 验收标准

```text
[ ] E18b K=200 embedding 模式可训练、可评估。
[ ] 与 E11c K=200 centroid 结果可并排比较。
[ ] trainable params 包含 embedding 参数。
[ ] prediction 文件保存成功。
```

---

## 9. E20：数据规模 × frozen best layer + BiLSTM

### 9.1 研究问题

```text
frozen middle-layer + BiLSTM 在 1h/3h/6.392h 下是否比 full fine-tuning 更稳定？
极低资源下冻结 encoder 是否反而有优势？
```

### 9.2 实验矩阵

| ID | Train data | Encoder | Layer | Head |
|---|---:|---|---:|---|
| E20a | 1.001h | frozen Wav2Vec2 | best layer 或 9 | BiLSTM-CTC |
| E20b | 3.001h | frozen Wav2Vec2 | best layer 或 9 | BiLSTM-CTC |
| E10b | 6.392h | frozen Wav2Vec2 | 9 | BiLSTM-CTC |

对照：

| Train data | Full fine-tune | Frozen layer + BiLSTM |
|---:|---|---|
| 1.001h | E3 | E20a |
| 3.001h | E5 | E20b |
| 6.392h | E2 | E10b/E13 best |

### 9.3 实现要求

复用 E10/E13 的缓存训练流程，但换用不同 train manifest：

```text
data/manifests/train_1h_effective_15s.jsonl
data/manifests/train_3h_effective_15s.jsonl
data/manifests/train_10h_effective_15s.jsonl
```

### 9.4 Codex 任务

```text
Add scripts/train_frozen_layer_data_scale.sh to train the selected frozen Wav2Vec2 layer with the same BiLSTM-CTC head on 1h, 3h, and 6.392h effective data. Reuse cached features when possible. The script should compare E20a/E20b/E10b against E3/E5/E2 and produce a data_scale_frozen_vs_finetune.csv and plot.
```

### 9.5 验收标准

```text
[ ] E20a 和 E20b 至少有 dev WER。
[ ] 生成 full fine-tune vs frozen-layer data-scale 曲线。
[ ] 能判断极低资源下 frozen encoder 是否更稳定。
```

---

## 10. E21：错误类型系统分析

### 10.1 研究问题

```text
不同系统的错误究竟差在哪里？
低资源、partial fine-tuning、frozen 表征、离散化分别增加哪类错误？
```

### 10.2 分析对象

| 对比 | 目的 |
|---|---|
| E2 vs E6a | full fine-tune 与 top-6 的差异 |
| E2 vs E6b | 参数不足带来的错误 |
| E2 vs E10/E13 best | full fine-tune 与 frozen middle-layer head 的差异 |
| E10/E13 best vs E11c/E17a | 连续表示离散化损失 |
| E3 vs E5 vs E2 | 数据规模变化带来的错误变化 |
| E2 vs E8 | masking 对错误类型的影响 |

### 10.3 输出指标

每个系统输出：

```text
WER
CER
Substitution count/rate
Deletion count/rate
Insertion count/rate
Exact match count
Average prediction length
Reference/prediction length ratio
Non-empty ratio
Top 30 word confusions
Top 30 deleted words
Top 30 inserted words
Function word error rate
Rare word error rate
Length bucket WER
Speaker-level WER
```

### 10.4 推荐脚本

新增：

```text
scripts/analyze_deep_dive_errors.py
scripts/plot_deep_dive_results.py
```

输出：

```text
results/deep_dive_error_analysis.md
results/deep_dive_error_stats.csv
results/figures/error_type_breakdown.png
results/figures/length_bucket_wer.png
results/figures/top_confusions.md
```

### 10.5 Codex 任务

```text
Implement scripts/analyze_deep_dive_errors.py. It should read multiple prediction JSONL files, compute word-level edit alignments, aggregate substitution/deletion/insertion counts, identify top word confusions, compute WER by duration bucket and speaker if speaker_id is available, and write both CSV and Markdown reports. The script should support pairwise comparison between two systems and multi-system summary tables.
```

### 10.6 验收标准

```text
[ ] 对 E2/E6a/E6b/E10/E11/E3/E5 至少生成错误统计。
[ ] Markdown 报告中有典型错误案例。
[ ] CSV 可直接画图。
[ ] 不依赖 GPU。
```

---

## 11. E22：masking 小网格（低优先级）

### 11.1 研究问题

```text
masking 是否真的无效，还是强度不合适？
```

### 11.2 实验矩阵

| ID | Data | mask_time_prob | mask_feature_prob | 说明 |
|---|---:|---:|---:|---|
| E22a | 1h | 0.02 | 0.00 | 弱 time masking |
| E12a | 1h | 0.05 | 0.00 | 已有 |
| E22b | 1h | 0.02 | 0.02 | 弱 time + feature masking |
| E8 | 6.392h | existing | existing | 已有 |

### 11.3 停止规则

```text
1. step 800 non-empty ratio < 0.01：停止。
2. 连续 5 次 eval WER 不下降：停止。
3. prediction 全 blank / apostrophe：停止。
4. E22 不得挤占 E13/E15/E16/E17 的时间。
```

### 11.4 Codex 任务

```text
Add a small masking grid script for 1h fine-tuning, but make it optional. The script must include early-stop checks for non-empty prediction ratio and blank/apostrophe collapse. Do not run E22 by default in the main deep-dive queue.
```

---

## 12. E23：关键配置第二随机种子（可选）

### 12.1 研究问题

```text
关键新增发现是否稳定？
```

### 12.2 推荐复跑对象

优先级：

```text
1. E13 best layer 或 E10 layer 9。
2. E15 中最优 head。
3. E16 top-5。
4. E17 K=500。
```

只建议使用一个额外 seed：

```text
seed = 1234
```

### 12.3 Codex 任务

```text
Add --seed support to all deep-dive training scripts and create scripts/rerun_best_deep_dive_seed.sh. It should read best configurations from summary files and rerun exactly one selected experiment with seed 1234.
```

---

## 13. 主控脚本设计

### 13.1 新增主脚本

建议新增：

```text
scripts/train_deep_dive_rtx3090.sh
```

支持参数：

```bash
bash scripts/train_deep_dive_rtx3090.sh e13
bash scripts/train_deep_dive_rtx3090.sh e13 e15 e16
bash scripts/train_deep_dive_rtx3090.sh all_p0
bash scripts/train_deep_dive_rtx3090.sh all_p1
bash scripts/train_deep_dive_rtx3090.sh all --overwrite
```

### 13.2 模式定义

```text
all_p0 = E13 + E15
all_p1 = E13 + E14 + E15 + E16 + E17 + E21
all    = E13 + E14 + E15 + E16 + E17 + E18 + E20 + E21 + optional E22/E23
```

默认不要运行：

```text
E22 masking grid
E23 second seed
```

除非显式传入：

```bash
bash scripts/train_deep_dive_rtx3090.sh e22
bash scripts/train_deep_dive_rtx3090.sh e23
```

### 13.3 Codex 任务

```text
Create scripts/train_deep_dive_rtx3090.sh as the entry point for E13-E23. It should parse experiment names, skip completed runs, fail fast on missing prerequisites, and write a final status table. It must not delete existing E1-E12 artifacts. Add logs under logs/deep_dive/ and append all metrics to results/deep_dive_metrics.csv.
```

---

## 14. 结果表格与图像

### 14.1 Layer-wise probing 表

| Layer | Head | Trainable Params | Dev WER | Test WER | CER | RTF |
|---:|---|---:|---:|---:|---:|---:|
| 1 | BiLSTM-CTC | TBD | TBD | TBD | TBD | TBD |
| 2 | BiLSTM-CTC | TBD | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... | ... | ... |
| 12 | BiLSTM-CTC | TBD | TBD | TBD | TBD | TBD |

图：

```text
results/figures/layerwise_wer_curve.png
```

### 14.2 Head capacity 表

| ID | Layer | Head | Trainable Params | Dev WER | Test WER | CER | RTF |
|---|---:|---|---:|---:|---:|---:|---:|
| E15a | best | Linear | TBD | TBD | TBD | TBD | TBD |
| E15b | best | MLP | TBD | TBD | TBD | TBD | TBD |
| E15c | best | 1-layer BiLSTM | TBD | TBD | TBD | TBD | TBD |
| E15d | best | 2-layer BiLSTM | TBD | TBD | TBD | TBD | TBD |

图：

```text
results/figures/head_capacity_wer_params.png
```

### 14.3 Parameter-efficient fine-tuning 表

| ID | Trainable Scope | Trainable Params | Dev WER | Test WER | CER | RTF |
|---|---|---:|---:|---:|---:|---:|
| E1 | CTC head only | 0.023M | TBD | 99.92% | TBD | TBD |
| E10/E13 best | frozen encoder + BiLSTM | 3.695M | TBD | TBD | TBD | TBD |
| E6b | top-3 | 26.403M | TBD | 22.41% | TBD | TBD |
| E16a | top-4 | TBD | TBD | TBD | TBD | TBD |
| E16b | top-5 | TBD | TBD | TBD | TBD | TBD |
| E6a | top-6 | 47.667M | TBD | 12.14% | TBD | TBD |
| E16c | top-8 | TBD | TBD | TBD | TBD | TBD |
| E2 | full | 90.194M | TBD | 11.44% | TBD | TBD |

图：

```text
results/figures/trainable_params_vs_wer.png
```

### 14.4 Discrete unit 表

| ID | Layer | K | Input Type | WER | CER | Used Units | Perplexity | Bitrate | Dedup Bitrate | RTF |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| E11a | 9 | 50 | centroid | 71.56% | TBD | 50 | 45.51 | 274.93 | 143.51 | TBD |
| E11b | 9 | 100 | centroid | 61.65% | TBD | 99 | 90.49 | 324.43 | 187.22 | TBD |
| E11c | 9 | 200 | centroid | 52.60% | TBD | 200 | 180.89 | 374.31 | 236.31 | TBD |
| E17a | best | 500 | centroid | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| E18b | best | 200 | embedding | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

图：

```text
results/figures/discrete_wer_vs_k.png
results/figures/discrete_wer_vs_bitrate.png
results/figures/codebook_utilization.png
```

### 14.5 错误分析表

| System | WER | Substitution | Deletion | Insertion | Exact | Avg Pred Len / Ref Len | Non-empty Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| E6a | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| E6b | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| E10/E13 best | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| E11c/E17a | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 15. 停止规则与降级策略

| 实验 | 观察点 | 停止条件 | 降级动作 |
|---|---|---|---|
| E13 | 每层 step 800 或中期 eval | 全 blank / apostrophe，且 loss 不降 | 跳过该层，继续下一层 |
| E14 | step 800 | layer weights NaN 或 WER 不降 | 放弃 mixture，不影响 E13 |
| E15 | 每个 head step 800 | non-empty ratio < 0.01 | 停止该 head，继续下一个 head |
| E16 | step 1000 | 明显 collapse 或 OOM | 优先保留 top-5，放弃 top-8/top-2 |
| E17 | K-means 阶段 | 内存不足 | 限制采样到 300k 帧，优先 K=500 |
| E18 | step 800 | embedding 模式明显 collapse | 停止 E18，不影响 E17 |
| E20 | step 800 | 1h/3h 全 blank | 只保留 dev 诊断，不继续 test |
| E22 | step 800 | non-empty ratio < 0.01 | 立即停止，禁止重试大网格 |

服务器回收前必须同步：

```text
exp/deep_dive/**/model.pt 或 checkpoint
exp/deep_dive/**/summary.json
exp/deep_dive/**/completion.json
artifacts/deep_dive/*
artifacts/kmeans/deep_dive_*/*
results/deep_dive_metrics.csv
results/deep_dive_error_stats.csv
results/predictions/deep_dive/*.jsonl
logs/deep_dive/*.log
configs/deep_dive/*.yaml
```

不必同步：

```text
features/wav2vec2_base/* 大体积缓存
中间 .tmp 文件
旧 checkpoint 多余副本
```

---

## 16. Codex 总任务清单

### Task 1：统一 deep-dive 配置与目录

```text
Create configs/deep_dive/ and add YAML configs for E13, E14, E15, E16, E17, E18, E20, E21. Create directories for exp/deep_dive, logs/deep_dive, results/predictions/deep_dive, artifacts/deep_dive. Add utility functions for atomic JSON writing, skip-existing behavior, and metrics appending to results/deep_dive_metrics.csv.
```

验收：

```bash
python -m compileall src scripts
```

### Task 2：E13 layer-wise probing

```text
Support arbitrary Wav2Vec2 layer extraction and training cached BiLSTM-CTC heads for layers 1-12. Add scripts/train_deep_dive_layerwise.sh and a plotting script for layer-wise WER.
```

验收：

```bash
bash scripts/train_deep_dive_rtx3090.sh e13
```

### Task 3：E14 learned scalar mixture

```text
Implement LayerScalarMixture and training over cached multi-layer features. Save learned layer weights and plot them.
```

验收：

```bash
bash scripts/train_deep_dive_rtx3090.sh e14
```

### Task 4：E15 head capacity ablation

```text
Add linear, MLP, 1-layer BiLSTM, 2-layer BiLSTM, and optional small Transformer CTC heads. Train them on the selected best frozen layer.
```

验收：

```bash
bash scripts/train_deep_dive_rtx3090.sh e15
```

### Task 5：E16 partial fine-tuning grid

```text
Verify --freeze_transformer_layers support. Add scripts/train_partial_ft_grid.sh for top-4, top-5, and top-8 fine-tuning. Record frozen/trainable layers and parameter counts.
```

验收：

```bash
bash scripts/train_deep_dive_rtx3090.sh e16
```

### Task 6：E17/E18 discrete unit extension

```text
Add K=500/K=1000 support and learned unit embedding mode. Ensure K-means only fits training frames and all unit bitrate metrics are computed.
```

验收：

```bash
bash scripts/train_deep_dive_rtx3090.sh e17
bash scripts/train_deep_dive_rtx3090.sh e18
```

### Task 7：E20 data-scale frozen representation

```text
Train frozen best-layer BiLSTM-CTC on 1h and 3h effective train sets. Produce a comparison plot against E3/E5/E2 full fine-tuning.
```

验收：

```bash
bash scripts/train_deep_dive_rtx3090.sh e20
```

### Task 8：E21 error analysis

```text
Implement deep-dive error analysis for multiple systems, including S/D/I counts, top confusions, function word errors, duration-bucket WER, and Markdown reports.
```

验收：

```bash
python scripts/analyze_deep_dive_errors.py \
  --pred_glob 'results/predictions/**/*.jsonl' \
  --out_csv results/deep_dive_error_stats.csv \
  --out_md results/deep_dive_error_analysis.md
```

### Task 9：主控脚本

```text
Create scripts/train_deep_dive_rtx3090.sh with modes e13, e14, e15, e16, e17, e18, e20, e21, all_p0, all_p1, all. It should check prerequisites, skip completed runs, and print a final status table.
```

验收：

```bash
bash scripts/train_deep_dive_rtx3090.sh all_p0
```

---

## 17. 论文结果章节建议

新增实验完成后，Results and Analysis 可组织为：

```text
5.1 Effect of Labeled Data Scale
    E3, E5, E2

5.2 Parameter-efficient Fine-tuning
    E1, E10/E13 best, E6b, E16a, E16b, E6a, E16c, E2

5.3 Layer-wise Analysis of Frozen SSL Representations
    E10a, E10b, E10c, E13, E14

5.4 Downstream Capacity for Frozen SSL Features
    E15a, E15b, E15c, E15d

5.5 Continuous versus Discrete Speech Representations
    E10/E13 best, E11a, E11b, E11c, E17a, E18b

5.6 Error Analysis
    E2, E6a, E6b, E10/E13 best, E11/E17, E3/E5/E2

5.7 Efficiency and Compression Trade-offs
    WER, trainable params, RTF, bitrate
```

---

## 18. 最终执行建议

最推荐的新增实验闭环：

```text
E13：回答“哪一层最有用”。
E15：回答“frozen 表征需要怎样的 head”。
E16：回答“partial fine-tuning 拐点在哪里”。
E17/E18：回答“离散化损失来自哪里”。
E21：把所有结论落到错误类型上。
```

不要优先做：

```text
1. 大量 masking 网格。
2. 继续堆 HuBERT 等第三个 SSL encoder。
3. 多 seed 全量复现。
4. 引入 LM beam search。
5. 大规模重构训练框架。
```

最终目标不是刷新最低 WER，而是形成一条更清晰的研究结论：

```text
低资源 ASR 中，自监督语音表征的价值不只取决于是否 fine-tune，
还强烈取决于 hidden layer、downstream head、参数更新范围以及离散化方式。
```
