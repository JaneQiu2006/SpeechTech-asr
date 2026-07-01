# 剩余 RTX 3090 Ti 训练时间实验规划文档

> 用途：本文件用于在已有低资源 ASR 项目基本完成训练闭环后，指导剩余租赁 GPU 时间的新增训练实验。文档假设：`test-clean` 评测、RTF、错误分析、metrics 清洗和结果传输均可在本地完成；服务器时间优先用于必须依赖 GPU 的新训练与训练型消融。

---

## 0. 当前状态与范围锁定

### 0.1 已有实验基础

当前项目已经完成或基本完成以下主线实验：

| ID | 系统 | 数据 | 状态 | 当前可用结论 |
|---|---|---:|---|---|
| E1 | Frozen Wav2Vec2 + CTC | 10h | 已完成 3 epoch，计划扩展至 30 epoch | 3 epoch 下 frozen linear CTC 基本失败 |
| E2 | Fine-tuned Wav2Vec2 + CTC | 10h | 已完成 | 当前最佳主模型，修复 padding 伪字符后 dev WER 约 11.09% |
| E3 | Fine-tuned Wav2Vec2 + CTC | 1h | 已完成但 collapse，计划 debug + retrain | 不能直接支撑 1h vs 10h 数据规模消融 |
| E4 | Fine-tuned WavLM + CTC | 10h | 已完成 | dev WER 约 16.60%，可用于 SSL 模型对比 |

当前最重要的结构性缺口：

```text
1. E3 1h 发生 blank-output collapse，数据规模消融尚不成立。
2. E1 frozen baseline 训练轮数偏少，计划扩展到 30 epoch。
3. E1 vs E2 对比过于极端：一个只训练 CTC head，一个训练约 90M 参数。
4. 当前已有 Wav2Vec2 vs WavLM，但还没有 HuBERT 作为第三种 SSL encoder。
5. 剩余服务器时间应优先做“训练型消融”，而不是 test/RTF/错误分析。
```

---

### 0.2 已确定要做的实验

你已经决定在服务器上继续执行：

| ID | 实验 | 目的 |
|---|---|---|
| E1-30 | Frozen Wav2Vec2 + CTC 训练至 30 epoch | 判断 frozen linear CTC 是否只是欠训练 |
| E3r | 1h Wav2Vec2 debug + retrain | 抢救极低资源数据规模消融 |

在完成以上两项后，保守估计仍有：

```text
可用于新增训练的时间：约 9–10 小时
```

本文主要规划这 9–10 小时的训练优先级。

---

### 0.3 新增训练的总体原则

新增训练不应追求“模型越多越好”，而应服务于报告中的分析维度。

优先级排序原则：

```text
第一优先级：补齐数据规模曲线。
第二优先级：补齐参数更新范围消融。
第三优先级：补齐不同 SSL encoder 对比。
第四优先级：探索训练稳定性或正则化。
第五优先级：更复杂 frozen downstream head，除非时间充裕。
```

不建议做：

```text
1. 不做 large 级别模型。
2. 不临时切换到 TTS。
3. 不从零搭建离散单元 ASR 主线。
4. 不做多 seed 完整复现实验。
5. 不把已监督 fine-tuned ASR checkpoint 当作主实验。
```

---

## 1. 推荐新增实验总表

### 1.1 最高推荐队列

在完成 E1-30 和 E3r 后，推荐执行：

```text
E5  ：Wav2Vec2 3h data-scale bridge
E6a ：Wav2Vec2 top-6-layer fine-tuning
E6b ：Wav2Vec2 top-3-layer fine-tuning
E7  ：HuBERT-base 10h fine-tuning（有剩余时间再启动）
```

对应优先级：

| 优先级 | ID | 实验 | 预计训练时间 | 推荐程度 | 主要论文价值 |
|---|---|---|---:|---|---|
| P0 | E5 | 3h data-scale bridge | 2–3 h | 必做 | 形成 0.6h / 3h / 6.4h 数据规模曲线 |
| P1 | E6a | top-6-layer fine-tune | 2–2.5 h | 强烈建议 | 参数量-性能权衡 |
| P1 | E6b | top-3-layer fine-tune | 1.5–2.5 h | 建议 | 参数高效微调极限 |
| P2 | E7 | HuBERT-base 10h fine-tune | 3–5 h | 有时间再做 | 三种 SSL encoder 对比 |
| P2 | E8 | SpecAugment / masking regularization | 2–3 h | 可选 | 低资源训练稳定性分析 |
| P3 | E9 | Frozen encoder + stronger CTC head | 3–4 h | 不优先 | 判断 E1 失败是否由 linear head 太弱导致 |

---

### 1.2 最小新增训练组合

如果剩余时间被压缩，只做以下三个：

```text
1. E5  ：3h data-scale bridge
2. E6a ：top-6-layer fine-tune
3. E6b ：top-3-layer fine-tune
```

这三个实验能让报告从“跑通了几个模型”升级为系统回答：

```text
1. 标注数据规模增加是否改善低资源 ASR？
2. 是否必须 full fine-tune 整个 SSL encoder？
3. 可训练参数量减少时，WER 如何变化？
```

---

## 2. 方向一：数据规模消融补强

### 2.1 背景动机

课程作业建议分析低资源训练数据规模对性能的影响。当前 E3 的实际有效数据约 0.591h，且发生 blank-output collapse，因此不能直接把 E3 的 100% WER 解释为正常的数据规模退化。

因此，除了 E3r 重新训练外，必须补一个中间规模点。

---

## 2.2 E5：Wav2Vec2 3h data-scale bridge

### 研究问题

```text
从极低资源 0.591h 增加到约 3h 后，Wav2Vec2 + CTC fine-tuning 是否变得稳定？
WER/CER 是否随标注数据量增加而明显下降？
```

### 实验设置

| 项 | 配置 |
|---|---|
| ID | E5 |
| 名称 | Wav2Vec2 fine-tune 3h |
| 模型 | `facebook/wav2vec2-base` |
| 训练数据 | 从 `train_10h.jsonl` 中固定 seed 抽取 3h |
| 评估数据 | `dev_clean.jsonl`，与 E2 保持一致 |
| 音频时长限制 | 建议沿用 `max_duration_in_seconds=15` |
| 训练策略 | 仅冻结 CNN feature encoder，Transformer encoder + CTC head 可训练 |
| learning rate | `7e-5` 或 `1e-4` |
| warmup ratio | `0.15` |
| weight decay | `0.005` |
| CTC loss | `mean` |
| ctc zero infinity | `true` |
| max steps | `1800` 或 `2000` |
| eval steps | `200` |
| save steps | `200` |
| save total limit | `5` |
| best model | 按 dev WER 选择 |

### 为什么选择 3h 而不是 5h

推荐数据规模曲线：

```text
0.591h → 约 3h → 6.392h
```

不优先选择 5h 的原因：

```text
1. 5h 与 E2 的 6.392h 太接近，区分度不强。
2. 3h 更能体现从极低资源到中等低资源的变化。
3. 3h 训练时间更短，更适合当前剩余算力。
```

### 预期论文表格

| ID | 名义数据 | 实际数据 | 模型 | 策略 | WER ↓ | CER ↓ |
|---|---:|---:|---|---|---:|---:|
| E3r | 1h | 约 0.591h | Wav2Vec2 | fine-tune | TBD | TBD |
| E5 | 3h | TBD | Wav2Vec2 | fine-tune | TBD | TBD |
| E2 | 10h | 约 6.392h | Wav2Vec2 | fine-tune | TBD | TBD |

### 结果解释模板

如果 E3r 成功：

```text
Increasing labeled speech from 0.591h to about 3h and 6.392h progressively reduces WER/CER, showing that SSL representations still benefit substantially from additional labeled data in low-resource ASR.
```

如果 E3r 仍失败：

```text
Under the extremely low-resource condition with only 0.591h effective speech, CTC fine-tuning remains unstable and collapses to blank outputs. Increasing the labeled data to around 3h restores training stability, and further increasing it to 6.392h yields the best recognition performance.
```

---

## 3. 方向二：参数更新范围消融

### 3.1 背景动机

当前 E1 和 E2 的对比过于极端：

| 实验 | 可训练部分 | 解释问题 |
|---|---|---|
| E1 | 仅 CTC head | 训练参数极少，可能 head 表达力不足或训练不足 |
| E2 | Transformer encoder + CTC head | 训练约 90M 参数，性能最好 |

因此 E1 vs E2 不能单独回答：

```text
到底是 frozen SSL representation 不够，还是可训练参数太少？
是否必须 full fine-tune 整个 encoder？
```

为此需要 partial fine-tuning 实验。

---

## 3.2 E6a：Wav2Vec2 top-6-layer fine-tuning

### 研究问题

```text
只微调 Wav2Vec2 高层 Transformer block，是否可以接近 full fine-tune 的识别性能？
```

### 实验设置

| 项 | 配置 |
|---|---|
| ID | E6a |
| 名称 | Wav2Vec2 top-6-layer fine-tune |
| 模型 | `facebook/wav2vec2-base` |
| 数据 | `train_10h.jsonl`，保持 ≤15s 过滤 |
| 冻结范围 | CNN feature encoder + Transformer bottom 6 layers |
| 可训练范围 | Transformer top 6 layers + CTC head |
| learning rate | `1e-4` |
| warmup ratio | `0.1` |
| weight decay | `0.005` |
| max steps | `2500`，或 20–30 epoch |
| eval steps | `300` 或 `500` |
| save total limit | `3` |

### 预期结论

如果 E6a 接近 E2：

```text
Updating only the high-level Transformer blocks recovers most of the benefit of full fine-tuning, suggesting that parameter-efficient adaptation is effective for low-resource ASR.
```

如果 E6a 明显差于 E2：

```text
The performance gap between top-layer adaptation and full fine-tuning indicates that deeper encoder adaptation remains important in this CTC-based low-resource setup.
```

---

## 3.3 E6b：Wav2Vec2 top-3-layer fine-tuning

### 研究问题

```text
进一步减少可训练参数时，低资源 ASR 的性能会下降多少？
```

### 实验设置

| 项 | 配置 |
|---|---|
| ID | E6b |
| 名称 | Wav2Vec2 top-3-layer fine-tune |
| 模型 | `facebook/wav2vec2-base` |
| 数据 | `train_10h.jsonl`，保持 ≤15s 过滤 |
| 冻结范围 | CNN feature encoder + Transformer bottom 9 layers |
| 可训练范围 | Transformer top 3 layers + CTC head |
| learning rate | `1e-4` 或 `1.5e-4` |
| warmup ratio | `0.1` |
| weight decay | `0.005` |
| max steps | `2000–2500` |
| eval steps | `300` |
| save total limit | `3` |

### 推荐执行顺序

```text
先做 E6a，再做 E6b。
```

原因：

```text
1. top-6 成功概率更高。
2. top-3 更像参数压缩极限实验。
3. 如果 top-3 失败，也不会影响主线结论。
```

---

## 3.4 训练脚本实现建议

建议在 `scripts/train_ctc.py` 中增加参数：

```text
--freeze_transformer_layers N
```

语义：冻结 Wav2Vec2 Transformer encoder 的前 N 层。

伪代码：

```python
# Wav2Vec2ForCTC: model.wav2vec2.encoder.layers
for i, layer in enumerate(model.wav2vec2.encoder.layers):
    if i < args.freeze_transformer_layers:
        for p in layer.parameters():
            p.requires_grad = False
```

对应关系：

| 实验 | `freeze_transformer_layers` | 可训练 Transformer 层 |
|---|---:|---:|
| E2 | 0 | 12 层 |
| E6a | 6 | top 6 层 |
| E6b | 9 | top 3 层 |
| E1 | 12，并冻结其他 encoder 参数 | 0 层 |

训练前必须打印：

```text
total parameters
trainable parameters
trainable ratio
frozen layer indices
trainable layer indices
```

---

## 3.5 预期论文表格

| ID | Strategy | Trainable Scope | Trainable Params | WER ↓ | CER ↓ |
|---|---|---|---:|---:|---:|
| E1-30 | Frozen encoder | CTC head only | TBD | TBD | TBD |
| E6b | Partial fine-tune | top 3 layers + CTC | TBD | TBD | TBD |
| E6a | Partial fine-tune | top 6 layers + CTC | TBD | TBD | TBD |
| E2 | Full fine-tune | all Transformer layers + CTC | TBD | TBD | TBD |

这张表是新增实验中最值得放入论文主结果的表之一。

---

## 4. 方向三：不同 SSL encoder 对比

### 4.1 背景动机

当前已有：

| ID | SSL encoder | 结果用途 |
|---|---|---|
| E2 | Wav2Vec2-base | 主模型 |
| E4 | WavLM-base | 不同 SSL 模型对比 |

如果补充 HuBERT-base，则可形成三种典型 SSL encoder 对比：

```text
Wav2Vec2 vs WavLM vs HuBERT
```

这与作业建议方向“不同自监督模型的比较”高度匹配。

---

## 4.2 E7：HuBERT-base 10h fine-tuning

### 研究问题

```text
不同自监督预训练目标得到的 speech encoder，在低资源 CTC ASR 中是否表现不同？
```

### 实验设置

| 项 | 配置 |
|---|---|
| ID | E7 |
| 名称 | HuBERT-base fine-tune 10h |
| 模型 | `facebook/hubert-base-ls960` |
| 数据 | `train_10h.jsonl`，保持 ≤15s 过滤 |
| 训练策略 | 仅冻结 CNN feature encoder |
| 可训练部分 | HuBERT Transformer encoder + CTC head |
| vocabulary | 沿用项目字符级 CTC vocabulary |
| learning rate | `1e-4` |
| warmup ratio | `0.1` |
| weight decay | `0.005` |
| CTC loss | `mean` |
| ctc zero infinity | `true` |
| epochs | 30，或与 E2 对齐的 max_steps |
| eval steps | `500` |
| save total limit | `2` |

### 执行条件

只有在以下条件满足时才启动 E7：

```text
1. E5 已完成。
2. E6a 至少已完成。
3. 剩余服务器时间 ≥ 3.5 小时。
4. HuBERT 启动测试 30 分钟内没有 OOM 或结构兼容问题。
```

### 预期论文表格

| ID | SSL Encoder | Training Strategy | Train Data | WER ↓ | CER ↓ |
|---|---|---|---:|---:|---:|
| E2 | Wav2Vec2-base | fine-tune | 10h | TBD | TBD |
| E4 | WavLM-base | fine-tune | 10h | TBD | TBD |
| E7 | HuBERT-base | fine-tune | 10h | TBD | TBD |

### 结果解释模板

```text
Although all three models are self-supervised speech encoders, their pretraining objectives differ. The comparison shows how such differences transfer to low-resource CTC-based ASR under the same training data, vocabulary, decoding, and optimization setup.
```

---

## 5. 方向四：训练稳定性与正则化消融

### 5.1 背景动机

当前项目中已经观察到：

```text
1. E2 的早期 3-epoch 配置发生输出塌缩。
2. E3 1h 配置发生 blank-output collapse。
3. 修正为更多训练步、较低学习率、warmup、weight decay、mean CTC loss 后，E2 才得到有效模型。
```

因此，训练稳定性本身可以作为分析方向。

---

## 5.2 E8：SpecAugment / masking regularization

### 研究问题

```text
在低资源 CTC fine-tuning 中，SpecAugment-style masking 是改善泛化，还是加重优化难度？
```

### 实验设置

| 项 | 配置 |
|---|---|
| ID | E8 |
| 名称 | Wav2Vec2 10h + masking regularization |
| 模型 | `facebook/wav2vec2-base` |
| 数据 | `train_10h.jsonl`，保持 ≤15s 过滤 |
| 基线 | E2 |
| 改动 | 开启或增强 time masking / feature masking |
| learning rate | `1e-4` |
| warmup ratio | `0.1` |
| weight decay | `0.005` |
| epochs | 20–30 |
| eval steps | `500` |

推荐只做一个配置：

```text
mask_time_prob = 0.05
mask_feature_prob = 0.0 或 0.05
```

### 执行条件

E8 只有在以下情况下才做：

```text
1. 训练脚本已经容易暴露 mask_time_prob / mask_feature_prob。
2. 不需要大规模改动工程结构。
3. E5 和 E6a 已完成。
4. 剩余时间不足以完整做 E7，但足够做一个 E8。
```

### 结果解释模板

| 结果 | 解释 |
|---|---|
| WER 降低 | masking 提升低资源泛化能力 |
| WER 升高 | 低资源下额外扰动导致优化更困难 |
| collapse | CTC fine-tuning 对增强强度敏感 |

---

## 6. 方向五：更强 frozen baseline

### 6.1 背景动机

E1 frozen baseline 只训练线性 CTC head。如果 E1-30 仍然失败，可能有两种原因：

```text
1. frozen Wav2Vec2 representation 在当前设置下不够适配 ASR。
2. linear CTC head 表达力太弱，无法充分利用 frozen representation。
```

为了区分这两个原因，可以设计 stronger downstream head。

---

## 6.2 E9：Frozen Wav2Vec2 + stronger CTC head

### 研究问题

```text
冻结 SSL encoder 后，增加下游模型容量是否能显著改善 ASR 性能？
```

### 实验设置

| 项 | 配置 |
|---|---|
| ID | E9 |
| 名称 | Frozen Wav2Vec2 + stronger CTC head |
| encoder | frozen `facebook/wav2vec2-base` |
| downstream | 2-layer BiLSTM + linear CTC，或 2-layer Transformer + CTC |
| 数据 | `train_10h.jsonl`，保持 ≤15s 过滤 |
| learning rate | `1e-3` for downstream head |
| max steps | `2500–3000` |
| eval steps | `300` |
| save total limit | `3` |

### 推荐程度

不优先做。

原因：

```text
1. 需要改模型结构，工程风险明显高于 E5/E6/E7。
2. E1-30 已经会回答 linear CTC head 多训练是否足够。
3. 当前更缺的是数据规模和参数更新范围消融。
```

只有在 E5、E6a、E6b 都完成后仍有较多剩余时间，才考虑 E9。

---

## 7. 推荐执行队列

### 7.1 队列 A：最稳妥，推荐

在已计划的 E1-30 和 E3r 之外，执行：

```text
1. E5  ：Wav2Vec2 3h bridge
2. E6a ：Wav2Vec2 top-6-layer fine-tune
3. E6b ：Wav2Vec2 top-3-layer fine-tune
4. E7  ：HuBERT-base 10h fine-tune（有 ≥3.5h 剩余时启动）
```

适用目标：

```text
保证至少产生 3 个新的、可写入论文的训练型消融。
```

---

### 7.2 队列 B：如果想强调不同 SSL 模型

```text
1. E5  ：Wav2Vec2 3h bridge
2. E7  ：HuBERT-base 10h fine-tune
3. E6a ：Wav2Vec2 top-6-layer fine-tune
4. E6b ：Wav2Vec2 top-3-layer fine-tune（有时间再做）
```

适用目标：

```text
把报告重点放在 Wav2Vec2 / WavLM / HuBERT 的模型比较。
```

风险：

```text
HuBERT 可能占用较长时间，挤压参数范围消融。
```

---

### 7.3 队列 C：如果 E3r 仍然失败

```text
1. 停止继续反复调 1h。
2. E5  ：3h bridge。
3. E5b ：5h bridge，可选。
4. E6a ：top-6 fine-tune。
5. E6b ：top-3 fine-tune。
```

报告叙事改为：

```text
Under the extremely low-resource condition with only 0.591h effective speech, CTC fine-tuning is unstable and collapses to blank outputs. Increasing the labeled data to 3h substantially improves training stability, and further increasing it to 6.392h yields the best recognition performance.
```

---

## 8. 时间规划建议

假设 E1-30 和 E3r 已经完成，剩余 9–10 小时：

| 时间段 | 实验 | 决策 |
|---|---|---|
| 0–3 h | E5 3h bridge | 必做，优先完成 |
| 3–5.5 h | E6a top-6 | 强烈建议完成 |
| 5.5–7.5 h | E6b top-3 | 建议完成 |
| 7.5–10 h | E7 HuBERT 或 E8 SpecAugment | 若剩余 ≥3.5h 做 E7，否则做 E8 或停止训练 |

如果训练速度比预期慢：

```text
保 E5 + E6a，放弃 E6b/E7。
```

如果训练速度比预期快：

```text
完成 E5 + E6a + E6b 后启动 E7。
```

---

## 9. 停止规则

为了避免浪费租赁时间，每个实验必须设置停止规则。

| 实验 | 观察点 | 停止条件 |
|---|---|---|
| E1-30 | epoch 10 / 15 | 仍全 blank 且 eval loss 基本不降，可提前停 |
| E3r | step 800 | 仍全 blank / apostrophe，停止，不再死磕 1h |
| E5 | step 800–1000 | 若仍完全 collapse，先检查 lr 和数据；不要盲跑满 |
| E6a | step 1000 | 若 WER 明显低于 E1，继续；若完全 collapse，停止 |
| E6b | step 1000 | 若明显差于 E6a 且无改善趋势，停止 |
| E7 | 启动后 30 min | OOM 或结构兼容问题无法解决，放弃 |
| E8 | step 1000 | 若明显劣于 E2 且 loss 不降，停止 |
| E9 | 启动后 1 h | 若工程改动导致训练不稳定，放弃 |

---

## 10. 结果组织方式

新增实验完成后，建议将结果组织为三张核心表。

### 10.1 数据规模表

| ID | Model | Effective Train Data | Fine-tuning Strategy | WER ↓ | CER ↓ |
|---|---|---:|---|---:|---:|
| E3r | Wav2Vec2-base | 0.591h | full fine-tune | TBD | TBD |
| E5 | Wav2Vec2-base | about 3h | full fine-tune | TBD | TBD |
| E2 | Wav2Vec2-base | 6.392h | full fine-tune | TBD | TBD |

---

### 10.2 参数更新范围表

| ID | Model | Trainable Scope | Trainable Params | WER ↓ | CER ↓ |
|---|---|---|---:|---:|---:|
| E1-30 | Wav2Vec2-base | CTC head only | TBD | TBD | TBD |
| E6b | Wav2Vec2-base | top 3 layers + CTC | TBD | TBD | TBD |
| E6a | Wav2Vec2-base | top 6 layers + CTC | TBD | TBD | TBD |
| E2 | Wav2Vec2-base | all Transformer layers + CTC | TBD | TBD | TBD |

---

### 10.3 SSL encoder 对比表

| ID | SSL Encoder | Train Data | Strategy | WER ↓ | CER ↓ |
|---|---|---:|---|---:|---:|
| E2 | Wav2Vec2-base | 10h | fine-tune | TBD | TBD |
| E4 | WavLM-base | 10h | fine-tune | TBD | TBD |
| E7 | HuBERT-base | 10h | fine-tune | TBD | TBD |

---

## 11. 论文结构更新建议

原报告的 Results and Analysis 可以改成：

```text
5. Results and Analysis
   5.1 Main Results
   5.2 Effect of Labeled Data Scale
   5.3 Frozen, Partial, and Full Fine-tuning
   5.4 Comparison of SSL Encoders
   5.5 Training Cost and Parameter Efficiency
   5.6 Error Analysis
```

新增实验对应关系：

| 小节 | 使用实验 |
|---|---|
| 5.1 Main Results | E1-30, E2, E3r/E5, E4 |
| 5.2 Effect of Labeled Data Scale | E3r, E5, E2 |
| 5.3 Frozen, Partial, and Full Fine-tuning | E1-30, E6b, E6a, E2 |
| 5.4 Comparison of SSL Encoders | E2, E4, E7 |
| 5.5 Training Cost and Parameter Efficiency | E1-30, E6a, E6b, E2 |
| 5.6 Error Analysis | 本地完成，不占 GPU |

---

## 12. 最终推荐结论

在当前条件下，最值得做的新增训练是：

```text
第一：E5 3h data-scale bridge
第二：E6a top-6-layer fine-tune
第三：E6b top-3-layer fine-tune
第四：E7 HuBERT-base 10h fine-tune
第五：E8 SpecAugment / masking regularization
第六：E9 stronger frozen downstream head
```

最低执行目标：

```text
完成 E5 + E6a。
```

理想执行目标：

```text
完成 E5 + E6a + E6b + E7。
```

如果 E3r 成功，最终论文可以形成：

```text
0.591h → 3h → 6.392h 的数据规模趋势；
frozen → top-3 → top-6 → full fine-tune 的参数效率趋势；
Wav2Vec2 → WavLM → HuBERT 的 SSL encoder 对比。
```

如果 E3r 失败，最终论文仍然可以形成：

```text
极低资源 CTC collapse 现象；
3h 后训练稳定性恢复；
partial fine-tuning 的参数效率分析；
不同 SSL encoder 的迁移性能比较。
```

这比继续重复普通 10h full fine-tune 更能提升最终报告的分析深度和可信度。
