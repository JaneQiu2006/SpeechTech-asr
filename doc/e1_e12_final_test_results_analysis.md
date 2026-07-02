# E1–E12 最终 test-clean 结果分析

> 数据来源：`results/metrics.csv`、`results/test_metrics.csv`、
> `results/representation_test_metrics.csv`、`results/predictions/*.jsonl`、
> K-means summary 和正式训练日志。  
> 分析日期：2026-07-02。  
> E7 按实验计划跳过，因此编号不连续。

---

## 1. 数据和结果完整性

### 1.1 评估数据

最终 test-clean 评估覆盖：

```text
utterances: 2620
audio:      5.403 h
words:      52,576
```

共检查16份完整 test prediction。所有文件的 utterance ID、reference、duration
和顺序完全一致，因此可以进行配对比较和 utterance-level bootstrap。

训练数据的实验名称沿用最初的 1h/3h/10h 命名，但在15秒过滤后，实际进入
训练的数据为：

| 实验组 | 有效训练时长 |
|---|---:|
| E3 / E12 | 1.001 h |
| E5 | 3.001 h |
| E1 / E2 / E4 / E6 / E8 / E9 / E10 / E11 | 6.392 h |

论文和图表应报告 effective hours，不能把 E2 等系统直接写成完整 10h。

### 1.2 指标口径

- WER/CER 均由保存的最终 prediction 重新计算。
- E2 dev prediction 已去除 Trainer padding 产生的伪末尾 apostrophe。
- test-clean 不参与模型、隐藏层或 codebook size 的选择。
- E10 的最佳隐藏层和 E11 的输入层只根据 dev WER 选择。
- RTF 为一次完整顺序测试中的模型前向时间除以音频时长，不包含音频读取。

---

## 2. 主结果

### 2.1 E1–E9 与 E12

| ID | 系统 | Trainable Params | Test WER | Test CER | 完全正确句 |
|---|---|---:|---:|---:|---:|
| E1 | Frozen Wav2Vec2 + linear CTC | 0.023 M | 99.92% | 85.46% | 0 |
| E2 | Full fine-tuned Wav2Vec2 | 90.194 M | **11.44%** | 3.36% | 597 |
| E3 | 1h fine-tuned Wav2Vec2 | 90.194 M | 40.51% | 11.65% | 12 |
| E4 | Full fine-tuned WavLM | 90.205 M | 17.50% | 5.08% | 291 |
| E5 | 3h fine-tuned Wav2Vec2 | 90.194 M | 17.12% | 4.88% | 319 |
| E6a | Wav2Vec2 top-6 fine-tuning | 47.667 M | 12.14% | 3.42% | 545 |
| E6b | Wav2Vec2 top-3 fine-tuning | 26.403 M | 22.41% | 6.03% | 130 |
| E8 | Full fine-tuning + feature masking | 90.194 M | **11.35%** | **3.34%** | 608 |
| E9 | Frozen Wav2Vec2 + BiLSTM-CTC | 3.694 M | 100.00% | 99.85% | 0 |
| E12a | 1h + time masking | 90.194 M | 40.73% | 11.67% | 22 |

E8 的点估计略优于 E2，但差距非常小；统计分析见第6节。当前最稳妥的主模型
仍应写为 E2/E8 同一性能水平，而不是宣称 E8 明显超过 E2。

### 2.2 E10：冻结连续表征层

| 系统 | Hidden Layer | Trainable Params | Test WER | Test CER | 完全正确句 |
|---|---:|---:|---:|---:|---:|
| E10a | 6 | 3.695 M | 30.03% | 9.37% | 72 |
| E10b | 9 | 3.695 M | **20.51%** | **6.24%** | 183 |
| E10c | 12 | 3.695 M | 46.18% | 14.44% | 3 |

Layer 9 明显优于 layer 6 和最终 layer 12，说明冻结 SSL 表征的可用性并不随
网络深度单调增加。选择最后一层作为通用表示会严重低估 Wav2Vec2 frozen
representation 的能力。

### 2.3 E11：离散语音单元

E11 使用 E10 在 dev 上选出的 layer 9，并保持相同 BiLSTM-CTC head。

| 系统 | K | Test WER | Test CER | Unit Rate | Bitrate | Dedup Bitrate |
|---|---:|---:|---:|---:|---:|---:|
| E11a | 50 | 71.56% | 30.91% | 49.91/s | 274.93 bit/s | 143.51 bit/s |
| E11b | 100 | 61.65% | 23.87% | 49.91/s | 324.43 bit/s | 187.22 bit/s |
| E11c | 200 | **52.60%** | **18.64%** | 49.91/s | 374.31 bit/s | 236.31 bit/s |

对应 test codebook 统计：

| K | Used Units | Utilization | Perplexity |
|---:|---:|---:|---:|
| 50 | 50 | 100% | 45.51 |
| 100 | 99 | 99% | 90.49 |
| 200 | 200 | 100% | 180.89 |

Codebook 使用充分且分布相对均衡，离散系统的性能下降不是 codebook collapse。

---

## 3. 标注数据规模

Wav2Vec2 full fine-tuning 的数据规模曲线为：

```text
1.001 h  E3: 40.51% WER
3.001 h  E5: 17.12% WER
6.392 h  E2: 11.44% WER
```

相对 WER 降幅：

| 数据增量 | Relative WER Reduction |
|---|---:|
| 1h → 3h | 57.74% |
| 3h → 6.392h | 33.19% |
| 1h → 6.392h | 71.76% |

结论：

1. 从1h增加到3h带来的收益最大，说明极低资源区间对新增标注非常敏感。
2. 从3h增加到6.392h仍有明显收益，但边际收益开始减小。
3. E3 已不再是早期的 blank collapse，因而这条曲线可以作为有效的数据规模
   消融。

---

## 4. 参数更新范围与下游容量

### 4.1 Linear frozen baseline

E1 只训练约23K个参数，test WER 为99.92%。这说明最终层 frozen feature 加
linear CTC 在当前配置下几乎无法识别。

该结果不能被泛化为“冻结 SSL encoder 无效”，因为 E10 使用相同冻结 encoder
但更强 BiLSTM head，并从合适层提取表示后达到20.51% WER。

### 4.2 Partial fine-tuning

```text
E2 full fine-tune:  90.194 M trainable, 11.44% WER
E6a top 6 layers:   47.667 M trainable, 12.14% WER
E6b top 3 layers:   26.403 M trainable, 22.41% WER
```

E6a 使用 E2 约52.85%的可训练参数，WER 仅高0.70个百分点。配对 bootstrap
95% CI 为 `[+0.465, +0.943]` 个百分点：差距真实存在，但幅度较小。

E6b 使用 E2 约29.27%的可训练参数，性能下降到22.41%，说明只更新最高3层
不足以接近 full fine-tuning。

### 4.3 Frozen layer 9 + stronger head

E10 layer 9 只训练3.695M参数，即 E2 的约4.10%，得到20.51% WER。它甚至
优于训练26.403M参数的 E6b：

```text
E10 layer 9: 20.51%
E6b top 3:   22.41%
difference: -1.90 percentage points
```

配对 bootstrap 95% CI 为 `[-2.445, -1.357]` 个百分点。该差距不能由当前
测试集抽样波动解释。

这表明在极低可训练参数预算下，冻结中间层表示配合更强序列 head，比只微调
少数顶部 Transformer 层更有效。

### 4.4 E9 failure

E9 的2620条 test prediction 全部为单个 apostrophe，属于确定的输出塌缩，
不能作为正常性能点。

E10 layer 12 在统一缓存训练管线下得到46.18% WER，因此 E9 的失败不能简单
归因于最终层完全没有信息，更可能与在线 frozen encoder、dropout、batch
配置或优化过程有关。

---

## 5. SSL encoder 对比

相同6.392h有效训练数据和 full fine-tuning 设置下：

| Encoder | Test WER | Test CER | RTF |
|---|---:|---:|---:|
| Wav2Vec2（E2） | **11.44%** | **3.36%** | 0.00403 |
| WavLM（E4） | 17.50% | 5.08% | 0.00886 |

WavLM 的 WER 比 Wav2Vec2 高约52.98%，CER 也明显更高。在当前字符 CTC、
数据量和优化配置下，Wav2Vec2 是更合适的 encoder。

RTF 方向上 WavLM 也更慢，但 E2/E4 是本地 RTX 4060 测量，不能与后来在
远程服务器测得的 E5–E12 RTF 直接比较。

---

## 6. Masking regularization

### 6.1 E8 vs E2

```text
E2: 11.4387% WER
E8: 11.3531% WER
difference: -0.0856 percentage points
```

使用10,000次 utterance-level paired bootstrap，E8−E2 的95% CI 为：

```text
[-0.313, +0.137] percentage points
```

区间包含0，因此当前单随机种子结果不能证明 feature masking 带来了稳定提升。
可以写为“性能基本持平，点估计略优”，不能写成“显著改善”。

错误计数也体现了混合变化：

| 系统 | Substitution | Deletion | Insertion |
|---|---:|---:|---:|
| E2 | 5,319 | 333 | 362 |
| E8 | 5,262 | 314 | 393 |

E8 减少 substitution/deletion，但 insertion 略有增加。

### 6.2 E12a vs E3

```text
E3:   40.5071% WER
E12a: 40.7334% WER
difference: +0.2263 percentage points
```

E12a−E3 的 paired-bootstrap 95% CI 为：

```text
[-0.053, +0.502] percentage points
```

区间同样包含0。没有证据表明 time masking 改善了1h低资源设置，也没有足够
证据断言它造成稳定退化。更强的 E12b masking 没有继续运行的必要。

---

## 7. 表征层分析

E10 的结果显示明显的倒 U 型层级趋势：

```text
layer 6  → 30.03%
layer 9  → 20.51%
layer 12 → 46.18%
```

可能解释：

1. 中间层保留较丰富的音系与局部内容信息。
2. 最终层更贴近预训练目标，并不一定最适合 frozen CTC recognition。
3. Layer 6 的抽象程度不足，BiLSTM head 仍需承担较多识别映射。

该实验直接支持“不同自监督表征层会带来明显差异”。最大差距超过25个百分点，
远大于 E8/E12 的正则化差异。

---

## 8. 连续表示与离散单元

### 8.1 性能—容量趋势

固定 layer 9 和 downstream head 后，增加 codebook size 带来单调改善：

```text
K=50:  71.56%
K=100: 61.65%
K=200: 52.60%
continuous: 20.51%
```

相对改善：

| 变化 | Relative WER Reduction |
|---|---:|
| K=50 → K=100 | 13.85% |
| K=100 → K=200 | 14.68% |

即使 K=200，WER 仍是连续 layer 9 的约2.56倍，说明当前 K-means 离散化
丢失了大量识别相关细节。

### 8.2 压缩效率

连续 layer 9 以 FP16、768维、约49.91 Hz 保存时，理论信息率为：

```text
613,344 bit/s
```

离散表示相对连续表示的压缩倍数：

| K | Empirical Bitrate | Compression Ratio |
|---:|---:|---:|
| 50 | 274.93 bit/s | 2,231× |
| 100 | 324.43 bit/s | 1,891× |
| 200 | 374.31 bit/s | 1,639× |

因此 E11 形成了非常清晰的 trade-off：

```text
更大的 K
→ 更高 bitrate
→ 更低 WER
→ 但仍明显弱于连续表示
```

这里的 bitrate 衡量中间表征的存储或传输成本，不代表完整系统推理成本。
E11 仍需运行完整 Wav2Vec2 encoder 产生隐藏表示，因此量化不会自动减少
encoder 计算量。

---

## 9. 推理效率与模型复杂度

远程同一轮测试中：

| 系统类别 | 代表系统 | RTF |
|---|---|---:|
| Wav2Vec2 + linear CTC | E6a/E6b/E8/E12 | 0.00122–0.00125 |
| Frozen Wav2Vec2 + online BiLSTM | E9 | 0.00287 |
| Continuous cached-head pipeline | E10 | 0.00359–0.00366 |
| Discrete quantization pipeline | E11 | 0.00370–0.00378 |

结论：

1. 冻结参数或减少 trainable layers 主要降低训练成本，不会跳过 encoder 层，
   因此不会显著降低推理计算。
2. BiLSTM head 增加了明显推理开销。
3. K-means 最近中心分配只带来小幅额外开销；E11 相比 E10 的主要代价仍是
   识别精度，而不是 RTF。
4. E5 与其他标准 Wav2Vec2 模型结构相同但首次测量较慢，说明单次 RTF 会受
   CUDA/cuDNN 冷启动影响，不应把其差异解释为模型结构差异。

参数量也需要区分 trainable parameters 和 total system parameters：

```text
E2 total:      94.395 M
E10/E11 total: 98.066 M
```

E10/E11 训练时只更新3.695M参数，但额外 BiLSTM 使完整部署模型反而略大于
E2。它们属于 parameter-efficient adaptation，不属于 model-size compression。

---

## 10. 错误类型

所有系统均以 substitution 为主要错误来源。

| 系统 | Substitution | Deletion | Insertion | Exact |
|---|---:|---:|---:|---:|
| E2 | 5,319 | 333 | 362 | 597 |
| E6a | 5,708 | 304 | 371 | 545 |
| E6b | 10,768 | 381 | 635 | 130 |
| E8 | 5,262 | 314 | 393 | 608 |
| E10 layer 9 | 9,580 | 491 | 712 | 183 |
| E11 K=200 | 23,227 | 2,291 | 2,136 | 5 |
| E12a | 19,851 | 671 | 894 | 22 |

离散量化不仅增加 substitution，也显著增加 deletion 和 insertion，说明损失
并非局限于少量相近音素混淆，而是影响了更广泛的序列建模和 CTC 对齐。

---

## 11. Dev–test 一致性

主要系统的 dev/test 方向一致：

| 系统 | Dev WER | Test WER | Test−Dev |
|---|---:|---:|---:|
| E2 | 11.09% | 11.44% | +0.35 pp |
| E5 | 16.69% | 17.12% | +0.43 pp |
| E6a | 11.92% | 12.14% | +0.22 pp |
| E6b | 21.95% | 22.41% | +0.46 pp |
| E8 | 10.98% | 11.35% | +0.37 pp |
| E10 layer 9 | 19.41% | 20.51% | +1.10 pp |
| E11 K=200 | 52.31% | 52.60% | +0.29 pp |
| E12a | 40.36% | 40.73% | +0.38 pp |

除 E10 layer 9 的 gap 稍大外，没有明显的 dev/test 反转或严重过拟合迹象。

---

## 12. 局限与结果使用边界

### 12.1 单随机种子

所有系统主要使用 seed 42。E8/E2、E12/E3 这类小于0.3个百分点的差距不能
解释为稳定算法收益。

### 12.2 E10/E11 的32标签配置

缓存训练器构造 tokenizer 时保留了默认 `<s>` 和 `</s>`，因此 E10/E11
输出层为32类，而正式字符词表为30类。检查全部 E10/E11 dev/test prediction
后，没有发现解码出的 `<s>` 或 `</s>`，但两个未监督 logits 仍参与 softmax。

该问题只增加1,026个 head 参数，相比3.695M参数很小；隐藏层和 K 的差距远大
于这一实现差异，因此主要趋势仍可分析。但严格的30类公平复现应修正 tokenizer
后重跑，报告中需要披露这一点。

### 12.3 E9 是失败实验

E9 为确定的 apostrophe-output collapse。它适合用于分析训练稳定性，不能
当作 frozen BiLSTM 架构的正常性能上限。

### 12.4 RTF 硬件与冷启动

E1–E4 的 RTF 来自本地 RTX 4060；E5–E12 来自远程 GPU。不同硬件之间不能
直接比较绝对 RTF。单次测量还包含首批 CUDA/cuDNN 初始化影响。

### 12.5 离散压缩不等于端到端压缩

E11 的 bitrate 描述 representation storage，而完整推理仍依赖 Wav2Vec2。
若要降低端到端模型大小或算力，还需要离线单元、蒸馏或轻量 acoustic tokenizer。

---

## 13. 最终结论

当前实验能够支持以下核心结论：

1. **标注数据规模是最主要因素之一。** 1h→3h 的收益最大，之后呈现边际收益
   递减。
2. **不必微调整个 encoder 才能接近最佳性能。** E6a 用约53%的可训练参数，
   WER 只比 E2 高0.70个百分点。
3. **只训练极少顶部层会明显损失性能。** E6b 无法接近 E2/E6a。
4. **冻结表示的层选择非常关键。** Layer 9 明显优于 layer 6 和 layer 12。
5. **强 downstream head 可以有效利用冻结中间层。** E10 layer 9 仅训练
   3.695M参数即达到20.51% WER。
6. **K-means 离散单元提供极高压缩率，但识别损失明显。** 增大 K 能稳定改善
   WER，形成清晰的 bitrate–accuracy trade-off。
7. **Masking 没有显示稳定收益。** E8 与 E2 基本持平，E12 与 E3 也基本持平。
8. **当前最佳可靠系统为 E2/E8。** 二者差距不显著，应避免过度强调 E8 的
   微小点估计优势。

建议论文主结果采用 E2 作为稳定基线，同时用 E8 说明 masking 基本保持性能；
参数效率部分突出 E6a 与 E10 layer 9；离散表示部分报告 K=50/100/200 的完整
WER–bitrate 曲线，而不是只报告最佳 K=200。
