# E1–E4 初步结果分析

> 分析日期：2026-07-01  
> 数据来源：`results/metrics.csv`、`results/predictions/*.jsonl`、实验配置与任务完成记录。  
> 分析边界：远程完整 checkpoint、完整训练日志、test-clean 预测、RTF 和多数训练耗时尚未同步，因此本文只分析当前本地可复核的 dev 预测结果。

## 1. 结果完整性检查

四个实验均已有 2513 条 dev 预测，utterance ID、顺序、reference 和 duration 完全一致，可以做配对比较。

但当前评估集不是完整的 LibriSpeech dev-clean：

| 范围 | Utterances | 时长 | Reference words |
|---|---:|---:|---:|
| 完整 `dev_clean.jsonl` | 2703 | 5.388 h | 54402 |
| 实际评估子集（音频不超过 15 秒） | 2513 | 4.364 h | 44349 |
| 被过滤的长音频 | 190 | 1.024 h | 10053 |

训练集也应用了相同的 15 秒上限。因此实验名中的 1h/10h 是 manifest 的名义规模，实际参与训练的数据约为：

| Manifest | 原始规模 | 实际训练规模 |
|---|---:|---:|
| train-1h | 278 条 / 1.000 h | 184 条 / 0.591 h |
| train-10h | 2841 条 / 10.000 h | 2012 条 / 6.392 h |

这不影响 E1/E2/E4 之间的同集比较，但论文中应写明实际训练和评估范围，也不应把当前数字直接当作标准完整 dev-clean 结果与外部工作比较。

## 2. 主结果

| ID | 系统 | 名义数据 | 实际数据 | WER ↓ | CER ↓ | 完全正确句 |
|---|---|---:|---:|---:|---:|---:|
| E1 | Frozen Wav2Vec2 + CTC | 10h | 6.392 h | 100.00% | 99.60% | 0 / 2513 |
| E2（文件原值） | Fine-tuned Wav2Vec2 + CTC | 10h | 6.392 h | 15.70% | 4.35% | 1 / 2513 |
| E2（去除已知 padding 伪字符） | Fine-tuned Wav2Vec2 + CTC | 10h | 6.392 h | **11.09%** | **3.30%** | **606 / 2513** |
| E3 | Fine-tuned Wav2Vec2 + CTC | 1h | 0.591 h | 100.00% | 99.84% | 0 / 2513 |
| E4 | Fine-tuned WavLM + CTC | 10h | 6.392 h | 16.60% | 4.86% | 322 / 2513 |

E2 的两行不是两次实验。当前 `wav2vec2_finetune_10h_dev.jsonl` 中 2509 条预测带有 Trainer 跨 batch 补齐产生的伪末尾 apostrophe；`metrics.csv` 的 15.70%/4.35% 是含该伪字符的结果。按仓库 `repair_ctc_prediction_padding.py` 的既有规则诊断性去除后，复算结果为 11.09%/3.30%。在正式汇总前应修复 prediction 和 `metrics.csv`，当前分析不直接改写原始结果文件。

此外，`metrics.csv` 当前缺少 E1 行；E1 数字由现有 prediction 重新计算，与任务记录中的 WER 1.0000、CER 0.99595 一致。

## 3. 初步结论

### 3.1 Fine-tuning 是当前最明确的有效因素

E1 的 encoder 完全冻结，仅训练 24608 个 CTC head 参数，预测基本退化为 apostrophe/blank；E2 则训练约 90.19M 参数。修正解码伪字符后，E2 相对 E1 的 WER 从 100% 降到 11.09%，说明仅靠当前 frozen SSL 表征加线性 CTC head 无法完成任务，而端到端适配带来了决定性收益。

这个结论同时混合了“可训练参数量”和“优化难度”的差异，不能解释为 Wav2Vec2 frozen representation 本身无信息。若希望得到更强的 frozen baseline，需要增加 head 训练步数，或使用更有表达力的 downstream head。

### 3.2 当前 E3 不能支撑有效的 1h vs 10h 数据规模结论

E3 的 2513 条预测中，2509 条为单个 apostrophe、4 条为空；去除 padding 伪字符后全部为空，说明模型发生 blank-output collapse。这个结果只能说明“当前 0.591h 有效数据与训练配置没有得到可用模型”，不能把 100% WER 直接解释为减少训练数据的正常性能退化。

E3 应优先检查远程训练曲线和所选 checkpoint。若中间 checkpoint 曾出现非空输出，应按 dev WER 重新选取；否则需要调整学习率、warmup、总步数或正则化后重跑。完成该诊断前，数据规模消融仍不成立。

### 3.3 当前 Wav2Vec2 优于 WavLM，但需先统一指标清洗

按修正后的可比结果，E2 的 WER/CER 为 11.09%/3.30%，E4 为 16.60%/4.86%；E2 分别相对低约 33.2% 和 32.1%。配对观察中，E2 正确而 E4 错误的句子有 371 条，反向只有 87 条，方向与总体指标一致。

这只能支持“在本项目当前配置下 Wav2Vec2 表现更好”，不能推导 Wav2Vec2 普遍优于 WavLM。两者都只有单次 seed，且尚未核对最佳 checkpoint 选择和完整训练曲线。

## 4. 初步错误分析

修正后的 E2 word-level 错误计数为 4322 substitutions、300 deletions、297 insertions；E4 为 6430 substitutions、493 deletions、439 insertions。两者都以 substitution 为主，E4 在三类错误上均更多。

常见混淆包括：

- 功能词/弱读词：`in → and`、`and → in`、`a ↔ the`。
- 拼写和音形近似：`burgess → burges`、`pepper → peper`、`thee → the`。
- 专有名词：如 `phoebe → feby`、`randal → randel`。
- 插入和删除也主要集中在 `a`、`the`、`in`、`to`、`it` 等短功能词。

按时长分桶的聚合结果如下：

| 时长 | 样本数 | E2 WER | E2 CER | E4 WER | E4 CER |
|---|---:|---:|---:|---:|---:|
| ≤ 5 秒 | 1118 | 12.42% | 3.80% | 18.46% | 5.74% |
| 5–10 秒 | 1018 | 10.54% | 3.13% | 15.83% | 4.60% |
| 10–15 秒 | 377 | 10.92% | 3.19% | 16.36% | 4.61% |

当前没有观察到 10–15 秒区间随时长明显恶化；短句的聚合 WER 反而略高，可能是短 utterance 中单个词错误占比更大。由于超过 15 秒的 190 条样本完全未评估，不能据此评价真正长句性能。

## 5. 当前缺口与下一步优先级

1. 用现有修复脚本清理 E2 prediction，并重建包含 E1–E4 的唯一正式 `metrics.csv`。
2. 从远程同步 E2/E3/E4 的 `trainer_state.json`、最佳 checkpoint 信息和训练日志；先诊断 E3 collapse，再决定是否重跑。
3. 明确报告“名义数据时长”和“15 秒过滤后的有效时长”，或补跑完整 dev-clean 评估。
4. 对最佳模型补齐 test-clean WER/CER；dev 结果用于选模，test 结果用于最终报告。
5. 补齐训练耗时、GPU、RTF、总参数和可训练参数，完成项目计划中的成本表。
6. 修正 prediction 中远程绝对 `audio_path`，避免后续错误分析脚本在本地无法读取音频。

当前可用于报告的核心结论是：10h 配置下 fine-tuned Wav2Vec2 已得到可用且最佳的 dev 结果；frozen baseline 无法学习，WavLM 次于 Wav2Vec2；1h 实验发生输出塌缩，必须诊断或重跑后才能完成数据规模消融。
