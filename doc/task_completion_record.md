# 任务完成记录

## 2026-07-01：Startup 基础代码

状态：基础代码与全量本地数据准备已完成；CTC 实际训练待安装深度学习依赖。

### 已完成

- 建立 `src/ssl_asr`、`scripts`、`tests`、`results`/`exp` 按需生成的基础工程结构。
- 实现英文 ASR 文本规范化：小写化、删除标点、保留 apostrophe、压缩空格。
- 实现 LibriSpeech local-first 数据发现：
  - 优先读取 `data/train-clean-100`、`data/dev-clean`、`data/test-clean`。
  - 仅在对应本地 split 不存在时，回退到 Hugging Face
    `openslr/librispeech_asr` 下载并写入 `data/<split>`。
- 实现无需额外音频包的 FLAC STREAMINFO 时长读取。
- 实现固定 `seed=42`、按累计时长构造嵌套的 1h/10h 训练子集。
- 实现 JSONL manifest 与字符级 CTC vocabulary 生成。
- 实现最小 CTC 训练入口：
  - 支持 `facebook/wav2vec2-base` 等 `AutoModelForCTC` 模型。
  - 支持完整 encoder freeze baseline，以及仅冻结 feature encoder 的 fine-tune。
  - 支持样本数/音频时长限制、fp16、gradient accumulation 和 gradient checkpointing。
  - 训练后输出 dev prediction JSONL、WER/CER、参数量和 `results/metrics.csv`。
- 补充安装、数据准备和 debug 训练说明。

### 已验证

```text
python -m py_compile ...
结果：所有新增 Python 文件通过语法检查。

python scripts/prepare_librispeech_subsets.py --splits dev-clean test-clean --local_only
结果：
  dev-clean  = 2703 utterances, 5.39h
  test-clean = 2620 utterances, 5.40h
  data/manifests/dev_clean.jsonl 生成成功
  data/manifests/test_clean.jsonl 生成成功
  data/vocab/vocab.json 生成成功（30 tokens）

python scripts/prepare_librispeech_subsets.py --local_only
结果：
  train-clean-100 = 28539 utterances, 100.59h
  train_1h.jsonl  = 278 utterances, 1.000h
  train_10h.jsonl = 2841 utterances, 10.000h
  全程使用 data/ 下的本地 split，未触发 Hugging Face 下载

python -m pytest -q
结果：2 passed
```

### 当前未完成/受环境限制

- 当前 Python 环境缺少 `torch`、`torchaudio`、`datasets`、`jiwer` 和
  `soundfile`，因此尚未执行真实 Wav2Vec2 CTC debug 训练。
- 当前已安装的 `transformers==5.0.0` 超出本项目锁定范围；执行
  `pip install -r requirements.txt` 会安装兼容的 `transformers>=4.40,<5`。

### 下一步命令

```powershell
pip install -r requirements.txt
python scripts/train_ctc.py `
  --train_manifest data/manifests/train_1h.jsonl `
  --eval_manifest data/manifests/dev_clean.jsonl `
  --output_dir exp/debug_wav2vec2 `
  --max_train_samples 200 `
  --max_eval_samples 100 `
  --num_train_epochs 1 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8
```

## 2026-07-01：Debug 启动无输出排查

- 确认 `ssl_asr` 环境可用：`torch==2.6.0+cu126`，CUDA 可用，
  `transformers==4.57.6`，`accelerate==1.14.0`。
- 复现到 `Trainer` 导入后进程以 `-1073741819` 退出，即 Windows
  `0xC0000005` 原生访问冲突，所以不会产生 Python traceback。
- 拆分依赖后确认 `datasets`、`pyarrow` 和 `Trainer` 单独导入均正常；当前环境中
  PyTorch 先于 `pyarrow.dataset` 初始化时会触发 DLL 加载顺序冲突。
- 已在训练脚本中显式先初始化 `pyarrow.dataset`，验证随后导入 `Trainer`
  可正常返回（exit code 0）。
- 已在依赖导入、checkpoint 加载、数据载入和训练开始前增加即时刷新的启动日志。
- 当前 Codex 终端默认处于 `base` 环境；执行训练前必须激活 `ssl_asr`，
  或直接使用该环境的 Python。

## 2026-07-01：E1 Frozen 10h baseline

状态：已完成。

- 新增 `configs/wav2vec2_frozen_10h.yaml` 和
  `scripts/train_wav2vec2_frozen_10h.ps1`。
- 使用 8GB GPU 保守设置：batch size 1、gradient accumulation 16、
  FP16、最大音频 15 秒、3 epochs。
- 冻结检查结果：总参数 94,396,320，可训练参数 24,608（0.03%），
  仅 CTC projection head 可训练。
- 第一次运行完成 epoch 1 训练，但全量 dev 评估在 2472/2513 时 OOM。
  原因是评估预测持续驻留 GPU，而非单个 batch 超限。
- 已增加 `eval_accumulation_steps=10`，使评估预测定期转移至 CPU；
  E1 使用修正配置重新运行成功。
- 最终结果：
  - 有效训练样本：2012（10h manifest 中时长不超过 15 秒的样本）。
  - dev 样本：2513。
  - 训练时长：1383.57 秒。
  - train loss：3.4773。
  - dev WER：1.0000。
  - dev CER：0.99595。
  - checkpoint：`checkpoint-252`、`checkpoint-378`。
  - prediction：`results/predictions/wav2vec2_frozen_10h_dev.jsonl`。
  - 指标已追加到 `results/metrics.csv`。

## 2026-07-01：CTC tokenizer/model vocabulary 一致性修复

状态：已完成。

- 修复远程 E1 训练中出现的
  `ValueError: Label values must be <= vocab_size`。
- 模型输出维度不再使用可能低估 ID 空间的 `len(tokenizer)`，改为
  `max(tokenizer.get_vocab().values()) + 1`。
- transcript 编码显式设置 `add_special_tokens=False`。
- 增加两层标签范围检查：
  - 训练开始前扫描 train/dev 的全部 transcript。
  - Dataset 读取每条样本时再次检查，并在异常中报告 utterance ID。
- 启动日志现在输出 tokenizer entries、最大 token ID、模型 vocab size 和
  added vocabulary，便于排查跨主机词表差异。
- 回归测试：`python -m pytest -q`，结果 `5 passed`。
- 真实 Trainer 冒烟测试：2 条 train、2 条 dev、1 optimizer step，训练、
  评估和 prediction 导出全部成功。

## 2026-07-01：E2 Wav2Vec2 fine-tune 10h

状态：远程正式训练与 dev prediction 已完成；当前本地 E2 指标需再次清理
已知 padding 伪字符，完整远程 checkpoint 尚未同步。

### 配置完成

- 新增 `configs/wav2vec2_finetune_10h.yaml` 和
  `scripts/train_wav2vec2_finetune_10h.ps1`。
- 模型固定为自监督预训练 checkpoint `facebook/wav2vec2-base`，未使用
  已监督微调的 `facebook/wav2vec2-base-960h`。
- 使用 `train_10h.jsonl` 训练、`dev_clean.jsonl` 验证。
- 仅冻结底层 CNN feature encoder；Transformer encoder 与 CTC head 可训练。
- 使用 8GB GPU 保守配置：batch size 1、gradient accumulation 16、FP16、
  gradient checkpointing、最大音频 15 秒、eval accumulation 10、
  learning rate 3e-4、3 epochs、最多保留 2 个 checkpoint。
- 正式启动脚本会拒绝覆盖已有的 `exp/wav2vec2_finetune_10h`。

### 启动验证

- 使用 16 条训练样本、8 条 dev 样本完成 1 epoch 的真实 CUDA
  前向、反向、优化、评估、预测保存与指标写入。
- 验证保持正式实验的关键显存配置：batch size 1、gradient accumulation
  16、FP16、gradient checkpointing、最大音频 15 秒和 eval accumulation 10。
- 参数检查结果：总参数 94,396,320，可训练参数 90,195,872（95.55%）。
  对比 E1 仅 24,608 个参数可训练，确认 E2 不是 frozen baseline。
- 启动验证未出现 OOM 或 NaN；train loss 8.4875，eval loss 6.9248。
- 验证产物与正式结果隔离：
  - `exp/wav2vec2_finetune_10h_startup_validation/`
  - `results/predictions/wav2vec2_finetune_10h_startup_validation_dev.jsonl`
  - `results/wav2vec2_finetune_10h_startup_validation_metrics.csv`
- 训练入口增加结构级断言：CNN feature encoder 必须全部冻结；E2 的
  Transformer encoder 必须存在可训练参数；冻结状态错误时立即终止。

### 正式训练

首次 3-epoch 正式训练已完成但发生输出塌缩，结果已归档为
`wav2vec2_finetune_10h_failed_3ep`。修复后的正式训练已在远程服务器完成，
dev prediction 已导入本地；当前同步文件重新带入已知 padding 伪字符，
指标需再次修复。完整远程 checkpoint 尚未同步。

### 首次正式训练结果与修复

- 3-epoch 配置完成了 378 个优化步，但三个 epoch 的 dev WER 均为
  1.0，eval loss 约为 2.93；2513 条预测中 2511 条退化为单个
  apostrophe。该结果不能作为有效 E2 主结果。
- 已验证数据标签编码、CTC blank ID、CNN 冻结、Transformer 参数更新均正确，
  且没有 OOM 或 NaN。
- 严格 30-token 输出、warmup、较低学习率、更多小样本更新和 CTC sum
  reduction 的隔离实验表明，问题并非词表错位或单一 loss reduction 设置。
- 对照 Hugging Face 官方 Wav2Vec2 CTC recipe，原实验的 378 steps 严重不足；
  官方示例使用 30 epochs，并在约 step 1000 后才出现明显 WER 改善。
- 修正配置采用：30 epochs、learning rate 1e-4、warmup ratio 0.1、
  weight decay 0.005、mean CTC loss、CTC zero infinity；其余 8GB
  显存设置保持不变。
- 首次失败结果重命名为 `wav2vec2_finetune_10h_failed_3ep` 保留，不作为
  修复后正式 E2 指标。
- 修复后完成 20 条样本、1000 个优化步的过拟合验证。模型在约 step 420
  后脱离塌缩，最终 eval loss 0.08486、WER 0.08087、CER 0.01679，
  确认训练链路能够有效学习。
- 1000-step 验证产物保存在
  `exp/wav2vec2_finetune_overfit_validation_1000steps/` 和独立 metrics
  文件中，未写入正式 `results/metrics.csv`。

### 远程正式训练结果导入与评估修复

- 已导入远程正式 E2 dev prediction，共 2513 条，与15秒过滤后的
  `dev_clean.jsonl` ID、顺序完全一致，无缺失、重复或空预测。
- 发现 Trainer 跨 batch 拼接不同长度 logits 时用 `-100` 补齐；旧代码对
  全 `-100` 帧直接 argmax，错误产生 ID 0 apostrophe。2509 条预测因此
  带有人为末尾 apostrophe。
- 训练入口现将全 `-100` padded frames 映射为 CTC blank 后再解码，并增加
  单元测试覆盖该行为。
- 原始导入文件保留为
  `results/predictions/wav2vec2_finetune_10h_dev_raw_padding_bug.jsonl`。
- 曾完成伪尾字符清理与本地路径重定位；但随后同步 E2/E3/E4 远程结果时，
  当前正式 prediction 又被原始 E2 文件覆盖。现有两个 E2 prediction 文件
  内容相同，均含2509个伪尾字符和远程绝对路径。
- 当前 `results/metrics.csv` 中 E2 为未修正的 WER 0.15703、CER 0.04347；
  按既有修复规则诊断性复算仍为 WER 0.11092、CER 0.03301。正式汇总前
  需要再次运行修复脚本并更新 metrics。
- 本地 `exp/wav2vec2_finetune_10h/` 目前仅含 step 126/252 的未完成旧
  checkpoint（计划总步数3780），不视为远程正式 E2 模型；完整远程
  checkpoint 仍需另行同步。

## 2026-07-01：E3 Wav2Vec2 fine-tune 1h

状态：远程正式训练与 dev prediction 已完成并导入本地；模型发生
blank-output collapse，完整远程 checkpoint 与训练日志尚未同步。

### 配置完成

- 新增 `configs/wav2vec2_finetune_1h.yaml` 和
  `scripts/train_wav2vec2_finetune_1h.ps1`。
- 使用 `train_1h.jsonl` 训练和 `dev_clean.jsonl` 验证；模型、CNN 冻结策略、
  30-token CTC head 与修复后的 E2 一致。
- 15 秒过滤后训练集为 184 条，gradient accumulation 16 时每轮约 12
  个优化步。为达到已验证的学习区间，E3 固定训练 1000 optimizer steps，
  而不是沿用会导致欠训练的 30 epochs。
- 每 100 optimizer steps 执行一次完整 dev 评估和 checkpoint 保存，
  `save_total_limit=2`，避免按约 84 个 epoch 逐轮评估。
- 正式脚本拒绝覆盖已有的 `exp/wav2vec2_finetune_1h`。

### 启动验证

- 使用 16 条 train-1h 样本、8 条 dev 样本和 1 个 optimizer step
  完成真实 CUDA 前向、反向、steps-based 评估、checkpoint 与预测保存。
- 参数检查：总参数 94,394,782，可训练参数 90,194,334（95.55%）；
  CNN feature encoder 已冻结，Transformer encoder 可训练。
- 未发生 OOM、NaN 或冻结错误；正式 `results/metrics.csv` 未被验证结果污染。
- 隔离产物：
  - `exp/wav2vec2_finetune_1h_startup_validation/`
  - `results/predictions/wav2vec2_finetune_1h_startup_validation_dev.jsonl`
  - `results/wav2vec2_finetune_1h_startup_validation_metrics.csv`

### 正式训练

远程正式训练已完成，本地已有
`results/predictions/wav2vec2_finetune_1h_dev.jsonl`，并已向
`results/metrics.csv` 写入 E3 记录。

- 实际训练数据：184 条、约 0.591 小时（15 秒过滤后）。
- dev 评估数据：2513 条、约 4.364 小时。
- dev WER：1.0000。
- dev CER：0.99839。
- 2513 条预测中 2509 条为单个 apostrophe、4 条为空。
- 按已知 Trainer padding 伪字符规则去除 apostrophe 后，预测全部为空，
  表明 E3 发生 blank-output collapse。
- 当前结果不能作为有效的 1h vs 10h 数据规模消融结论；需同步远程
  trainer state、训练曲线和 checkpoint，检查中间 checkpoint 或调整配置重跑。

### E3 修复配置（待 GPU 重跑）

- 新增过滤后再累计时长的
  `data/manifests/train_1h_effective_15s.jsonl`，实际为 1.000h；旧
  `train_1h.jsonl` 和失败结果均保留。
- 修复实验使用独立目录 `exp/wav2vec2_finetune_1h_repaired`，不会覆盖原 E3。
- CTC head / encoder 学习率分别为 `1e-4` / `2e-5`；前 100 steps 暂停
  encoder 更新，有效 batch 从 16 降至 8。
- 显式关闭 time/feature masking，按 dev WER 恢复最佳 checkpoint，并保留
  5 个 checkpoint。
- 每次评估记录非空预测比例和帧级 blank 比例；连续 8 次评估的非空比例
  不超过 1% 时自动停止。训练结束后在训练子集上计算诊断 WER/CER。
- 已使用 16 条训练样本、8 条 dev 样本和 1 个 optimizer step 完成 RTX
  4060 启动验证。分层 optimizer、encoder 更新延迟、masking 关闭、最佳
  checkpoint 恢复、blank/non-empty 指标以及训练子集诊断均正常执行；产物
  隔离在 `exp/wav2vec2_finetune_1h_repaired_startup_validation/` 和独立
  metrics 文件中。

## 2026-07-01：E4 WavLM fine-tune 10h

状态：远程正式训练与 dev prediction 已完成并导入本地；完整远程
checkpoint 与训练日志尚未同步。

### 配置完成

- 按项目计划选择 `microsoft/wavlm-base`，新增
  `configs/wavlm_finetune_10h.yaml` 和
  `scripts/train_wavlm_finetune_10h.ps1`。
- 使用 `train_10h.jsonl` 训练、`dev_clean.jsonl` 验证，只冻结 CNN
  feature encoder，训练 WavLM Transformer encoder 和随机初始化 CTC head。
- 为保证 E2/E4 公平比较，除 SSL checkpoint 外均对齐修复后的 E2：
  15 秒时长限制、batch size 1、gradient accumulation 16、FP16、
  gradient checkpointing、30 epochs、learning rate 1e-4、warmup 0.1、
  weight decay 0.005、mean CTC loss 和 eval accumulation 10。
- 正式脚本拒绝覆盖已有 `exp/wavlm_finetune_10h`。

### 启动验证

本地未单独执行启动验证；远程正式训练已经完成，因此该待办不再阻塞
结果分析。

### 正式训练

远程正式训练已完成，本地已有
`results/predictions/wavlm_finetune_10h_dev.jsonl`，并已向
`results/metrics.csv` 写入 E4 记录。

- 实际训练数据：2012 条、约 6.392 小时（15 秒过滤后）。
- dev 评估数据：2513 条、约 4.364 小时。
- 总参数：94,405,006。
- 可训练参数：90,204,558。
- dev WER：0.16600。
- dev CER：0.04863。
- 完全正确句：322 / 2513。
- 当前 prediction 未发现 E2 所见的伪末尾 apostrophe。

## 2026-07-01：RTX 3090 服务器迁移脚本

- 新增 `scripts/train_e2_e4_rtx3090.sh`，用于在 Linux RTX 3090 24GB
  服务器上串行执行 E2、E3、E4。
- 初始 train/eval batch size 4、gradient accumulation 4 在远程 3090 Ti
  首个 cuDNN 卷积处触发 `CUDNN_STATUS_NOT_INITIALIZED`。保守配置修正为
  batch size 2、gradient accumulation 8、data loader workers 0；有效
  batch size 仍为 16，不改变实验定义。
- 远程服务器的独立最小 Conv1d cuDNN 预检同样触发
  `CUDNN_STATUS_NOT_INITIALIZED`，确认问题属于服务器 cuDNN runtime，
  与数据和模型无关。训练入口新增 `--disable_cudnn`；3090 脚本在 cuDNN
  预检失败时自动验证并切换到原生 CUDA FP16 Conv1d。
- E4 WavLM 在原生 CUDA Conv1d fallback 下使用 batch 2 时首个 batch
  峰值约22.9GiB并 OOM。E4 单独调整为 batch size 1、gradient
  accumulation 16，并启用 `expandable_segments:True`；有效 batch、
  15秒时长上限、模型和训练目标均保持不变。
- cuDNN 环境修复后，WavLM 在 batch 1 的首个 backward 仍占用约22.46GiB。
  堆栈显示旧式 reentrant checkpoint 在12层 Transformer 中递归调用
  `CheckpointFunction.backward`。训练入口改用 PyTorch 推荐的
  non-reentrant checkpoint（`use_reentrant=False`），避免嵌套反向图；
  不改变模型、数据或有效 batch。
- 支持从 `e2`、`e3` 或 `e4` 开始，便于中断后继续后续实验。
- E2/E3 完成后，默认起点已改为 `e4`；仍可显式传入 `e2` 或 `e3`
  进行重跑。
- 每项实验写入独立 `logs/*_rtx3090.log`，并拒绝覆盖已有 experiment
  目录或 prediction 文件。
- 已通过 Bash 静态语法检查；未在本机占用中的 GPU 上启动服务器配置训练。

## 2026-07-01：E1–E4 结果初步分析与完整性审计

状态：已完成现有本地结果的初步复核，分析记录见
`doc/e1_e4_preliminary_results_analysis.md`。

- E1–E4 均有2513条 dev prediction，四组 ID、顺序、reference 和 duration
  完全一致，可进行配对比较。
- 实际评估的是15秒以内的 dev-clean 子集：2513条、4.364小时；完整
  dev-clean manifest 为2703条、5.388小时，另有190条长音频未评估。
- 15秒过滤后的实际训练量为：
  - E1/E2/E4：2012条、6.392小时，而非完整10小时。
  - E3：184条、0.591小时，而非完整1小时。
- 当前可复核结果：
  - E1：WER 1.0000，CER 0.99595。
  - E2：文件原值 WER 0.15703、CER 0.04347；去除已知 padding
    伪字符后为 WER 0.11092、CER 0.03301。
  - E3：WER 1.0000、CER 0.99839，确认 blank-output collapse。
  - E4：WER 0.16600、CER 0.04863。
- `results/metrics.csv` 当前缺少 E1，且 E2 行仍是未清理值；尚不能直接作为
  最终论文主结果表。
- 初步结论：E2 是当前最佳系统，E4 次之；E1 frozen baseline 无有效输出；
  E3 需要检查训练曲线、checkpoint 或重跑，暂不能形成有效数据规模消融。
- 尚缺完整远程 checkpoint/日志、test-clean 指标、训练耗时和 RTF。
