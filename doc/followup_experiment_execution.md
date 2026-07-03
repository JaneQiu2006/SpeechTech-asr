# 后续实验执行说明

本轮代码将综合分析第 14 节的建议落实为以下新实验，全部使用独立目录，不覆盖
既有 E1–E23 结果。

| ID | 实验 | 目的 |
|---|---|---|
| E24A | Layer 8 continuous + 2-layer BiLSTM，30 类 | 复跑最佳连续表示 |
| E24B | Layer 8, K=200 centroid，30 类 | centroid 严格对照 |
| E24C | Layer 8, K=200 embedding-256，30 类 | 复现原 E18 embedding 设置 |
| E24D | Layer 8, K=1000 centroid，30 类 | 复跑当前最佳离散配置 |
| E24E | Layer 8, K=200 embedding-768，30 类 | 与 centroid 做输入维度控制 |
| E25 | Top-5 partial fine-tuning, seed=1234 | 与 E16B seed=42 比较稳定性 |

E24A–D 显式禁用 tokenizer 的默认 BOS/EOS，保证输出层严格等于
`data/vocab/vocab.json` 的 30 个 ID。评测器默认使用 `auto`，可同时正确加载
旧 32 类模型和新 30 类模型。

## 远程 Linux 一次性执行

在仓库根目录运行：

```bash
conda activate ssl_asr
bash scripts/run_all_followup_experiments_rtx3090.sh
```

若服务器 cuDNN 仍无法初始化：

```bash
DISABLE_CUDNN=1 bash scripts/run_all_followup_experiments_rtx3090.sh
```

脚本可安全重跑，已完成实验会跳过，部分缓存会恢复。指定起始阶段：

```bash
bash scripts/run_all_followup_experiments_rtx3090.sh --from seed
bash scripts/run_all_followup_experiments_rtx3090.sh --from analysis
```

只有明确需要重训同名 E24/E25 产物时才使用：

```bash
bash scripts/run_all_followup_experiments_rtx3090.sh --overwrite
```

## 自动生成的分析产物

训练和 test-clean 评测完成后，主脚本继续生成：

- `results/master_metrics.csv`：统一 split、去重并用 prediction 复算指标；
- `results/followup_paired_bootstrap.csv`；
- `results/followup_paired_bootstrap.md`；
- 更新后的 `results/deep_dive_error_stats.csv`；
- 更新后的 `results/deep_dive_error_analysis.md`；
- 更新后的 `results/figures/*.png`。

配对 bootstrap 覆盖：

1. 旧 32 类 layer 8 与 E24A 30 类复跑；
2. E24B centroid 与 E24C embedding-256，以及 E24E embedding-768；
3. E24B K=200 与 E24D K=1000；
4. E16B seed 42 与 E25 seed 1234；
5. 1h full fine-tuning 与 1h frozen layer 8。

`master_metrics.csv` 会对已知系统性尾部 apostrophe padding 伪字符仅在指标复算
时进行修正，并在 `prediction_correction` 列记录；原 prediction 文件不会被
静默修改。
