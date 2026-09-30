# 实验日志

> 约定：每个 Stage 完成后追加一节，记录关键数字与 `results/results.csv` 归档指针。
> 产物（ckpt/CSV/日志）在服务器，不进 git；本文件只记结论与路径。

---

## Stage 0 — 基础设施与重构（2026-09-30，attn-predictor @ f364d79…）

### 重构与门禁
- `compression.py` → `conformer_rnnt/compression/` 包（core/detectors/mixers），
  detector × mixer 可组合；新配置 schema `model.compression.{detector,mixer,keep_rate}`。
- **parity 门禁**：5 个 legacy filter 逐位一致（`tests/test_parity.py`），服务器 6/6 PASS。
- **smoke 门禁**：15 个 B-matrix cell + 7 个固定率 detector 前向/反向/梯度可达
  （`scripts/smoke_chain.py`），服务器 23/23 PASS。
- ckpt 命名体现配置：`val-attn-none-…`、`train-attn-savgol-hard-r0.27-…`（train.py）。
- `eval.py`：`--out` per-sample CSV + `RESULT per=… rtf=…` 机器行 + `--beam` 覆盖。

### 已知问题（沿用）
- encoder mask 边界 bug（utt 长度 ≡ 0 mod stride 时 x_len+rc == T'）——历史遗留，
  合成数据绕开；LibriSpeech 真实数据未见触发。
- `save_weights_only=True` → 崩溃 run 只能重跑，不能 resume（队列策略：重启整 run）。

### 基线与锚点
- attn+savgol 主配置锚点：`/test21/conformer-rnnt-asr/exp/checkpoints/val-epoch=42-step=26700.ckpt`
  （greedy PER 11.12%，权重内嵌 hyper_parameters 已验证 savgol）。
- **none 基线从零训练**：tmux `baseline-none` → `/test21/conformer-rnnt-asr/exp_baseline_none/`，
  2026-09-30 11:09 UTC 启动，120 epochs，~3 min/epoch，ETA 当日 ~17:50 UTC。

### A0-3 压缩率实测（锚点 s42 前端，300 条 test-clean，CPU）
| detector | mean | p10 | p50 | p90 |
|---|---|---|---|---|
| naive | 0.391 | 0.361 | 0.394 | 0.421 |
| fir | 0.246 | 0.207 | 0.250 | 0.276 |
| savgol | **0.274** | 0.244 | 0.275 | 0.302 |

→ **等码率 r = 0.27**，A4 与 B 全部配置锁定（`configs/exp/*.yaml`，gen_configs.py）。
none anchor 训练完成后复测（`analysis/out/rate_distribution_none_anchor.csv`）。

### P12 DP oracle
- `dp_seg` detector：全局最优 k 分段（piecewise-constant L2 的 DP），配 `seg_avg` mixer。
- 单元验证：合成 4 段常值信号边界精确恢复；T=400/D=512 ≈10 s/句。
- 评测方式：none anchor ckpt + `configs/exp/oracle-dp.yaml`，推理-only（manifest 第 4 列）。

### 自动管线
- tmux `pipeline`（`scripts/stage1_pipeline.sh`）：等 baseline → smoke 门禁 → A0 诊断 →
  A4 队列（`scripts/queue_runner.py`，幂等，结果落 `results/results.csv`）。
- 队列内容：oracle-dp（推理）→ a4-uniform / random / topk-diff / energy / max-pool /
  avg-pool / mmdp / ple / cosine-diff / savgol-cal（各 ~5.5 GPU·h 训练 + greedy eval）。
- Stage 3 B-matrix（15 runs）就绪未启动：`scripts/stage3_pipeline.sh`，A4 Go/No-Go 后手动放行。
