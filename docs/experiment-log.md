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

---

## Stage 1 — A0 诊断 + 管线自动推进（2026-09-30 17:37–，UTC）

### 基线完成
- none 基线 17:37 UTC 训完（120 epochs, step 76079）。ckpt 冻结现象说明：
  val ckpt 停在 **ep26**（val_loss 最优，之后过拟合）；train ckpt 停在 ep91
  （save_top_k=2 + train_loss 平台）。**锚点 = val-attn-none-epoch=26-step=16600.ckpt**（val-best）。

### 自动管线里程碑
- 17:37 smoke 门禁启动 → 17:45 通过（2-epoch 全链路 train→ckpt→eval→CSV，
  `results/results.csv` 首行 smoke-detector PER 0.9694 —— 链路正确性验证，质量无意义）。
- 17:45–17:47 A0 诊断（none anchor，CPU）：
  - rate 复测（300 utt）：savgol **0.308**（0.280/0.309/0.333）、fir 0.279、naive 0.405
    → 两锚点差异（savgol-anchor 0.274 vs none-anchor 0.308）= rate drift 素材。
    **r=0.27 维持**：等码率预算锚定主方法实际率。
  - layer_redundancy_none.csv 已存（analysis/out/）。
- 17:47 A4 manifest 生成（11 runs，oracle-dp 首位、用 anchor ckpt 推理-only）。
- 17:47– oracle-dp 贪心评测中（~44 utt/min，预计 ~18:47 完成；中期 cum PER 0.667
  —— 压缩式 oracle 在未见过压缩的模型上退化属预期，作同条件天花板用）。
- 之后：a4-uniform 起跑，10 个训练 run 串行（~55 GPU·h）。

### 2026-10-01 修正与决策
- **oracle-dp 完成：PER 0.6624, RTF 0.1865**（none-anchor ep26, r=0.27, n=2620）。
  天花板解读：全局最优分段在未经压缩训练的模型上也退化到 0.66 → 压缩必须适应性训练（A4 从头训练验证）。
- **速度诊断**：seg_avg/learned mixer 的 Python 逐段循环使 batch 前向 ~2.4s
  （hard 仅 38-64ms），a4-avg-pool 实测 8.1 min/epoch（基线 3.3 it/s ≈ 2.7 min/epoch）。
  **向量化**（index_add / masked windowed gather），~ms 级；learned 与 legacy 循环
  仅 ~1e-7 FP 累加顺序差，parity 门禁对 learned 放宽 allclose(1e-5)，hard 仍位精确。
  seg_avg 首段偏移 bug（frames[0:indices[0]] 不属于任何段）由 smoke 门禁 19/23 抓到，
  修复后 parity 6/6 + smoke 23/23。
- **【用户决定】所有训练 run 统一 50 epochs**（原 120）。理由：none 基线 val-best 在
  ep26，120 epochs 后半段纯浪费。a4-avg-pool 的 120-ep 跑（~ep65）终止归档
  （exp_a4/a4-avg-pool.aborted-120ep），50-ep + 向量化 mixer 重跑。
- 管线重启（04:14 UTC），幂等跳过 smoke/oracle 已完成项，a4-avg-pool 重跑中。

### 待办
- [ ] oracle-dp 结果核对（天花板 PER + RTF）✅（0.6624 / 0.1865）
- [ ] a4-avg-pool（50ep 向量化版）epoch 速度确认
- [ ] A4 各 run 完成后 Go/No-Go：savgol ≥ uniform/avg-pool 且 DP-oracle 差距 >5%
