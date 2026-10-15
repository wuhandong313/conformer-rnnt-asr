# 训练计划（GPU 调度）v1.0

> 配套 `docs/research-plan.md` v2.1 §10。硬件：1 × A100-80GB（K8s 容器，`/test21`，conda env `whd`）。
> 实测基准（2026-09-19 attn run）：44.3M 参数 / train-clean-100 / 120 epochs = **76,300 步 ≈ 4.2 h**（302 步/min）；全量 test-clean eval：greedy ≈ 0.5 h，beam-20 ≈ 1 h。
> **结论先行：单卡 A100 足够。** 全部 P0–P1 实验 ≈ 16–20 个 GPU 日，14 周（98 天）内 GPU 占用率 <20%——瓶颈在基线实现与分析工程，不在算力。调度原则：**GPU 串行排队，工程在 GPU 空窗期推进**。

---

## 1. 运行队列（按优先级）

### 批次 1：A0 诊断 + 地基（第 1 周，GPU ≈ 1 天）

| # | 实验 | 内容 | 单次时长 | 次数 | 合计 |
|---|---|---|---|---|---|
| A0-1/2/3 | 诊断 pilot | 推理-only，`analysis/` 三件套跑现有 none-ckpt（逐层冗余 / 音素+词边界 F1 / 压缩率分布） | ~3 h | 1 | **3 h** |
| A1 | 向量化 | CPU 工程，不占 GPU | — | — | — |
| A4 实现 | 十基线 + PLE/cosine-diff 复现 + DP oracle | 编码 + `scripts/smoke_chain.py` 冒烟，不占 GPU | — | — | — |

### 批次 2：A4 基线（第 1–3 周，GPU ≈ 3 天）—— **Go/No-Go 判据**

| # | 实验 | 配置 | 单次 | 次数 | 合计 |
|---|---|---|---|---|---|
| A3 | ~~锚点 ×3 种子~~ **已取消**——无压缩锚点直接用 attn v2 既有 best ckpt：`exp/checkpoints/val-epoch=42-step=26700.ckpt`（greedy PER 11.12%，100-sample）。exp 分支的 Medusa-heads 种子 run（s42/s1111）架构不符，弃用 | — | 0 | 0 | 0 |
| A4 | 十基线（均匀/随机/top-k-diff/能量/max-pool/avg-pool/**PLE**/**cosine-diff**/MMDP）+ 我方主配置 | `configs/exp/a4-*.yaml` | 5 h | 12 | 60 h |
| — | DP oracle 天花板 | 推理-only，CPU 即可 | 2 h | 1 | 2 h |

> ⚠️ 全部实验以 **attn-predictor 分支**为基座（predictor trunk 4 层自注意力架构，服务器工作目录 `/test21/conformer-rnnt-asr`，其 configs 的 exp_dir 指向主 repo `/test21/conformer-rnnt-asr/exp`）。exp/a3-baseline-seeds 分支仅作文档/analysis 载体，不再跑训练。
> **No-Go 条件**（等压缩率对比）：均匀/avg-pool 与主配置打平且 oracle 差距 <5% → 转分析型故事（R1）。

### 批次 3：B-矩阵消融（第 4–6 周，GPU ≈ 3.5 天）

| # | 实验 | 内容 | 单次 | 次数 | 合计 |
|---|---|---|---|---|---|
| B-矩阵 | 检测器{raw,FIR,SG,PLE,cosine-diff} × 混合器{硬选,段平均,学习权重}，等 FLOPs 20% 档 | `configs/exp/b-*.yaml` | 5 h | 15（关键格 2 种子） | 75 h |

### 批次 4：泛化骨干 + 学习门控 + 360h 锚点（第 6–10 周，GPU ≈ 4 天）

| # | 实验 | 内容 | 单次 | 次数 | 合计 |
|---|---|---|---|---|---|
| D0+D1+D2 | 架构/任务/数据泛化 | 4–7 runs | 5 h | 6 | 30 h |
| C2 | 可微门控（信号初始化 vs 随机初始化） | 2–3 runs | 5 h | 3 | 15 h |
| **A6** | **360h 规模锚点**（train_subset: train-clean-360） | 1 run | **15–18 h** | 1 | 18 h |

### 批次 5：收尾（第 10–14 周，GPU ≈ 4 天 + 缓冲）

D3–D8 泛化补齐（~10 runs）、A6 关键对照（2 × 18 h）、Phase E 效率实测（推理-only）、种子补齐与重跑缓冲 ≈ **90 h**。

**总计 ≈ 260 GPU·h ≈ 11 个连续 GPU 日**，分布到 14 周。（v1.1：A3 种子实验取消，复用 attn v2 既有 ckpt，−30 h。）

---

## 2. 基础设施约定（全部批次通用）

1. **exp_dir 隔离**（吸取 attn run 教训）：每个批次独立目录——`exp_a0/ exp_a3/ exp_a4/ exp_b/ exp_d/ exp_a6/`。ckpt 文件名不带 run 标识，共用目录会被静默覆盖且 `strict=False` 加载不报错。
2. **screen 命名**：`a3-train` / `a4-train` / `b-train` …；监视器沿用 `monitor_iter2.py` 模式，**换批次先清 `monitor_eval/` 缓存**。
3. **启动模板**：
   ```bash
   screen -dmS a4-train bash -c 'cd /test21/conformer-rnnt-asr && /root/anaconda3/envs/whd/bin/python train.py configs/exp/a4-ple.yaml 2>&1 | tee artifacts/logs/exp_a4_train.log'
   ```
   串行排队用 `queue_*.sh`（A3 种子已有先例）。
4. **eval 管线**：每 run 结束后 greedy 全量 test-clean（0.5 h）进队尾；beam-20 只对主配置与关键基线跑（1 h）。
5. **结果落盘**：每批次结束即写 `docs/experiment-log.md`（AGENTS.md 硬性要求），数字带 ckpt 标识与样本量口径。

## 3. A6（360h）特别说明

- 数据已就位：`/test21/medusa/LibriSpeech/train-clean-360` + 音素标注（104,014 句，2026-09-30 生成）。
- 配置改动仅 `data.train_subset: train-clean-360`；步数/epoch ≈ 3.6×（≈2,290 步/epoch）。**epochs 降到 40**（总步数与 100h×120 相当，控制对比变量）；warmup 不变。
- 单 run ≈ 15–18 h，排在批次 4 末尾（前面 A0/A4 结论成立再开）。

## 4. 风险

| 风险 | 对策 |
|---|---|
| 单卡串行，某 run 挂掉阻塞队列 | 队列脚本带 `set -e` 逐 run 落盘日志；夜间队列每日早检 |
| 共享盘 105T 被写满（ckpt ×50 runs） | `save_top_k: 2` + `save_weights_only`；每批次结束清理非 best ckpt |
| 服务器 pod 被停（2026-09-30 已发生一次） | 面板重启后先 `git pull` 核对 exp_dir 完整性再续队 |
