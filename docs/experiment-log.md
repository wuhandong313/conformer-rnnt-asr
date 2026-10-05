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
- [x] a4-avg-pool（50ep 向量化版）epoch 速度确认 ✅
- [ ] A4 各 run 完成后 Go/No-Go：savgol ≥ uniform/avg-pool 且 DP-oracle 差距 >5%

### 首个 50-epoch 结果（2026-10-01 07:31 UTC）
| run | PER | RTF | val-best | 用时 |
|---|---|---|---|---|
| a4-avg-pool (avg_pool+seg_avg, r=0.27) | **0.1057** | 0.0381 | ep30/50 | 3.2h |

- val-best 落在 ep30 → **50 epochs 决策被数据支持**（后半程无浪费）。
- 向量化 mixer 生效：3.2h/run（此前 120ep 旧代码预计 ~17h/run）。
- 参考：savgol 锚点（120ep, s42）PER 11.12%；oracle-dp 天花板（推理-only）66.24%。
  注意 avg-pool 10.57% 与 savgol 锚点 epoch 数不同，等 A4 队列全部 50ep 出齐后统一对比。
- 队列接力：a4-cosine-diff 07:31 起跑（字母序第 2/10）。A4 预计 ~10-02 12:30 UTC 跑完。

### A4 记分板（50 epochs, r=0.27, greedy, test-clean n=2620）
| run | PER | RTF | val-best | 完成时间 |
|---|---|---|---|---|
| a4-avg-pool (avg_pool+seg_avg) | 0.1057 | 0.0381 | ep30 | 10-01 07:31 |
| a4-cosine-diff (cosine_diff+hard) | 0.1166 | 0.0730 | ep48 | 10-01 09:56 |
| a4-energy (energy+hard) | 0.1525 | 0.0379 | ep40 | 10-01 11:59 |
| a4-max-pool (max_pool+hard) | **0.1026** | 0.0733 | ep30 | 10-01 15:50 |
| a4-mmdp (mmdp+hard) | 0.1061 | 0.0469 | ep30 | 10-02 00:04 |
| a4-ple (ple+hard) | **0.7028** ⚠ | 0.0253 | ep15 | 10-02 01:47 |
| a4-random (random+hard) | 0.1302 | 0.0403 | ep31 | 10-02 03:47 |
| a4-savgol-cal (savgol+hard) | 0.1108 | 0.0413 | ep39 | 10-02 06:06 |
| a4-topk-diff (topk_diff+hard) | 0.1596 | 0.0420 | ep44 | 10-02 08:06 |
| a4-uniform (uniform+hard) | 0.1130 | 0.0408 | ep31 | 10-02 10:03 |

### A4 收官 + Go/No-Go 判决（10-02 10:03 UTC，队列 drain，总用时 ~40h）

完整排序（greedy PER，r=0.27，50ep，n=2620）：
**max-pool 0.1026** < avg-pool 0.1057 < mmdp 0.1061 < **savgol-cal 0.1108** < uniform 0.1130 < cosine-diff 0.1166 < random 0.1302 < energy 0.1525 < topk-diff 0.1596 < ple 0.7028⚠

判据评估：
1. "savgol ≥ uniform/avg-pool"：savgol 胜 uniform（0.1108 < 0.1130）但**负于 avg-pool（+0.51pt）与 max-pool（+0.82pt）→ 不成立**
2. "DP-oracle 差距 >5%"：oracle 0.6624 为推理-only 探针（未适应训练的模型上压缩即崩），非有效天花板，判据按原意失效

**结论：按计划判据 = No-Go**（自适应峰值检测在等码率 PER 上不敌简单池化）。
支持性观察：
- savgol 50ep 0.1108 ≈ 120ep 锚点 0.1112（epoch 策略再次验证）
- 自适应（savgol）> uniform（+0.22pt）：动态选帧本身有价值，但"帧差峰值"信号弱于"段内极值/均值"
- 帧差类动态检测器（cosine-diff/topk-diff/energy）普遍弱于 random 之外的池化系，ple 崩坏

### B-matrix 记分板（5 detector × 3 mixer，r=0.27，50ep，greedy n=2620；10-02 14:25 UTC 起跑）
| detector | hard | seg_avg | learned |
|---|---|---|---|
| cosine_diff | 0.1160 | 0.0959 | **0.0915** 🏆 |
| fir | 0.1118 | **0.0917** | 0.0977 |
| raw | 0.1212 | 0.0968 | 0.1119 |
| ple | 0.7224 ⚠ | 0.5013 ⚠ | 0.6390 ⚠ |
| savgol | 0.0983 | 0.0967 | 0.0926 |

注：b-cosine_diff-hard 0.1160 vs A4 a4-cosine-diff 0.1166 —— 跨批次复现一致性 ±0.06pt。
seg_avg 落库（10-02 21:56 UTC）：cosine_diff 行 mixer 排序 **learned(0.0915) < seg_avg(0.0959) < hard(0.1160)** 成立；
seg_avg 也超过 A4 冠军 max-pool（0.1026）——soft mixing 系（learned/seg_avg）整体强于 hard 选帧。
b-fir-hard 落库（10-03 00:21 UTC）：0.1118（RTF 0.0388，val-best ep45），介于 savgol-cal 与 cosine_diff-hard 之间。
b-fir-learned 落库（10-03 03:26 UTC）：0.0977（RTF 0.0422，val-best ep38）——**learned < hard 在第二个 detector（fir）上复现**（−1.41pt），
且 0.0977 也优于 A4 冠军 max-pool（0.1026）。两个 detector 一致：learned mixer 增益 −1.4 ~ −2.45pt。
b-fir-seg_avg 落库（10-03 06:11 UTC）：0.0917（RTF 0.0498，val-best ep41）——fir 行集齐。
**注意：seg_avg 与 learned 的排序在 fir 上与 cosine_diff 相反**（fir: seg_avg 0.0917 < learned 0.0977；cosine_diff: learned 0.0915 < seg_avg 0.0959）——
两个 soft mixer 互有胜负（差距 ≤0.6pt），但**两个 detector 上 soft 系（learned/seg_avg）都大幅领先 hard（−1.4 ~ −2.4pt）不变**。
b-fir-seg_avg 0.0917 与全场纪录 b-cosine_diff-learned 0.0915 仅差 0.02pt。
ple 行落库（10-03 07:57-12:27 UTC）：hard 0.7224（ep17 崩）/ learned 0.6390（ep12 崩）/ seg_avg 0.5013（ep48 跑完仍不可懂）
——**复现 A4 的 ple 崩坏，且换 mixer 无法挽救**：PLE 游走分数 + threshold=6.0 在 r=0.27 等率 top-k 下选帧机制本身坏掉，与 mixer 无关（可注：soft mixer 让崩溃稍轻 0.72→0.50）。
b-raw-hard 落库（10-03 14:37 UTC）：0.1212，比 fir-hard（0.1118）差——raw 峰值无平滑处理确实需要滤波。
raw-learned/seg_avg + savgol-hard/learned 落库（10-03 晚，27/28 行）：b-raw-learned 0.1119 / b-raw-seg_avg 0.0968 / b-savgol-hard 0.0983 / b-savgol-learned 0.0926。
**关键结论（14/15 run）**：
- soft >> hard 在全部 4 个正常 detector 上成立，平均增益 −1.87pt（hard 均值 0.1118 → 最优 soft 均值 0.0932）
- learned vs seg_avg 2:2 平（cosine_diff/savgol 上 learned 胜，fir/raw 上 seg_avg 胜），互有胜负、差距 ≤1.5pt
- **A4 No-Go 被翻案**：savgol+learned 0.0926 反超 A4 冠军 max-pool 0.1026（−1.0pt）——A4 输在 mixer 不在检测器
- 前三浓缩：cosine_diff-learned 0.0915 ≈ fir-seg_avg 0.0917 ≈ savgol-learned 0.0926（0.11pt 内），检测器选择远不如 mixer 重要

### B-matrix 收官（10-04 03:30 UTC，15/15，~37h）

完整记分板（greedy PER，r=0.27，50ep，n=2620）：

| detector | hard | seg_avg | learned | 最优 soft 增益 |
|---|---|---|---|---|
| cosine_diff | 0.1160 | 0.0959 | **0.0915** 🏆 | −2.45pt |
| fir | 0.1118 | 0.0917 | 0.0977 | −2.01pt |
| raw | 0.1212 | 0.0968 | 0.1119 | −2.44pt |
| savgol | 0.0983 | 0.0967 | 0.0926 | −0.57pt |
| ple | 0.7224 ⚠ | 0.5013 ⚠ | 0.6390 ⚠ | （机制崩坏） |

**五条结论**：
1. **mixer 是主导因素**：4 个正常 detector 上 soft 全胜 hard，平均 −1.87pt（0.1118 → 0.0932）。压缩本身不掉分，掉分的是丢掉未选帧信息。
2. **A4 No-Go 翻案**：savgol+learned 0.0926 反超 max-pool 0.1026（−1.0pt）；主方法（动态压缩+软混合）首次完整跑赢全部固定率池化基线。
3. **检测器只需"合理"**：前三在 0.11pt 内扎堆（cosine_diff/fir/savgol + soft），raw 慢 0.4-0.5pt。
4. **learned vs seg_avg 2:2 平**（cosine_diff/savgol→learned，fir/raw→seg_avg），互有胜负 ≤1.5pt，等价性待多种子检验。
5. **ple 独立崩坏**（三 mixer 全崩 0.50-0.72，ep12-17/48）：等率 top-k 与游走分数机制冲突，与 mixer 无关。

**对比不压缩基线**（baseline-none greedy，2620 utts 全量）：**0.0781**（RTF 0.0487）。压缩到 r=0.27 的质量代价：冠军 +1.34pt（greedy 口径）。
**RTF 陷阱**：baseline-none RTF 0.049 ≈ 压缩 run RTF（0.033-0.055）——墙钟被 RNN-T 解码循环/IO 主导，encoder FLOPs 的 3.65× 缩减不体现在 RTF 上，Phase E 必须做 FLOPs/延迟分解（已列 Stage 5）。

**后续（已布置）**：eval-none 后自动跑 beam20（cosine_diff-learned / fir-seg_avg / savgol-learned / max-pool 对照）；然后 Stage 4（D0-D2 + C2 + A6-360h）。
**🏆 learned mixer 重大发现**：cosine_diff+learned 0.0915 = 全场新纪录——
比同 detector 的 hard 好 2.45pt（相对 21%），比 A4 冠军 max-pool（0.1026）好 1.11pt，
比 savgol 锚点（0.1112）好 2pt。mixer 选择是主导因素：可学习加权混合在峰值处
的局部平滑显著优于硬选帧。val-best ep47（近上限，后续可试 60ep）。

参考（非受控对比）：savgol 锚点（120ep s42）0.1112；oracle-dp 天花板 0.6624。
进行中：**a4-uniform（08:06 起，最后一个 run）**，预计 ~10:45 UTC 出数后汇总 Go/No-Go。
注：savgol-cal 50ep=0.1108 vs savgol 锚点 120ep=0.1112 —— 几乎一致，验证 50-epoch 策略。

## Beam20 收尾评测（2026-10-04~05，2620 utts 全量，beam=20，per-model configs/exp/*.yaml）

| 模型 | greedy | beam20 | 增益 | RTF |
|---|---|---|---|---|
| cosine_diff-learned | 0.0915 | **0.0879** | +0.36pt | 0.592 |
| fir-seg_avg | 0.0917 | **0.0884** | +0.33pt | 0.591 |
| savgol-learned | 0.0926 | **0.0904** | +0.22pt | 0.568 |
| max-pool（A4 冠军对照） | 0.1026 | 重跑中（v2） | — | — |
| baseline-none | 0.0781 | 重跑中（v2） | — | — |

- 三个 B-matrix 头部模型（soft mixer）beam20 一致优于 greedy（+0.22~+0.36pt）；cosine_diff-learned beam20 **0.0879** 为全场最优。
- beam RTF ~0.57-0.59（约为 greedy 的 10 倍）——RNN-T beam 解码主导墙钟，再次印证 Phase E 需 FLOPs/延迟分解。
- 编排插曲：首轮编排器（scripts/eval_beam_top.sh）搜 max-pool ckpt 只查了 exp_b/exp 漏了 exp_a4 → 该格空缺；前三个模型用 per-model config 正常完成。

### ⚠ 评测伪影记录（2026-10-05，重要教训）

先前用 `configs/server_eval_beam20.yaml`（legacy `filter: "savgol"`，**无 keep_rate**）补跑的两组结果**全部作废**：
- max-pool beam20 = 0.1436（日志留档 *.invalid-savgol-artifact）
- baseline-none beam20 = 0.6444

**根因**：`eval.py` 用 config 建模 + `load_state_dict(strict=False)`——架构不匹配被**静默跳过**，模型实际以"未训练 savgol 检测器 + hard mixer + 动态率"运行，与被评 ckpt 的架构（max_pool-hard r=0.27 / none）完全不符。据此撤销"hard 池化与 beam 解码不兼容"的结论（无证据支撑）。原始三个 b-* beam 结果用 per-model config，未受影响。
**基础设施风险**：eval.py strict=False 是颗地雷，任何 config/ckpt 错配都不报错只出垃圾数——后续 eval 前应加架构一致性断言（列为 Stage 4 待办）。
**Go/No-Go 前哨**：受控 savgol（0.1108）目前落后 max-pool（0.1026，-0.82pt）与 avg-pool（0.1057，-0.51pt），判据第一半（savgol ≥ 固定率池化基线）暂不成立，等 uniform 出数后汇总判决。
⚠ a4-ple 训练崩坏：val-best 停在 ep15 且 PER 70% 接近不可懂——PLE（cosine 累积距离游走 + threshold=6.0）在 r=0.27 等率 top-k 下选帧不稳定导致训练发散。待 A4 出齐后分析（候选解释：PLE scores 分布与 top-k 重排序交互不良）。
