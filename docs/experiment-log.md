# 实验记录 (Experiment Log)

> 目标: 在不牺牲提前预测 (early draft) 目标与无损性的前提下, 持续提高 Medusa
> 链式头的投机接受率 (h1/h2/h3)。
> 每轮迭代实验结束后在此追加一节, 并更新总览表。评估口径见文末「评估约定」。

## 总览

| 迭代 | 核心改动 | h1 | h2 | h3 | overall | PER | 结论 |
|------|---------|-----|-----|-----|---------|------|------|
| iter-1 | 多头 + 消融筛选 | 0.80 | 0.22 | — | ~0.41 | 0.073 | kl=0 胜出 (stream_h2_kl0) |
| iter-2/2b/2c | 链式草拟 chain_states | — | — | — | — | 0.073 | 链式条件建立 |
| iter-2d | lookahead 1→8 (解除流式限制) | ↑ | — | — | — | 0.073 | h1 天花板是信息受限 |
| iter-3 | draft-conditioned 链式头 | — | — | — | — | 0.073 | train/infer 同构条件 |
| iter-4 | rolled language states | 0.911 | 0.45 | 0.10 | ~0.49 | 0.073 | 全 test-clean lossless 2619/2620 |
| iter-5 | 窗口帧级 attention | ✗ | ✗ | ✗ | ✗ | 0.50 | **失败**, 破坏 greedy, 回滚 |
| iter-6 | 共享深路径 context_mlp_extra | 0.927 | 0.690 | 0.365 | 0.661 | 0.073 | 零初始化扩容有效 |
| iter-7 | 逐头容量 (6.3M→13M) | 0.920 | 0.688 | 0.346 | 0.651 | 0.0729 | **容量翻倍无效**, 平台确认 |
| iter-8 | 1 帧预算: sim 扫描+时延分解+直注 | 0.841 | 0.494 | 0.228 | 0.521 | 0.0756 | **未超 iter-7, 舍弃** (见教训) |
| iter-8h2 | 推理窗修复 (a56ab42) + 2 头重训 | 0.889 | 0.612 | — | 0.751 | 0.0733 | 超 iter-7 同头口径, 已停 |
| iter-9 | **投机显示 (Speculative Display)** | — | — | — | — | — | 价值模型换轨, 见下 |
| iter-10 | 删窗 + draft seq model + BNT 蒸馏 (**从头训练**) | 0.551 | 0.261 | — | 0.406 | 0.1077 | 全 test-clean, ep32; h1 仍上升中 |

## 评估约定

- 接受率: `eval.py configs/server_chain.yaml --max-samples 100`, test-clean 前
  100 条 (batch_size=1, 与 monitor 同口径); 全量 test-clean 数字会单独注明。
- 指标定义: head-k 的 rate = accepted/drafts (链式条件积, 含前缀接受);
  overall = 全 draft 合并。PER = 音素错误率; lossless = 投机输出与 greedy
  oracle 逐 token 一致 (结构性保证, 每轮抽验)。
- 训练环境: 服务器 `/test21/conformer-rnnt-asr`, conda env `whd`
  (python 3.10, torch 2.6.0+cu124, **需 soundfile** 供 flac 加载),
  screen 会话 `iter2-train`, 监控 `scripts/monitor_iter2.py`。

---

## iter-1 — 多头基线与消融 (multihead 分支起点)

- 改动: joiner 之上加 K 个零初始化 Medusa 残差头 (stream_local 上下文);
  消融矩阵筛选 (medusa_auto_train.py)。
- 结果: WINNER = `stream_h2_kl0`, h1=0.80 / h2=0.22; KL 拉向 blank-dominant
  的 base 分布, 与 CE 冲突 → **kl_weight: 0 成为定局** (47cdf03)。
- 结论: 零初始化 + 冻结 base 的安全模式成立; KL 出局。

## iter-2 / 2b / 2c / 2d — 链式草拟 + lookahead 解除 (2026-08 下旬)

- 改动: `chain_states: true` — head-k 条件于 teacher-forcing 滚动状态;
  `draft_after_emit` (post) 对齐; iter-2d 用 probe_emit_gap 证明 lookahead=1
  下 h1 的错误 60% 发生在 gap=1、40% 在 gap≥2 (emit 帧不可见) → 解除到 8
  (d7c5b1c, 非流式)。
- 结果: h1 显著提升但存在信息天花板; h2/h3 仍低。
- 结论: 链式框架定型; lookahead=8 为后续所有 iter 的训练设定 (但违反流式
  right_ctx, iter-8 收回)。

## iter-3 — draft-conditioned chain heads (ec54135)

- 改动: head-k (k≥2) 额外消费前一头 draft 的 embedding; 训练用前头 argmax
  (detach), 推理用真实 draft — 条件严格同构, 取代失败的 LSTM 状态链。
- 结论: 同构条件是后续一切的基础。

## iter-4 — rolled language states (b79dfed / fb90ec4)

- 改动: head-k 额外接收 rolled 语言状态 (训练: `tgt_enc[l+k-1]`; 推理:
  predictor 在已接受 draft 上滚动), 零初始化投影注入。
- 结果: warm-start from iter-3 best; **h1=0.9105** (全 test-clean,
  lossless 2619/2620), h2≈0.45 / h3≈0.10。
- 结论: 语言状态滚动让 h1 破 0.91; h2/h3 仍差。

## iter-5 — 窗口帧级 attention (9d19fdd / c328eea) — 失败

- 改动: rolled state 作 query 对 (B,T,W,D) 声学窗做帧级 attention。
- 结果: 训练不稳, greedy PER 0.07→0.50 (输出被破坏)。回滚:
  `win_attn_enabled=False`, 模块仅为 checkpoint 兼容保留。
- 教训: 会触碰 verifier 路径数值的机制一律不做; additive-only。

## iter-6 — 共享容量扩容 (a33f31e / cb3f5a6 / 4d89ef5)

- 改动: `context_mlp_extra` — 平行 3 层深路径, 输出层零初始化 (step-0 与
  iter-4 bit-identical); warm-start BEST_iter4.ckpt; 途中经历 warm-start
  PER 瞬态 (monitor 8139271 加了标记) 自愈。
- 结果: **h1=0.927 / h2=0.690 / h3=0.365** (超 iter-4)。
- 结论: 深路径有效; 但 h2 在 ~0.68 平台 10 个 epoch → 共享容量饱和假设。

## iter-7 — 逐头容量 (54ec1dd / 2be6c65) — 容量路线证伪

- 改动: 每头独立 `context_mlp_heads` (3 层) + `chain_state_proj_heads`;
  medusa 参数 6.3M→13M (翻倍); warm-start iter-6 best (h1=0.915/h2=0.678/
  h3=0.362 @epoch17)。
- 结果 (monitor @ep14, step=37400): **h1=0.920 / h2=0.688 / h3=0.346 /
  overall=0.651**, PER=0.0729, RTF=0.0405; h3 对 oracle tail 也只有 0.383,
  失败集中在 emit 距离 2-4 帧 (接受率 0.37)。
- 结论: **容量不是瓶颈** — 参数翻倍后 h2/h3 原地踏步。瓶颈是头缺"验证器
  会发射什么"的信息/计算结构。iter-8 转向。

## iter-8 — 1 帧流式预算: 验证器前瞻 + 时延分解 (进行中)

设计原则 (用户指定): 放弃规模路线; 架构+loss 双改; 严格满足 streaming 分支
的每帧 1 帧未来预算 (`streaming.yaml` right_ctx=4 mel = 1 reduced 帧)。

- 改动 (74eaf3a):
  1. 声学窗重排 `[t-7..t-1, t, t+1]` (past_frames=7, 宽度不变, warm-start
     保形) — 非法未来帧换合法过去帧;
  2. `chain_sim`: head-k 读真 joiner 在 `[t, t+1]` 窗对其 rolled state 的
     逐步 argmax (token/偏移/置信度, 递归起点对齐验证器), 零初始化注入;
     训练从 base_logits 网格 gather (零 FLOPs), 推理 K 次小 joiner 窗;
  3. `delay_decompose`: 每头时延分类器, 蒸馏验证器 blank-walk 时延标签;
  4. `accept_ce` 升级为 `verifier_distillation_loss` (验证器发射帧完整
     softmax 软目标; 训练网格全知, 允许取窗口外帧)。
- **上限诊断** (diag_rollout_ceiling.py, iter-7 权重, 100 样本):
  sim 读出上限 h1=0.948 / h2=0.918 / h3=0.894 (覆盖率 0.966/0.939/0.918);
  完全信息上限 0.977/0.991/0.991 — **天花板成立**, 1 帧预算几乎不亏。
- 8a 训练 (iter-7 warm-start): 冷启动 0.452 (窗重排瞬态) → epoch0 末 0.504
  → epoch2 平台 **0.50-0.51**。PER 全程 0.072-0.078, lossless=0。
- 8b 调整 (f66bf66): 蒸馏压力 ×3 (accept_ce 3.0 / delay_ce 1.0) → 0.524,
  斜率不足。
- 8b' 直注 (a5964c3): 隐路径复制电路 (proj→res_block→w_vocab) 学习太慢;
  新增 logit 级直注 `logits_k += sim_bias_heads[k][sim_id]` (零初始化
  Embedding(V,V), 1.7 万参数)。step400=0.523 / step1200=0.519; bias 活跃行
  0.3-0.5 @step1200, 预计 epoch1-2 见跳变 (滚动更新)。
- 教训 (滚动): 蒸馏损失 5.18→2.5 说明监督信号有效, 瓶颈在头的表达能力
  通路; 直注把复制问题变成 75 类直达梯度查表。
- 8b'' 直注修复 (469901f): 解剖发现偏置在 blank 行学出巨大反 blank 信号
  (范数 6.2) 而真实 token 行对角元 ≈0 — 稠密网格上发射稀疏, 复制梯度被
  稀释。改为只在 found 位置应用偏置 + rezero 重学; 10x 专用 lr 组加速。
- **最终结果** (epoch=16, step=44400, 100 样本): h1=0.841 / h2=0.494 /
  h3=0.228 / **overall=0.521**, PER=0.0756, lossless=0。偏置对角元学到
  0.5-1.4 (逐头递增) 但行范数同时涨到 20+ (复制与偏移混合)。
- **未超 iter-7 (0.651), 舍弃。** 三条教训:
  1. **信息守恒**: 1 帧预算砍掉的 7 帧未来声学是真实损失 — sim 只能恢复
     窗口内可解析事件 (h2 32%/h3 7%), 其余靠 LM 先验 (~0.5 条件命中),
     补不回 iter-7 隐路径从 8 帧未来获得的泛化。
  2. **上限诊断的方法论错误**: diag_rollout_ceiling 的扫描未限窗 (用了
     全序列 enc), 测出 0.92 是「无约束 rollout」上限而非 1 帧窗上限 —
     误导了方案可行性判断。天花板测量必须与部署可见窗一致。
  3. **直注查表对训练动态敏感**: blank 稀释 → 行范数失控 → 竞争干扰,
     三连坑; 隐路径 (chain_sim_proj→res_block→w_vocab) 复制电路学习慢
     的根因是两层冻结变换, 不是监督不足。
- 保留资产: verifier_distillation_loss (软目标蒸馏, loss 5.18→2.5 有效);
  过去帧换未来帧的窗重排被证伪 (分布 shift 伤害 warm-start 路径);
  delay 分类器被扫描语义包含, 冗余。

## iter-8h2 — 推理窗修复 + 2 头重训 (进行中)

- 背景: 发现 iter-8 全系评估的推理路径 bug (a56ab42) — 推理时
  `_chain_shared_terms` 对 spec_decode 已建好的 (B,W,D) 窗再次
  `_anchor_window(·, T=1)`, past 列清零、当前列错取 `enc[t-past]`、
  lookahead 列清零, context_acoustic 在推理时只见过单列。训练路径
  不受影响 (bit-identical), 故 0.521 是坏推理测出的数字, "信息守恒"
  结论存疑, 需以修复后的推理重训。
- 改动: 配置 2 头 (num_heads: 2, h3 依既定决策舍弃, warm-start 加载
  自动丢 medusa_heads.2.*); ce_loss_weights [0.8, 0.64]; max_draft 2;
  其余沿用 iter-8 最终态 (accept_ce 3.0 / sim_bias found-only +
  10x lr / sim_bias_rezero), warm-start 8b''' 重启点
  (epoch=01-step=5200-v1)。
- 口径: monitor 100 样本, 对照 iter-7 matched-head h1=0.920/h2=0.688
  (ep14) 或 h1=0.925/h2=0.702 (ep11)。
- **最终结果** (epoch=09, step≈25500, 100 样本): **h1=0.889 / h2=0.612 /
  overall=0.751**, PER=0.0733, lossless_fail=0。按同头数口径已超 iter-7
  (h1 0.920/h2 0.688 是 3 头模型的 matched-head 数字)。
- 已停止 (用户指令 2026-08-31): 训练 screen `iter2-train` 与监控
  `iter8-watch` 均退出, 无残留进程; ckpt 与 log 留档供 iter-9 warm-start。

---

## iter-9 — 投机显示 (Speculative Display) — 价值模型换轨 (进行中)

**换轨动机 (用户提出)**: Medusa 草稿只用于上屏显示, 真实 token 到达时
确认/替换; 解码回归纯贪心 RNNT。因此**明示放弃真实提速** (RTF 不变),
旧指标"接受率"作废, 新指标为

    value(head-k) = 命中率 × 平均提前量(ms)  −  替换(闪烁)惩罚

即"感知提前量" = 提前量 × 命中率。提前量以 reduced 帧计 (1 帧 = 4 mel
= **40ms**); 命中 = 草稿与随后真实 emit 的 token 相同; 替换 = 上屏后被
改写 (感官闪烁)。

### 本轮产出

- `scripts/eval_display.py` (新): 两趟评估器。pass 1 纯贪心产出参考序列
  并快照每个 emit 点的 post-emit 状态; pass 2 在快照上草拟并与
  `target = u + 1 + j` 对齐打分。输出 `hit% / repl% / leadP50,P90,P99 /
  leadAvg / value`; chain_sim 开启时追加 `found% / simRight% / copied%`。
- ckpt 选择**去掉 `val-*` 通配回退** (曾抓到 iter-8h2 的 ckpt → strict 加载
  报 11 个 unexpected 键), 只认本配置 tag 或 `paths.from_checkpoint`。
- `configs/server_display.yaml` (新, 阶段 1 训练配置, 尚未启动):
  2 头 / `stream_local` / lookahead=1 / **past_frames=7** / post-emit /
  chain_states / `freeze_base: true` / `tag_suffix: "disp"`。
- `configs/display_iter7.yaml` / `configs/display_iter8h2.yaml` (新):
  两套评估配置, 分别对应 iter-7 的训练布局与 iter-8h2 的流式布局。
- `common.py`: `model_ckpt_tag()` 增加 `tag_suffix` 旋钮 (参数相同但价值
  模型不同的 ckpt 不再同名, 避免 monitor 缓存串档)。
- `scripts/smoke_chain.py`: 新增测试 10, 哨兵值法断言共享 MLP 层保留 /
  过期头键丢弃 (缺数据集时自动 SKIPPED)。
- `docs/iter9-speculative-display.html` (新, 单页说明): §2 是可交互的**显示
  时间线仿真**。token 单位 = **音素** (数据集
  `Librispeech_phoneme_transcription`, spm 74 片), 演示串
  `/DH AH0 S AH1 N IH1 Z/` = 23 帧 (920 ms), 7 次真实 emit。
  **该时间线是合成的, 不是实测**: 实测每头平均提前量只有 33 ms (h1) /
  57 ms (h2), 演示为让"上屏 → 确认/替换"肉眼可见把 emit 间隔放大到 2–4 帧,
  只有上屏时机与评分的逻辑与真实系统一致 (上屏发生在草拟那一刻, 与将来是否
  命中无关)。演示终态: h1 hit 83.3% / leadAvg 112 ms / value 93.3 ms,
  h2 hit 60.0% / leadAvg 227 ms / value 136.0 ms — 复现了两头的实测关系
  (h2 命中率低但每次命中提前更久)。
- `docs/iter-key-nodes.html` (新, 回顾页): iter-1 → iter-9 关键节点回顾 ——
  按四阶段组织 (建立框架 / 容量实验 / 流式预算 / 换轨显示), 14 张节点卡片
  各含动机/改动/结果/教训, 附 8 条方法论与总览表 + 口径说明。数字全部取自
  本日志, 无新测量。

### bug 修复 (两个, 都影响过历史数字)

1. **warm-start 误删共享 MLP 权重** (`lightning_module.py:618-660`):
   过期头键过滤取 `medusa_heads.` 后第一个数字段, 把共享 MLP 的**层**索引
   (`context_mlp.2`、`context_mlp_extra.2/4`) 误判为头索引 ≥2 而丢弃
   (BEST_iter7 中范数 97.9 / 76.4 / 40.2 → 丢弃后回落零初始化)。改为只在
   per-head ModuleList (`res_blocks`, `context_mlp_heads`,
   `chain_state_proj_heads`, `sim_bias_heads`, `delay_heads`) 下读索引。
   **iter-8h2 与 iter-9 首次基线都中了这招**, 数字须以此后的为准。
2. **布局错配的假基线**: BEST_iter7 的 `context_acoustic` 是 `(512, 4608)`
   = 9 列, 但语义是 **iter-2d 解除流式后的 `[t..t+8]`** (past=0,
   lookahead=8, 非流式); iter-8h2 才是 `[t-7..t+1]` (past=7, lookahead=1,
   流式合法)。**同宽不同义** — 用 iter-7 权重跑 past=7 布局是错配, 实测
   h1 仅 47.91% / h2 14.81%, 与日志 0.920 严重不符, 数字无效。

### 三组显示基线 (100 样本, 服务器)

| 配置 | 窗口 | h1 hit% | h2 hit% | h1 value | h2 value | merged hit% |
|------|------|---------|---------|----------|----------|--------|
| A `display_iter7.yaml` (BEST_iter7) | `[t..t+8]` look=8 **非流式** (训练布局) | 94.81 | 67.60 | 35.1ms | 45.0ms | 81.21 |
| B `display_iter8h2.yaml` (ep07 st20400) | `[t-7..t+1]` look=1 **流式合法** (sim 开) | 90.60 | 59.38 | 30.3ms | 34.0ms | 74.99 |
| C BEST_iter7 @ 错配 | `[t-7..t+1]` | 47.91 | 14.81 | — | — | — |

提前量分布: leadP50=40ms / P90=80ms / P99=120-160ms (两配置一致)。

- **结论 1**: 流式预算几乎不亏 — lookahead 8→1 只掉 h1 −4.2pt / h2 −8.2pt。
  (iter-8 的"信息守恒"结论曾据此存疑, 现确认流式不是主要矛盾。)
- **结论 2**: 瓶颈是**提前量** (h1 33ms / h2 57ms ≈ 1 帧上下), 不是命中率;
  两头合计期望感知提前量 ≈ 64ms。
- **结论 3**: iter-9 原计划"重训流式合法头"收益很小 — 天花板就是 A。

### h2 瓶颈诊断 (chain_sim 推理端, 配置 B)

```
head      scan   found%  simRight%   copied%
h1        8370    85.72     100.00     99.79
h2        8269    45.02      82.76     99.81
```

- `found` = sim 扫描在**可见窗口**内问到了答案 (问的是 oracle 自己, 故
  found 时 simRight 必为 100%); `copied%` = 直注查表 `sim_bias_heads`
  被点亮的比例, ≈100% 说明复制电路通畅。
- h2 训练曲线是平的 (ep00 0.590 → ep02 0.579 → ep04 0.589 → ep07 0.612),
  续训救不了。
- **根因是物理上限, 不是能力问题**: h2 目标 token 中位落在 1 帧后、P90
  落在 2 帧后, 而 lookahead=1 的窗只有 `[t, t+1]` — `found%=45%` 与
  `P(lead ≤ 1 帧)=45%` 精确吻合。
- h2 那 17.2% 错误是**级联**: h2 的状态建立在 h1 草稿上, h1 猜错 9.4% →
  传导给 h2。h1 位置上"头其实是多余的" (直注复制率 99.8%, 跑 2 次小
  joiner 就能拿到 85.7% 覆盖 + 100% 正确), 头真正不可替代的只有给更深的
  头提供**滚动状态**。

### 通道拆分诊断: 复制区 vs 预测区 (2026-08-31)

> 用户纠正口径: "我要做的就是预测啊, 给这些信息让头学会预测, 你就是要测
> 预测准确率"。此前把 `found%` 当"头的上限"是错的 — 那量的是复制通道,
> 不是头的预测能力。头干两件事: ① 照抄 sim 读出 (复制区, 头是多余的);
> ② 答案在窗外时自己猜 (**预测区, 这才是头的战场**)。

新增测量 (`eval_display.py`): 区域拆分 + `--no-sim` 纯预测模式 (medusa 加
`sim_inject_enabled` 开关, 扫描照跑但不注入) + `_lm_free_run` 纯语言先验
基线 (predictor 自回归喂自己的 argmax, 声学侧固定为 enc 句级均值)。

**结果 1 — h1 与 h2 的预测能力完全相同** (配置 B, 100 样本):

```
head    copy_n  copyHit%  pred_n  predHit%   LMhit%  LMguess%
h1        7175     99.79    1195     43.01    13.19     16.25
h2        3723     82.76    4546     42.85     1.48      1.77

预测区 LM 先验 vs 头 (同一批样本):
h1  pred_n=1195  headHit 43.01   LMhit 2.18   gap +40.84
h2  pred_n=4546  headHit 42.85   LMhit 0.64   gap +42.21
```

**43.01% vs 42.85%** — 两头一样。h2 总命中低 (59.38 vs 90.60) 的唯一原因
是它在预测区的时间是 h1 的 3.8 倍 (55% vs 14%), 不是它笨。**h1 那个 90%
是复制通道刷出来的虚高。**

**结果 2 — LM 先验几乎不可用, 头远超它**: 预测区 LM 只有 2.18% / 0.64%,
且 84-98% 的时间直接弃权 (这个 predictor 在常量声学下几乎恒输出 blank)。
头高出先验 40+ 个点 → **头确实在用声学窗外推, 不是在当语言模型**。声学窗
是有用的, 头也确实用上了。

**结果 3 — 预测能力是「窗内未来音频帧数」的函数** (配置 A, iter-7, 无 sim
通道故全部落在预测区, lookahead=8):

```
h1  pred_n=8441  headHit 95.94   LMhit 14.31   gap +81.63
h2  pred_n=8341  headHit 69.22   LMhit  1.70   gap +67.52
```

同一机制、同样没有复制通道, 只是窗从 1 帧放宽到 8 帧: **43% → 96% / 69%**。
命中率可分解为

    hit% ≈ 覆盖率 × ~100%  +  (1 − 覆盖率) × 43%

覆盖率由 lookahead 决定 (h1 85.7% / h2 45.0% @look=1), 43% 是与头深度无关
的常数。**瓶颈钉死在「窗里能看到多少未来音频」。**

**结果 4 (意外) — 头把「该不该发射」外包给了 sim 通道**: 关掉 sim 注入后
h1 弃权数 10 → 4984 (弃权率 0.1% → 59%), hit% 反升到 96.37%。头自己的预测
路径没学会何时开口, 只学会了照抄。注意 no-sim 的样本集因弃权行为改变而
自选择 (found% h1 85.72→98.31 / h2 45.02→67.43), **不可与 sim 模式直接比**。

**杠杆估算** (以 h2 为目标):

| 杠杆 | 机制 | h2 命中率估算 | 代价 |
|------|------|--------------|------|
| 提覆盖率 | lookahead 1→2/3 (chunk 对齐) | ≈78% (0.88×0.83 + 0.12×0.43) | chunk 内零延迟 (最多 3 帧); 跨 chunk ≈+160ms; medusa 重训 |
| 提 43% | 加长过去窗 (外推更强) | ≈70% (假设 43→60%) | 零延迟, medusa 重训 |
| 两者都做 | — | ≈80% | 两者代价之和 |

注: "提 43%" 能否成立未验证 (见下), 而覆盖率杠杆有配置 A 的实测支撑,
确定性高得多。

### 衍生: 门控显示 (零训练成本)

只在 `sim found` 时上屏 (扫不到就弃权不显示):

| head | 闪烁 before → after | value before → after |
|------|--------------------|----------------------|
| h1 | 9.4% → 0% | 30.3ms → 28.3ms |
| h2 | 40.6% → **17.2%** | 34.0ms → 21.0ms |

代价: h2 命中率本身不会超过 60% (受 found% 45% 硬约束)。

### 流式预算口径修正 (2026-09-12)

早期表述「lookahead 只能是 1」沿用 iter-8 的 right_ctx 模型, **有误**。正确的零延迟
预算是 **chunk 对齐** 的: 流式块 = 左 64 mel + chunk 16 mel + 右 4 mel, stride 4
⇒ 一个 chunk 同时产出 4 个 reduced 帧 (右上下文不单独压缩), 故帧 t 的零延迟未来
预算 = 3 − (t mod 4) ∈ {3,2,1,0} (均值 1.5 帧); 只有跨 chunk 读取才付 ≈16 mel
≈ 160ms 真实延迟。lookahead=1 是 iter-8h2 权重的训练布局 (保守取整), 不是物理上限;
iter-8 章节「每帧 1 帧未来预算」的框架自此作废 (其接受率实验结论不受影响, 但不宜
再引用其预算表述)。影响: 上表「提覆盖率」杠杆代价由「+40ms 系统延迟 + base 重训」
修正为「chunk 内零延迟 (最多 3 帧); 跨 chunk ≈+160ms; medusa 重训」。
注: 动态压缩下 mel→reduced 非严格 4:1, 3 为标称值, 实际延迟按 ms 校核。

---

## iter-10 — 删头的声学窗 + draft 侧因果序列模型 + BNT 蒸馏 (从头训练, 进行中)

设计文档: `docs/iter10-experiment.md`。架构改动 (6fd9176): 头侧声学窗整体删除
(`predict_heads` 只接受 (B,1,D) 单帧), 新增 draft 侧 3 层因果 transformer
seq_model (d=512, 零初始化 draft_state_proj 注入 + LM_aux), BNT 蒸馏
(标签 = 验证器 ≤4 帧内首个 non-blank, δ=0 治弃权)。

**运行口径变更**: 按用户指示**从头训练** (`configs/server_display_scratch.yaml`,
`from_checkpoint: null` + `freeze_base: false`, epochs 120 / warmup 10000,
lm_weight 0.1 / bnt_weight 0.2 / seq_lr_scale 0.5), 不复用 iter-9 warm-start。
冒烟测试 8 节全绿后启动 (tmux `iter10-train`)。

**事故记录** (2026-09-18):
1. 第一次运行死于 ep24: 宿主机层 (docker overlay, 928G/932G) 写满, ckpt 保存
   mkstemp ENOSPC。本容器可见文件仅 ~25G, 需宿主机清理。
2. `exp_iter2/` 整目录 (含 iter-2~9 全部历史 ckpt) 在 NFS 上被外部删除, 非
   本会话所为; 历史最优 warm-start 源不可恢复 (从头训练路线下影响有限)。
3. 从头重启后监控两次被 tmux server 重启打断; monitor 修复两处:
   只匹配 `val-*disp10*` ckpt; 评估前快照 ckpt 到 `monitor_eval/` (防
   save_top_k 轮换删除引发 FileNotFoundError)。

**训练轨迹** (monitor 100 样本, tag disp10):
- h1 接受率: 0.27 (ep0) → 0.45 (ep12) → 0.50 (ep22) → 0.532 (ep26) →
  0.592 (ep32, step 86600, val ckpt) — 持续上升, 未平台。
- val_loss: ep18 起 ~20 平台 (19.5–21.5 震荡), medusa_ce 微升 (3.14→3.4)。
- PER (eval.py 100 样本): ep18–25 稳定 0.063–0.070。

**PER 假高诊断** (重要, 防误读): ep26 起 100 样本均值 PER 跳到 0.38–0.39,
但逐样本核查: idx=33 一条在贪心解码陷入**单 token 重复循环** (句尾无限发射
`z`, hyp 长 ~32 倍, 该条 PER=32.15), 中位数 PER 仅 0.041–0.052。模型整体
正常, 是 RNN-T 贪心解码鲁棒性问题 (predictor 进环不吐 blank), 后续可加
重复守卫; 不影响接受率结论的有效性。

**全量 test-clean 评估** (2620 样本, ckpt = ep32 val snapshot
`val-naive-h2-free-sl-post-chain-disp10-epoch=32-step=86600`):
- PER **0.1077**; RTF 0.0705; 投机解码无损校验 0 失败。
- 接受率: h1 **0.5509**, h2 0.2614, overall 0.4062 (drafts h1=76814)。
- 对照: 从头 44 epoch 即达 iter-1 (0.41 overall) 水平; 距 90% h1 目标尚远,
  h1 曲线仍在上升, 训练停于 ep44 (用户叫停), 后续可续训观察。

**显示指标** (eval_display, z-loop 防护后; 训练停在 ep44):

口径 A — 100 样本 (`artifacts/logs/iter10_eval_ep32.log`, ep32 val ckpt step 86600,
lm 0.1/bnt 0.2, `loop_skipped=1`):

| 头 | disp | abst | hit% | repl% | leadP50/P90/avg (ms) | value (ms) | LM gap |
|---|---|---|---|---|---|---|---|
| h1 | 8609 | 0 | **53.72** | 46.28 | 40/40/31 | 16.5 | +39.74 (LM 13.99) |
| h2 | 8609 | 0 | **34.56** | 65.44 | 40/80/59 | 20.3 | +32.51 (LM 2.04) |

口径 B — 全 test-clean 2620 样本 (`artifacts/logs/iter10_full_display.log`, 同 ckpt,
`loop_skipped=2`, 2618 句 / ref_tokens 190111):

| 头 | disp | abst | hit% | repl% | leadP50/P90/avg (ms) | value (ms) | LM gap |
|---|---|---|---|---|---|---|---|
| h1 | 190103 | 8 | **54.55** | 45.45 | 40/40/32 | 17.3 | +38.65 (LM 15.89) |
| h2 | 190103 | 0 | **34.91** | 65.09 | 40/80/60 | 21.0 | +32.35 (LM 2.56) |

与 eval.py 接受率 (h1 0.5509) 交叉印证一致。**注意**: 修复前的
eval_display 数字 (h1 8.63%) 是 z-loop 样本稀释的假值, 已作废。

结论 (对照 §7.2 锚点):
- **弃权问题反转**: iter-9 弃权 ~59% → iter-10 几乎 0% — BNT δ=0 监督
  生效但过头, 头从不开口弃权, 低置信草稿也全上屏 (h1 repl% 45%)。
- **hit% 显著抬升**: 43.01 → 54.55 (全量), gap 保持 +38 (头不是 LM)。
- **value 反而回落** (34.0→17.3): 命中草稿平均提前量 32ms (iter-9 ~79ms)
  — 头学会「贴着验证器当前决策」发言, 提前看到正确 token 但只提前 1 帧。
  下一步杠杆: 加强 δ>0 帧「token 快来了」监督 (升 bnt_weight 或调 BNT
  标签分布) / 权重扫描 (文档 §6.2), 把 lead 拉长。
- 头间分化: h1 偏保守 (hit 54.6, lead 短), h2 偏激进 (hit 34.9, lead 长),
  链式接受率 h2 0.26 与此一致。

---

## 维护规则

1. 每轮实验: 先在此追加/更新对应小节与总览行, 再 commit
   (`docs(experiment): ...`)。
2. 数字必须带口径 (100 样本 / 全 test-clean) 与 ckpt 标识 (epoch/step)。
3. 失败实验同样记录 (iter-5 模式) — 负结论防止重复踩坑。
