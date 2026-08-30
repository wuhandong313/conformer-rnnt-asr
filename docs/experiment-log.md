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
| iter-8h2 | 推理窗修复 (a56ab42) + 2 头重训 | — | — | — | — | — | 进行中 |

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

---

## 维护规则

1. 每轮实验: 先在此追加/更新对应小节与总览行, 再 commit
   (`docs(experiment): ...`)。
2. 数字必须带口径 (100 样本 / 全 test-clean) 与 ckpt 标识 (epoch/step)。
3. 失败实验同样记录 (iter-5 模式) — 负结论防止重复踩坑。
