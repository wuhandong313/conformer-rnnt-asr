# iter-10 实验文档：删头的声学窗 + draft 侧因果序列模型 + BNT 蒸馏

本文档是 iter-10 的完整操作手册：架构改了什么、每个配置键什么含义、
怎么准备 warm-start、怎么在服务器上训练与监控、怎么调参、怎么 eval 出数。

---

## 1. 背景与动机（iter-9 诊断结论）

iter-9 用 100 样本 eval_display（tag `disp`）得到三个诊断事实：

| 事实 | 数字 | 含义 |
|---|---|---|
| 预测区命中率为常数 | h1 43.01% / h2 42.85% | 与头容量无关，瓶颈不在头本身 |
| LM 先验几乎不可用 | 2.18% / 0.64% | 头依赖的 frozen predictor 只有 2 层×128 LSTM，从不是好 LM |
| 弃权率高 | ~59% | 头没学会「何时开口」 |

架构反思：**头不应有自己的声学窗**。解码到哪一帧就条件于哪一帧的编码
特征；上下文感知是 encoder chunk 注意力（left/segment/right context）的
职责。iter-8 的 `_anchor_window`（9 帧锚定窗）是补丁，iter-10 整体删除。

序列建模增强转向 **draft 侧可训练模块**（因果 transformer 首测，
deep LSTM 备选）；弃权/时机问题用 **BNT 蒸馏**（蒸馏验证器的发射决策）
攻击。范围约束：不动 base、不动 encoder、只训头（`freeze_base: true`）。

## 2. 架构改动

### 2.1 删除（随窗一并退役）

- `_anchor_window`（medusa.py）与 9 帧窗布局
- `context_acoustic`（4608→512 输入投影）——ckpt 里的旧键加载时静默丢弃
- `window_prebuilt`、`past_frames` / `lookahead_frames`（头侧窗概念）
- `win_attn_*`（iter-5 已禁用的占位）、rhythm 占位
- spec_decode 推理侧的窗拼装；`predict_heads` 现在只接受 **(B,1,D) 单帧**
  `enc[:, t:t+1]`，宽度 >1 直接 raise

### 2.2 新增：draft 侧因果序列模型 `seq_model`

配置键 `model.medusa.draft_seq_model`，两种实现（输出接口一致）：

- **transformer**（首测）：emb 256→512，3 层 pre-norm 因果 self-attn
  （8 头）+ FFN 1024 (SiLU)，learned position embedding（max_len 4096），
  ≈9.5M 参数。训练走 TF 全序列（`forward_full`），推理走 KV cache 单步
  （`forward_step`）；快照 = 浅拷贝（cat 产生新张量），回滚 = 恢复引用。
- **lstm**（备选）：4 层×256 LayerNorm LSTM + 256→512 投影，≈2.4M 参数。
  推理隐状态 tuple 即 cache，回滚 = 深拷贝快照恢复。

两条路径的数值同构性由冒烟测试 5/6 保证（见 §8）。

### 2.3 融合公式与零初始化注入

```
fused = context_predictor(tgt_enc)            # 冻结
      + context_base_logits(base_logits.detach())  # 冻结
      + chain_state_proj(rolled)              # LSTM rolled 状态通道（可训）
      + draft_state_proj(h_seq)               # ← 新，零初始化
      + acoustic_frame_proj(enc[t])           # ← 新，零初始化
        → context_norm → context_mlp* → res_blocks → w_vocab（冻结）
```

零初始化 ⇒ warm-start 后 step-0 与旧头逐位等价（冒烟测试 1 验证）。
step-0 梯度只达零初始化*输出*投影（bootstrap 入口：context_mlp[-1]、
res_blocks.linear2），一步更新后上游（acoustic_frame_proj /
draft_state_proj / seq_model）梯度可达——这是标准零初始化 bootstrap
语义，不是 bug。

### 2.4 推理侧 cache 管理（spec_decode.py）

- BOS 约定与 `_init_decode` 对齐（blank token 进 cache）。
- emit 后：快照 cache → 草拟；draft 侧 cache **总是**随草拟推进
  （含未验证 draft）， predictor rolled 状态仍只在接受路径推进。
- 验证结束后对账：`committed == cache_appended`（全接受）免回滚；
  否则 restore 快照 + 重放已提交 token（纯 token 侧，O(1) 截断）。

### 2.5 保留（ep07 热载）

res_blocks、context_mlp / context_mlp_extra / context_mlp_heads、
context_predictor、context_base_logits、chain_draft_embed/proj、
chain_state_proj(+heads)、w_vocab 共享（冻结）。

## 3. 损失设计

对齐律（三条损失共享的不变量）：**head-k（k=1..K）在网格列 l 条件于
「消费完 y_{l+k-1} 的状态」，预测 y_{l+k}**。三通道同构：LSTM rolled
切片、seq 模型因果隐藏态 `h_seq[l+k-1]`、推理 chain_hiddens。

```
loss = RNNT (权重 1.0, clamp)                      # base 验证器，不经头
     + Σ_k w_k·CE_k   shifted_ce [0.8, 0.64]       # 主头损失, offset=1
     + lm_weight  · LM_aux                         # 0.1，见下
     + bnt_weight · BNT                            # 0.2，见下
     + kl_weight  · KL                             # 0（保留不启用）
```

1. **LM_aux**（新）：纯 token 侧 next-token CE，
   `logits_lm[l] = w_vocab(draft_lm_proj(h_seq[l]))`，目标 `y_{l+1}`。
   直接攻击 2.18% LM 短板，给 seq 模型独立于声学通路的监督。
   `draft_lm_proj` 零初始化、w_vocab 冻结 ⇒ 梯度只进 proj + seq 模型。
2. **BNT**（`base_next_token_ce_loss`，chain_align=True）：
   head-k 在 (t, l) 的标签 = `base_ids[t..t+bnt_max_blank, l+k-1]` 中首个
   non-blank argmax——即**验证器从这个状态出发 ≤4 帧内会发射什么**，
   正是「接受事件」本身。δ=0 帧教头匹配验证器当前决策（治弃权）；
   δ>0 帧教头「token 快来了」（提前量来源）。found=False（句尾）无损失。
3. **KL**：`chain_align` 接线已修正（此前漏传会错对一列；权重 0 无影响）。
   维持 0（iter-1 结论：与 CE 目标冲突）。

梯度流（加性规则）：base_term detach、prev_logits detach（k≥1）、
encoder/predictor/w_vocab 冻结、RNNT 不经头 ⇒ 头侧加性注入不碰 base 数值。

## 4. 配置说明（configs/server_display.yaml）

`model.medusa` 节 iter-10 相关键：

| 键 | 值 | 含义 |
|---|---|---|
| `num_heads` | 2 | 固定 2 头（h1/h2） |
| `draft_seq_model.type` | transformer | 首测；训练不稳换 `lstm` |
| `draft_seq_model.num_layers/d_model/nhead/ffn_dim/dropout` | 3 / 512 / 8 / 1024 / 0.1 | transformer 结构 |
| `seq_lr_scale` | 0.5 | seq 模型单独参数组，LR = 0.5×主 LR（冷启动防震荡） |
| `lm_weight` | 0.1 | LM_aux 权重 |
| `bnt_weight` | 0.2 | BNT 蒸馏权重 |
| `bnt_max_blank` | 4 | BNT 前瞻帧数，必须 = `decoding.spec.per_token_blank` |
| `kl_weight` | 0.0 | 保持 0 |
| `tag_suffix` | disp10 | ckpt 文件名 tag，与历史 ckpt 区分（防 monitor 缓存串档） |
| `from_checkpoint` | BEST_iter10.ckpt | warm-start 源（= iter-9 ep07，见 §5.1） |

已删除键：`lookahead_frames` / `past_frames`（窗概念随架构删除，
旧配置文件里出现会直接被忽略——工厂不再接收）。

## 5. 训练流程（GPU 服务器）

服务器：`ssh internal`，`/test21/conformer-rnnt-asr`，
一律用 `conda env whd`（`/root/anaconda3/envs/whd/bin/python`）。

### 5.1 准备 warm-start 源

```bash
cd /test21/conformer-rnnt-asr
# iter-9 最优 ckpt (ep07, step=20400; val-*disp* 系列里最优) 复制为
# BEST_iter10.ckpt —— 按实际文件名 glob:
ls exp_iter2/checkpoints/val-*disp*epoch=07*
cp exp_iter2/checkpoints/<val-*disp*epoch=07*.ckpt> exp_iter2/BEST_iter10.ckpt
```

约定（同历代）：`BEST_iterN.ckpt` = 该迭代选定起点；iter-10 新模块
（seq_model/draft_state_proj/acoustic_frame_proj/draft_lm_proj）冷启动
零初始化，旧模块热载；ep07 携带的 `context_acoustic.*` / `win_attn_*`
键加载时自动丢弃，2 头配置不产生 stale per-head 键。

### 5.2 训练前自检（本地已过，服务器复跑一遍更稳）

```bash
/root/anaconda3/envs/whd/bin/python scripts/smoke_chain.py        # 全绿 (服务器上测试 8 也会跑)
/root/anaconda3/envs/whd/bin/python -m compileall -q train.py eval.py common.py conformer_rnnt
/root/anaconda3/envs/whd/bin/python eval.py configs/server_display.yaml   # 投机输出 == 贪心输出 (逐 token)
```

### 5.3 启动训练（screen `iter2-train`）

```bash
screen -dmS iter2-train bash -c 'cd /test21/conformer-rnnt-asr && \
  /root/anaconda3/envs/whd/bin/python train.py configs/server_display.yaml \
  2>&1 | tee exp_iter10_train.log'
```

### 5.4 启动监控（screen `iter10-watch`）

先**清 monitor 缓存**（monitor_iter2.py 每个 ckpt 只评一次并缓存 JSON，
同 tag/step 复用会报旧数字）：

```bash
rm -rf /test21/conformer-rnnt-asr/exp_iter2/monitor_eval/*
screen -dmS iter10-watch bash -c 'for i in $(seq 1 48); do \
  echo "=== $(date +%H:%M) ===" >> /test21/conformer-rnnt-asr/iter10_watch.log; \
  cd /test21/conformer-rnnt-asr && timeout 1900 /root/anaconda3/envs/whd/bin/python \
  scripts/monitor_iter2.py >> /test21/conformer-rnnt-asr/iter10_watch.log 2>&1; sleep 900; done'
```

注意 `scripts/monitor_iter2.py` 顶部 `TRAIN_LOG` 当前指向
`exp_iter8h2_train.log`，启动监控前改成 `exp_iter10_train.log`。

### 5.5 预期现象

- **warm-start 瞬态**：重启后头几个 val PER 会跳（fresh optimizer/LR
  warmup + 零 init 模块刚开始 bootstrap），属预期，自愈。
- step-0 附近 `Losses/*_medusa_lm` 从 ~ln(74)≈4.3 量级起步下降、
  `*_medusa_bnt` 从 ~2.9 量级下降——这两个曲线是 seq 模型是否在学的
  最直接信号。

## 6. 调参指南

### 6.1 看什么曲线，判断什么

| 现象 | 判断 | 动作 |
|---|---|---|
| lm_aux 降、bnt 降，但 hit% 不动 | seq 模型学成纯 LM | 降 `lm_weight` → 0.05；看 LM gap 指标 |
| bnt 与 shifted_ce 一升一降 | 两目标冲突 | 降 `bnt_weight` → 0.1 |
| transformer loss 震荡/发散 | 训练不稳 | 升 `seq_lr_scale` 降（0.25）；或换 `type: lstm` |
| 弃权率仍 >40% | BNT 力度不够 | 升 `bnt_weight` → 0.5 |
| hit% 全面低于 43% 基线 | 删窗后声学信息不足 | 启用应急补丁（见 §6.3） |

### 6.2 权重扫描（从主轮最优 ckpt warm-start 短链）

| 轮次 | lm_weight | bnt_weight |
|---|---|---|
| 主轮（默认） | 0.1 | 0.2 |
| 扫描 1 | 0.1 | 0.1 |
| 扫描 2 | 0.1 | 0.5 |
| 扫描 3 | 0.05 | 0.2 |

每次扫描：改 YAML → 复制主轮最优 ckpt 为新 `BEST_iter10.ckpt` 起点 →
`tag_suffix` 不变即可（monitor 记得再清一次 `monitor_eval/`）。

### 6.3 降级与应急

- **transformer → lstm**：`draft_seq_model: {type: lstm, num_layers: 4,
  hidden_dim: 256, dropout: 0.1}` 同位替换（接口一致，冒烟测试共用）。
- **应急补丁**（hit% 大幅低于 43% 时）：头侧保留单帧合法项 `enc[t+1]`
  （encoder right_ctx 提交后可得，非窗概念）作为额外注入通道。

## 7. Eval 流程

### 7.1 一级指标（scripts/eval_display.py）

```bash
/root/anaconda3/envs/whd/bin/python scripts/eval_display.py \
  configs/server_display.yaml 100
# 不带 ckpt 参数会自动选 exp_iter2/checkpoints/ 里最新的 val-iter10-disp10-*.ckpt
```

指标口径（每头 + 合并）：

- `disp`（上屏数）/ `abst`（blank 弃权数）——弃权率 = abst/(att)
- `hit%` = 命中/显示；`repl%` = 1−hit%（未命中即被真实 token 替换 = 闪烁）
- `leadP50/P90/P99/avg`：命中草稿的提前量（真实 emit 帧 − 草稿帧，×80ms）
- `value` = **hit% × 命中草稿平均提前量**（期望感知提前量，一级指标）
- LM 对照表：headHit% − LMhit% = gap。**gap 接近 0 ⇒ 头只是个 LM**；
  iter-9 基线 gap +40.84/+42.21。

### 7.2 观测锚点（100 样本，tag disp 口径）

| 指标 | iter-9 基线 | iter-10 目标 |
|---|---|---|
| h1/h2 hit%（预测区常数） | 43.01 / 42.85 | 显著抬升（transformer + 蒸馏主杠杆） |
| overall hit% | 75.1（ep07） | 80–85（90 留给 iter-11 lookahead） |
| 弃权率 | ~59% | 显著下降（BNT δ=0 监督） |
| LM gap | +40.84 / +42.21 | 不收窄（gap≈0 = 头退化成 LM） |
| value (ms) | 34.0/30.3（B 行口径） | 抬升 |

### 7.3 无损验证（eval.py）

解码路径改动必须验证投机输出与贪心 oracle 逐 token 一致：

```bash
/root/anaconda3/envs/whd/bin/python eval.py configs/server_display.yaml
```

（本地无数据集/无 pytorch_lightning，此项只能在服务器跑；本地已由
smoke_chain.py 测试 2 用随机模型覆盖同一条不变量。）

## 8. 冒烟测试清单（scripts/smoke_chain.py，CPU 无需数据集）

| 节 | 验证内容 |
|---|---|
| 1 | 训练前向 6 元组 + 四损失反传 + 零 init 等价（K 头 ≈ base）+ 梯度只落头侧 + bootstrap 两步可达性 |
| 2 | 链式投机解码无损（with-seq / no-seq 双模型，输出 == 贪心 oracle 前缀） |
| 3 | oracle replay 状态推进 |
| 4 | per-head 接受率统计 sanity |
| 5 | TF 全网格 vs 单点推理（KV cache 吃同前缀）同构（transformer + lstm） |
| 6 | cache 增量 == 全序列；回滚后 == 全新 cache（transformer + lstm） |
| 7 | `predict_heads` 拒绝宽度 >1 输入 |
| 8 | 3→2 头 warm-start 键过滤（共享 MLP 层索引保留、stale 头键丢弃）+ 旧窗模块键静默丢弃（需 sp 模型，服务器上跑） |

## 9. 结果记录规范

出数后在 `docs/experiment-log.md`：更新总览表 + 追加 iter-10 节。每条
数字必须带 (a) 样本量口径（100 样本 or 全 test-clean），(b) checkpoint
标识（epoch/step + tag），(c) 损失配置（lm/bnt 权重）。对照 §7.2 锚点写
结论：hit% 相对 43 常数动了多少、弃权率相对 59%、gap 是否保持。

## 10. 风险与后续

- 删窗后 enc[t] 信息不足 → §6.3 应急补丁（enc[t+1] 单帧注入）。
- transformer 训练不稳 → seq_lr_scale 降档或换 lstm。
- 曝光偏差平台期 → EAGLE 式 train-time test（TF 网格 1-token 扰动）。
- BNT 与 shifted_ce 冲突 → 权重扫描（§6.2）。
- **iter-11 杠杆**：overall 逼近 85 后叠加 lookahead 2–3（chunk 内零延迟
  合法）冲 90% 接受率。
