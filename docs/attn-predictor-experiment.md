# attn-predictor 实验 (Family A — MSA-Transducer 风格)

## 动机

把 RNNT predictor 的 LSTM trunk 换成 4 层受限因果自注意力 Transformer
(`predictor.type="attn"`, d_model 256 / 4 heads / ffn 1024 / left_context
1024 / learned 绝对位置)，观察其 PER 与 Medusa 草稿配合的潜力。

## 训练

- 配置: `configs/server.yaml` (predictor.type=attn, predictor_lr_scale 0.5)
- 从头训练 (from_checkpoint: null)，120 epochs，exp_dir 复用主仓库
  `/test21/conformer-rnnt-asr/exp`
- 44.31M 参数 (predictor trunk 4.38M)，warmup 10000 步
- val_loss 最优 **18.37 @ val-epoch=42-step=26700**，之后过拟合

## 解码期 bug (训练正常、仅推理触发，已修复于 2d873f8)

1. `PredictorAttn.forward` 忽略 state — torchaudio beam search 的 decode
   契约是 `model.predict → predictor.forward` 带状态单步；旧实现每步把
   末 token 重新 embed 在位置 0（无历史），logits 恒定 → 贪心解码首位
   token 无限循环。修复: forward 带 state 且单步时委托 `predict()`。
2. 变长 KV cache 无法过 beam-search 批处理 — torchaudio 同帧内对 a-hypo
   续扩，不同 hypo 消费 token 数不同，`_batch_state` 无法 cat。修复:
   定长 `left_context` 循环 cache（写槽位 `p % left_context`，有效性掩码
   由 p 推导，P (batch,) 逐行携带绝对位置）。

冒烟测试 (`scripts/smoke_attn.py`) 覆盖: full==incremental 等价（含截断
区间）、混合 token 数 hypo 批处理等价、index_select 切片续解码、超
max_len 显式报错、JIT 契约。

## 结果 (全量 test-clean 2620 样本, ckpt val-epoch=42-step=26700)

| 解码 | PER | RTF |
|---|---|---|
| greedy (beam_width=1) | **0.1112** | 0.0435 |
| beam-20 | **0.1069** | 0.1481 |

对照 (同为全量 test-clean, beam-20):
- iter-10 (LSTM predictor + Medusa, 从头训练, ep32): PER 0.1077
- LSTM warm-start 基线 (iter-4): PER 0.073

结论: attn trunk 从头训练即可与 iter-10 从头训练持平 (0.1069 vs 0.1077)，
但仍显著落后于 warm-start LSTM 基线 (0.073)。作为 Medusa 草稿 backbone
的价值需后续在该 trunk 上接 Medusa 头再评估。

## attn trunk × Medusa 多头架构 (attn-multihead 分支, 2026-10-03)

从 attn-predictor 拉出 `attn-multihead` 分支，把 Medusa 头接到 attn trunk 上，
架构链路补齐并修了两个对齐 bug。

### 架构

- trunk: `PredictorAttn` (受限因果自注意力, KV cache 定长循环写槽位)，
  输出 g (B, U+1, 512)，blank 前置 (`[blank, y_0..y_{U-1}]`)。
- 头: `MedusaHeads` — head 0 = trunk 直投影 (由 RNN-T loss 经 joiner 训，
  aux 侧只作 KL 的 detach 锚)；head k = ResBlock_k(g) → 共享 w_vocab，
  零初始化 ⇒ 初始 draft ≡ base 分布。共享 512→V 投影 (Medusa 论文同款)。
- 训练: `forward_train_aux` 输出 (logits_base, [logits_1..logits_K])，
  aux 损失 = Σ λ_k·shifted_CE + β·KL(head_k ‖ head_0.detach)，
  λ_k=0.8^k、β=0.05，已接入 `lightning_module._step`。
- 解码: `spec_decode.speculative_greedy_decode` — 无损自投机贪心，草拟
  K=2 (h1/h2, 钉死决策)，head-0 argmax 逐步验证。

### 对齐契约 (两个 bug 的根因)

`medusa.shifted_ce_loss` 的语义: head k 在列 l (trunk 已消费
`[blank, y_0..y_{l-1}]`) 预测 **y_{l+k}**。由此：

1. **草拟时机 = draft_before_emit**: spec 解码必须在**消费刚 emit 的
   pred0 之前**查询头——此时 d_1 = y_{u+1} 恰是下一个 emit。旧实现在
   消费 pred0 之后草拟 (post-emit)，草稿系统性偏移一位，接受率趋 0。
   (multihead 分支 iter-10 的 draft_after_emit 模式是配合重写的头语义，
   与本分支瘦身头不兼容。)
2. **验证扫描边界**: 旧验证循环 `t < T-1` 会在最后一帧跳过 emit 检查
   (oracle 在 t=T-1 仍会 emit)，spec 输出少于 oracle。已改为 `t < T`
   逐步复刻 oracle；blank 预算 (per_token_blank) 耗尽即弃权，外层从同
   (t, g_last) 继续，无损性不受影响。

### 冒烟 (`scripts/smoke_multihead.py`, CPU)

zero-init 头≡base 等价 / shifted_CE 对齐钉死 (one-hot 构造, 对齐正确
loss≈0 偏移一位 loss>1) / aux 损失反传 (梯度到达 res_blocks + attn
trunk) / spec-decode 无损 (13 个 blank-biased 随机 seed 逐 token 等于
oracle) / **oracle 喂样满接受验收** (头恒草拟真实未来 token ⇒ R=2 轮
吞掉 M=6 个 emit, R≈ceil(M/3)，若草拟偏移一位则 R≈M) / config factory。
注意: 测试模型必须 `attn_dropout=0.0` + `model.eval()`，否则 dropout
采样使解码逐次不一致 (排查时踩过)。

### 训练配置

`configs/server_attn_multihead.yaml` — 与 server.yaml 仅两处差异:
`medusa.num_heads: 2`、`exp_dir: /test21/conformer-rnnt-asr/exp_mh`
(与单头基线 ckpt 隔离)。

## 复现

```bash
# 贪心
python eval.py configs/server_eval_greedy.yaml \
    --checkpoint /test21/conformer-rnnt-asr/exp/checkpoints/val-epoch=42-step=26700.ckpt
# beam-20
python eval.py configs/server_eval_beam20.yaml \
    --checkpoint /test21/conformer-rnnt-asr/exp/checkpoints/val-epoch=42-step=26700.ckpt
```
