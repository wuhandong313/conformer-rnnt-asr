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

## 复现

```bash
# 贪心
python eval.py configs/server_eval_greedy.yaml \
    --checkpoint /test21/conformer-rnnt-asr/exp/checkpoints/val-epoch=42-step=26700.ckpt
# beam-20
python eval.py configs/server_eval_beam20.yaml \
    --checkpoint /test21/conformer-rnnt-asr/exp/checkpoints/val-epoch=42-step=26700.ckpt
```
