#!/usr/bin/env python3
"""iter-10 冒烟测试 — 小随机模型, CPU, 无需数据集。

验证:
  1. 训练前向 + 三条新损失 (shifted_ce / KL(chain_align) / LM_aux / BNT)
     反传; 零初始化注入 ⇒ 初始 head logits == base; freeze_base 下梯度
     只落头侧参数 (encoder/predictor/joiner/w_vocab 无梯度); seq 模型
     梯度在注入投影非零后可达 (零初始化 bootstrap 两步验证)。
  2. 链式投机解码与 greedy oracle 逐 token 一致 (无损) — 随机权重下
     draft 几乎必错, 走 fallback + cache 回滚重放路径 (含/不含 seq 模型)。
  3. oracle replay 状态推进不变量。
  4. 接受率统计健全性。
  5. TF/推理同构: 全网格 head logits@(t,l) == 单点 predict_heads
     (KV/状态 cache 吃同前缀 token) — transformer 与 lstm 两种 seq 模型。
  6. seq 模型 cache: 增量 == 全序列; 快照回滚 == 全新 cache。
  7. predict_heads 拒绝宽度 > 1 输入 (头无声学窗)。
  8. iter-9: 3 头 ckpt → 2 头 warm-start 只丢 per-head 索引 >= 2 的键;
     iter-10: 旧 ckpt 的 context_acoustic / win_attn 键静默丢弃。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from conformer_rnnt.medusa import (
    MedusaHeads,
    base_next_token_ce_loss,
    build_draft_seq_model,
    kl_to_base_loss,
    lm_aux_loss,
    shifted_ce_loss,
)
from conformer_rnnt.spec_decode import greedy_decode, speculative_greedy_decode

SEQ_TF = {"type": "transformer", "num_layers": 2, "d_model": 16, "nhead": 2,
          "ffn_dim": 32, "dropout": 0.0, "emb_dim": 8}
SEQ_LSTM = {"type": "lstm", "num_layers": 2, "hidden_dim": 8, "dropout": 0.0,
            "emb_dim": 8}


def build_tiny_model(K: int, chain: bool, seq_spec=None):
    from conformer_rnnt.model import conformer_rnnt_model

    torch.manual_seed(0)
    model = conformer_rnnt_model(
        input_dim=8,
        encoding_dim=16,
        num_symbols=5 + 1,  # 5 symbols + blank
        right_context_length=0,
        time_reduction_input_dim=8,
        time_reduction_stride=1,
        num_heads=2,
        ffn_dim=16,
        num_layers_block1=1,
        num_layers_block2=1,
        dropout=0.0,
        depthwise_conv_kernel_size=3,
        compression_filter="none",
        symbol_embedding_dim=8,
        num_lstm_layers=1,
        lstm_hidden_dim=8,
        lstm_layer_norm=True,
        lstm_dropout=0.0,
        joiner_activation="relu",
        medusa_num_heads=K,
        medusa_input_mode="stream_local",
        medusa_draft_after_emit=True,
        medusa_chain_states=chain,
        medusa_draft_seq_model=seq_spec,
    )
    return model


# 测试 8 需要构造 LightningModule, 而它构造时就加载 sp 模型 / 全局统计量。
# 二者只在数据集侧存在 (本地无数据集时跳过该测试)。
LIT_SP_MODEL = os.environ.get("SP_MODEL_PATH", "/test21/medusa/spm_bpe_74.model")
LIT_GLOBAL_STATS = os.environ.get(
    "GLOBAL_STATS_PATH", "/test21/medusa/global_stats.json"
)


def build_tiny_lit(K: int):
    """极小 LightningModule (仅用于 warm-start 键过滤测试)。"""
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule

    return LibriSpeechRNNTModule.from_config(
        {
            "paths": {
                "sp_model_path": LIT_SP_MODEL,
                "global_stats_path": LIT_GLOBAL_STATS,
            },
            "model": {
                "input_dim": 8,
                "encoding_dim": 16,
                "num_symbols": 5,
                "time_reduction_input_dim": 8,
                "time_reduction_stride": 1,
                "right_context_length": 0,
                "num_heads": 2,
                "ffn_dim": 16,
                "num_layers_block1": 1,
                "num_layers_block2": 1,
                "depthwise_conv_kernel_size": 3,
                "dropout": 0.0,
                "compression": {"filter": "none"},
                "predictor": {
                    "symbol_embedding_dim": 8,
                    "num_lstm_layers": 1,
                    "lstm_hidden_dim": 8,
                    "lstm_layer_norm": True,
                    "lstm_dropout": 0.0,
                },
                "joiner": {"activation": "relu"},
                "medusa": {
                    "num_heads": K,
                    "input_mode": "stream_local",
                    "draft_after_emit": True,
                    "chain_states": True,
                },
            },
            "decoding": {"spec": {"per_token_blank": 4}},
        }
    )


def fake_batch(B=1, T=12, L=6, V=6):
    torch.manual_seed(1)
    feats = torch.randn(B, T, 8)
    feat_lens = torch.full((B,), T, dtype=torch.int32)
    targets = torch.randint(0, V - 1, (B, L), dtype=torch.int32)
    tgt_lens = torch.full((B,), L, dtype=torch.int32)
    return feats, feat_lens, targets, tgt_lens


def freeze_base(model):
    """复刻 lightning freeze_base: 冻结全部, 只解冻头侧 (w_vocab 除外)。"""
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.medusa_heads.named_parameters():
        if not n.startswith("w_vocab."):
            p.requires_grad = True


def head_grads(model):
    return {
        n: (p.grad is not None and p.grad.abs().sum().item() > 0)
        for n, p in model.medusa_heads.named_parameters()
        if "w_vocab" not in n
    }


def main():
    K = 3
    V = 6
    blank = V - 1
    print("=== 1. training forward + losses + zero-init + grad scope ===")
    model = build_tiny_model(K, chain=True, seq_spec=SEQ_TF)
    freeze_base(model)
    model.eval()
    feats, feat_lens, targets, tgt_lens = fake_batch(V=V)
    prep = targets.new_empty((targets.size(0), targets.size(1) + 1))
    prep[:, 1:] = targets
    prep[:, 0] = blank
    logits, src_len, _, base, future, logits_lm = model.forward_train_aux(
        feats, feat_lens, prep, tgt_lens + 1
    )
    assert base.shape == logits.shape
    assert len(future) == K
    assert logits_lm is not None, "seq model configured → lm logits required"
    # 零初始化注入 (acoustic_frame_proj/draft_state_proj/chain_state_proj/
    # context_mlp 输出层) ⇒ 每个 head 初始 logits == base。
    for k in range(K):
        assert torch.allclose(future[k], base, atol=1e-5), (
            f"head-{k+1} must equal base at init (zero-init injection)"
        )
    print(f"  shapes OK; base {tuple(base.shape)}; all {K} heads ≈ base at init: OK")

    ce = shifted_ce_loss(future, targets, [0.8] * K, tgt_lens, src_len,
                         blank_idx=blank, future_state_offset=1)
    kl = kl_to_base_loss(future, base, 0.05, src_len, tgt_lens,
                         future_state_offset=1, chain_align=True)
    lm = lm_aux_loss(logits_lm, prep, tgt_lens + 1)
    acc = base_next_token_ce_loss(future, base, [0.8] * K, tgt_lens, src_len,
                                  blank_idx=blank, future_state_offset=1,
                                  max_blank=4, chain_align=True)
    total = ce + kl + lm + acc
    total.backward()
    # 反传范围: 头侧参数有梯度; encoder/predictor/joiner(w_vocab) 无梯度。
    # step-0 语义: 零初始化的*输出*投影 (context_mlp[-1]/res_blocks[..].linear2/
    # chain_draft_proj/...) 有梯度 (bootstrap 起点); 上游模块 (acoustic_frame_
    # proj / draft_state_proj / seq 模型) 的梯度此时被零输出层挡住, 在下一节
    # bootstrap 步后可达。
    grads = head_grads(model)
    assert any(grads.values()), "medusa head params received no gradient"
    assert grads.get("context_mlp.2.weight"), (
        "context_mlp output layer got no gradient (bootstrap entry broken)"
    )
    assert grads.get("res_blocks.0.linear2.weight"), (
        "res_block output projection got no gradient"
    )
    for p in model.transcriber.parameters():
        assert p.grad is None, "encoder received gradient (freeze broken)"
    for p in model.predictor.parameters():
        assert p.grad is None, "predictor received gradient (freeze broken)"
    assert model.joiner.linear.weight.grad is None, "w_vocab received gradient"
    print(f"  ce={ce:.4f} kl={kl:.4f} lm={lm:.4f} acc={acc:.4f}; "
          "grads stay head-side: OK")

    # Bootstrap 步: 零初始化投影更新为非零后, 梯度应可达注入通道与 seq 模型。
    with torch.no_grad():
        model.medusa_heads.context_mlp[-1].weight.normal_(0, 0.1)
        model.medusa_heads.context_mlp[-1].bias.normal_(0, 0.1)
        model.medusa_heads.draft_state_proj.weight.normal_(0, 0.1)
        model.medusa_heads.draft_lm_proj.weight.normal_(0, 0.1)
    model.zero_grad()
    _, _, _, base2, future2, lm2 = model.forward_train_aux(
        feats, feat_lens, prep, tgt_lens + 1
    )
    (shifted_ce_loss(future2, targets, [0.8] * K, tgt_lens, src_len,
                     blank_idx=blank, future_state_offset=1)
     + lm_aux_loss(lm2, prep, tgt_lens + 1)).backward()
    grads2 = head_grads(model)
    for name in ("acoustic_frame_proj.weight", "draft_state_proj.weight",
                 "draft_lm_proj.weight", "chain_state_proj.weight"):
        assert grads2.get(name), f"{name} got no gradient after bootstrap"
    assert any(g for n, g in grads2.items() if n.startswith("seq_model.")), (
        "seq model received no gradient after bootstrap step"
    )
    print("  injection channels + seq model gradients reachable after "
          "bootstrap: OK")
    model.zero_grad()

    print("=== 2. chain decode lossless (random weights → fallback/replay) ===")
    # 随机权重模型会幻觉 (同帧无限 emit), 用 max_steps 把两者截到可比长度。
    MAX = 200
    for label, m in (("with-seq", model), ("no-seq", build_tiny_model(K, chain=True))):
        oracle = greedy_decode(m, feats, feat_lens, blank, max_steps=MAX)
        spec, stats = speculative_greedy_decode(
            m, feats, feat_lens, blank,
            max_draft=K, per_token_blank=4, max_steps=MAX * 4, return_stats=True,
        )
        # spec 每步最多产出 1+K 个 token; 两者按公共前缀比较 (幻觉截断点不同
        # 不影响无损性判断 — 前 MAX 步的 oracle 输出必须是 spec 的前缀)。
        assert spec[: len(oracle)] == oracle, (
            f"chain spec decode ({label}) must equal greedy oracle"
        )
        print(f"  [{label}] tokens={len(spec)} drafts={stats['total_drafts']} "
              f"accepted={stats['total_accepted']} lossless: OK")

    print("=== 3. oracle replay state advancement ===")
    from conformer_rnnt.spec_decode import (
        _advance_predictor,
        _init_decode,
        _join_argmax,
        _make_tok,
    )

    enc, _ = model.transcribe(feats, feat_lens)
    T = enc.size(1)
    oracle = greedy_decode(model, feats, feat_lens, blank, max_steps=MAX)
    g, st, one = _init_decode(model, blank, feats.device)
    t = 0
    consumed = 0
    while t < T and consumed < len(oracle):
        p = _join_argmax(model, enc, t, g, one)
        if p == blank:
            t += 1
            continue
        g, st = _advance_predictor(model, _make_tok(p, feats.device), st, one)
        consumed += 1
    assert consumed == len(oracle), f"replay consumed {consumed} != {len(oracle)}"
    print("  oracle replay state advancement: OK")

    print("=== 4. acceptance stats sanity ===")
    per_head = stats["per_head"]
    assert len(per_head) == K
    for j, ph in enumerate(per_head):
        assert ph["drafts"] >= ph["accepted"], "accepted cannot exceed drafts"
        if ph["drafts"] == 0 and ph["accepted"] == 0:
            print(f"  head-{j+1}: no drafts (early stop)")

    print("=== 5. TF/inference isomorphism (transformer + lstm seq) ===")
    # 全网格 head-k logits@(t,l) 必须等于单点 predict_heads:
    #   chain_states[k] = tgt_enc 列 l+k, chain_hiddens[k] = h_seq 列 l+k,
    #   chain_drafts = 前序 head 在 (t,l) 的 argmax (与训练 prev_logits 同构)。
    for label, seq_spec in (("transformer", SEQ_TF), ("lstm", SEQ_LSTM)):
        m = build_tiny_model(K, chain=True, seq_spec=seq_spec)
        m.eval()
        fw, flw, tgw, tlw = fake_batch(B=1, T=12, L=6, V=V)
        prepw = tgw.new_empty((tgw.size(0), tgw.size(1) + 1))
        prepw[:, 1:] = tgw
        prepw[:, 0] = blank
        with torch.no_grad():
            _, _, _, base_g, fut_g, _ = m.forward_train_aux(
                fw, flw, prepw, tlw + 1
            )
            encw, _ = m.transcribe(fw, flw)
            tgt_enc, _, _ = m.predict(prepw, tlw + 1, None)
            h_seq = m.medusa_heads.seq_model.forward_full(prepw)
            t_probe, l_probe = 5, 1
            assert l_probe + K - 1 <= tgt_enc.size(1) - 1
            drafts = [
                int(fut_g[j][0, t_probe, l_probe].argmax(dim=-1).item())
                for j in range(K - 1)
            ]
            chain_states = [
                tgt_enc[:, l_probe + k, :].unsqueeze(1) for k in range(K)
            ]
            chain_hiddens = [
                h_seq[:, l_probe + k, :].unsqueeze(1) for k in range(K)
            ]
            _, fut_inf, _ = m.predict_heads(
                encw[:, t_probe : t_probe + 1], tgt_enc[:, l_probe : l_probe + 1],
                chain_drafts=drafts, chain_states=chain_states,
                chain_hiddens=chain_hiddens,
            )
            for k in range(K):
                grid = fut_g[k][0, t_probe, l_probe]
                point = fut_inf[k][0, 0, 0]
                assert torch.allclose(grid, point, atol=1e-4), (
                    f"{label}: head-{k+1} grid vs point mismatch "
                    f"(max {float((grid - point).abs().max()):.6f})"
                )
        print(f"  [{label}] grid == point at (t={t_probe}, l={l_probe}): OK")

    print("=== 6. seq cache: incremental == full; rollback == fresh ===")
    for label, spec in (("transformer", SEQ_TF), ("lstm", SEQ_LSTM)):
        seq = build_draft_seq_model(num_vocab=V, output_dim=16, spec=spec)
        tokens = torch.randint(0, V - 1, (1, 7))
        h_full = seq.forward_full(tokens)
        h, cache = seq.init_cache(tokens[:, :5])
        h, cache = seq.forward_step(tokens[:, 5:6], cache)
        assert torch.allclose(h[0, 0], h_full[0, 5], atol=1e-5), (
            f"{label}: incremental step 5 != full-sequence hidden"
        )
        # 回滚: 快照于消费 0..5 之后; 走错一步; 恢复; 再走错误 token ==
        # 全新 cache 走同一步。
        snapshot = seq.clone_cache(cache)  # after tokens 0..5
        h, cache = seq.forward_step(tokens[:, 6:7], cache)
        assert torch.allclose(h[0, 0], h_full[0, 6], atol=1e-5), (
            f"{label}: incremental step 6 != full-sequence hidden"
        )
        cache = seq.restore_cache(snapshot)
        wrong_tok = (tokens[:, 6:7] + 1) % (V - 1)
        h_rb, cache_rb = seq.forward_step(wrong_tok, cache)
        h_fresh, cache_fresh = seq.init_cache(tokens[:, :6])
        h_fresh, cache_fresh = seq.forward_step(wrong_tok, cache_fresh)
        assert seq.cache_length(cache_rb) == seq.cache_length(cache_fresh), (
            f"{label}: rollback cache len {seq.cache_length(cache_rb)} != "
            f"fresh cache len {seq.cache_length(cache_fresh)}"
        )
        assert torch.allclose(h_rb[0, 0], h_fresh[0, 0], atol=1e-6), (
            f"{label}: rollback cache != fresh cache"
        )
        print(f"  [{label}] incremental==full & rollback==fresh: OK")

    print("=== 7. predict_heads rejects window width > 1 ===")
    g0, _, _ = _init_decode(model, blank, feats.device)
    try:
        model.predict_heads(enc[:, :2], g0)
    except ValueError:
        print("  width-2 input rejected: OK")
    else:
        raise AssertionError("predict_heads accepted a multi-frame window")

    print("=== 8. warm-start key filter (3→2 heads) + stale module keys ===")
    if not (
        os.path.exists(LIT_SP_MODEL) and os.path.exists(LIT_GLOBAL_STATS)
    ):
        print(f"  SKIPPED (no sp model / global stats at {LIT_SP_MODEL})")
        print("ALL SMOKE TESTS PASSED")
        return
    # Regression test: the stale-head drop used to take the first numeric
    # segment after `medusa_heads.`, which also matched *layer* indices of
    # the shared MLPs (context_mlp.2, context_mlp_extra.2/4). Those trained
    # weights were then silently reset to their zero-init value on every
    # 3 -> 2 head warm-start (iter-8h2 and the iter-9 baseline both hit it).
    torch.manual_seed(0)
    src3 = build_tiny_lit(3)
    sd3 = src3.state_dict()
    # Sentinel values: preserved keys must load, dropped keys must not.
    sentinels = {
        "model.medusa_heads.res_blocks.1.linear1.weight": 3.0,   # kept head
        "model.medusa_heads.context_mlp.2.weight": 7.0,          # shared MLP
        "model.medusa_heads.context_mlp_extra.2.weight": 6.0,    # shared MLP
        "model.medusa_heads.context_mlp_extra.4.weight": 5.0,    # shared MLP
        "model.medusa_heads.res_blocks.2.linear1.weight": 9.0,   # stale head
        "model.medusa_heads.context_mlp_heads.2.0.weight": 8.0,  # stale head
    }
    sd3 = {
        k: (torch.full_like(v, sentinels[k]) if k in sentinels else v.clone())
        for k, v in sd3.items()
    }
    # iter-10: 旧 ckpt 携带的已删模块键 (窗投影 / iter-5 attention) 须被
    # 静默丢弃, 不得报 unexpected key。
    stale_modules = {
        "model.medusa_heads.context_acoustic.weight": torch.zeros(512, 4608),
        "model.medusa_heads.win_attn_key.weight": torch.zeros(128, 16),
    }
    sd3.update(stale_modules)
    dst2 = build_tiny_lit(2)
    dst2.load_state_dict(sd3, strict=True)
    got = dst2.state_dict()
    # Shared-MLP layer indices must survive (the bug zeroed them).
    for key in (
        "model.medusa_heads.context_mlp.2.weight",
        "model.medusa_heads.context_mlp_extra.2.weight",
        "model.medusa_heads.context_mlp_extra.4.weight",
        "model.medusa_heads.res_blocks.1.linear1.weight",
    ):
        want = sentinels[key]
        assert torch.allclose(got[key], torch.full_like(got[key], want)), (
            f"{key} was dropped/reinitialized on warm-start (expected all {want})"
        )
    # Genuine per-head keys with index >= num_heads must be dropped — the
    # 2-head model has no such parameters at all.
    for key in (
        "model.medusa_heads.res_blocks.2.linear1.weight",
        "model.medusa_heads.context_mlp_heads.2.0.weight",
        "model.medusa_heads.chain_state_proj_heads.2.weight",
    ):
        assert key not in got, (
            f"{key} exists in the 2-head model — stale head keys were not "
            "dropped (or the head count is wrong)"
        )
        assert key in sd3, f"test setup: {key} missing from the 3-head source"
    for key in stale_modules:
        assert key not in got, f"{key} must not exist on the iter-10 module"
    print("  kept shared MLP layers + dropped stale head/module keys: OK")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
