#!/usr/bin/env python3
"""iter-8 上限诊断 — 1 帧流式预算下的接受率天花板 (纯推理, 不训练)。

测量两个上限 (100 样本 test-clean, 与 monitor 同口径):

  A. sim-as-draft 上限: 用验证器模拟扫描 (窗口 = 1+lookahead 帧, 递归
     起点/blank 预算与 spec_decode 验证循环逐步一致) 解析出的 token 直接
     当 draft, 走真实验证循环统计接受率。这是「头完美读出 sim 特征」的
     接受率天花板; 同时报告窗口覆盖率 (found 比例)。
  B. 完全信息上限: draft = oracle 的真实后续 token (任何头都不可能超过
     这个)。差距 = 纯空白窗耗尽 (连续 blank > 预算) 造成的不可接受事件。

用法: python scripts/diag_rollout_ceiling.py <config> [max_samples]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from common import load_config, setup_logging
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
from conformer_rnnt.spec_decode import (
    _advance_predictor,
    _init_decode,
    _join_argmax,
    _make_tok,
)


def build_model(config, checkpoint=None):
    module = LibriSpeechRNNTModule.from_config(config)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        module.load_state_dict(state)
    module = module.to("cpu").eval()
    return module


@torch.no_grad()
def analyze_sample(model, features, feature_lengths, blank_idx, K, pb):
    """对单样本返回 (sim 统计, oracle 统计) 两个 per-head dict 列表。"""
    enc, enc_len = model.transcribe(features, feature_lengths)
    T = enc.size(1)

    sim_stats = [
        {"drafts": 0, "found": 0, "accepted": 0} for _ in range(K)
    ]
    oracle_stats = [
        {"drafts": 0, "accepted": 0} for _ in range(K)
    ]

    g_last, state, one = _init_decode(model, blank_idx, features.device)
    t = 0
    # oracle 未来 token 队列: 先整段 greedy 一遍记录 (帧, token) 序列。
    oracle_emits: list = []  # (frame, token)
    g_o, st_o, one_ = _init_decode(model, blank_idx, features.device)
    to = 0
    while to < T:
        p = _join_argmax(model, enc, to, g_o, one_)
        if p == blank_idx:
            to += 1
        else:
            oracle_emits.append((to, p))
            g_o, st_o = _advance_predictor(model, _make_tok(p, features.device), st_o, one_)
    next_oracle = 0  # 指向 oracle_emits 中「下一个未消费 token」

    out: list = []
    while t < T:
        pred0 = _join_argmax(model, enc, t, g_last, one)
        if pred0 == blank_idx:
            t += 1
            continue
        # 活解码与 oracle walk 严格同操作 ⇒ next_oracle 必指向 pred0。
        # 完全信息上限的 drafts = oracle 后续 K 个 token
        oracle_drafts = [
            oracle_emits[i][1]
            for i in range(next_oracle + 1, min(next_oracle + 1 + K, len(oracle_emits)))
        ]
        # 推进到 post-emit 状态 (draft_after_emit)
        g_last, state = _advance_predictor(model, _make_tok(pred0, features.device), state, one)

        # ---- sim 扫描 (验证器语义: 同帧递归 + blank 预算 pb) ----
        drafts: list = []
        sim_found: list = []
        prev_emit = t  # 相对绝对帧; head-0 从当前帧起
        g_sim, st_sim = g_last, state
        ok_chain = True
        for j in range(K):
            if not ok_chain:
                drafts.append(blank_idx)
                sim_found.append(False)
                continue
            resolved = None
            for d in range(pb + 1):
                tf = prev_emit + d
                if tf >= T:
                    break
                p = _join_argmax(model, enc, tf, g_sim, one)
                if p != blank_idx:
                    resolved = (p, d)
                    break
            if resolved is None:
                drafts.append(blank_idx)
                sim_found.append(False)
                ok_chain = False
                continue
            tok, d = resolved
            drafts.append(tok)
            sim_found.append(True)
            # 递归: 验证器接受后 t 不前进, 同帧继续; 状态滚动
            g_sim, st_sim = _advance_predictor(
                model, _make_tok(tok, features.device), st_sim, one
            )
            prev_emit = prev_emit + d

        # ---- 真实验证循环 (复刻 spec_decode) ----
        def verify(draft_list):
            nonlocal g_last, state, t
            v_g, v_st = g_last, state
            v_t = t
            accepted_flags = []
            for d in draft_list:
                if d == blank_idx:
                    accepted_flags.append(False)
                    continue
                blanks = 0
                emitted = None
                while blanks < pb and v_t < T - 1:
                    p = _join_argmax(model, enc, v_t, v_g, one)
                    if p == blank_idx:
                        v_t += 1
                        blanks += 1
                    else:
                        emitted = p
                        break
                if emitted is None:
                    accepted_flags.append(False)
                    break
                if emitted != d:
                    accepted_flags.append(False)
                    out.append(emitted)
                    v_g, v_st = _advance_predictor(
                        model, _make_tok(emitted, features.device), v_st, one
                    )
                    break
                accepted_flags.append(True)
                out.append(d)
                v_g, v_st = _advance_predictor(
                    model, _make_tok(d, features.device), v_st, one
                )
            else:
                pass
            # 把验证器终态写回外层 (与 spec_decode 一致: 失配提交/接受推进)
            g_last, state, t = v_g, v_st, v_t
            return accepted_flags

        sim_accept = verify(drafts)
        # oracle drafts: 重新走一遍验证 (需要从同一状态出发 — 上面 verify
        # 已推进状态, 故 oracle 上限改用解析法: 空白 run ≤ pb 即可接受)。
        # 统计 oracle 连续 blank gap:
        for j in range(len(oracle_drafts)):
            oracle_stats[j]["drafts"] += 1
            idx = next_oracle + 1 + j
            if idx >= len(oracle_emits):
                continue
            f_prev = oracle_emits[idx - 1][0]
            f_cur = oracle_emits[idx][0]
            if f_cur - f_prev <= pb:
                oracle_stats[j]["accepted"] += 1
        for j in range(K):
            sim_stats[j]["drafts"] += 1
            if j < len(sim_found) and sim_found[j]:
                sim_stats[j]["found"] += 1
                if j < len(sim_accept) and sim_accept[j]:
                    sim_stats[j]["accepted"] += 1
        next_oracle += 1
    return sim_stats, oracle_stats


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    config = load_config(sys.argv[1])
    setup_logging(config)
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    checkpoint = config.get("paths", {}).get("from_checkpoint")
    model = build_model(config, checkpoint)
    inner = model.model  # ConformerRNNT
    blank = model.blank_idx
    K = inner.medusa_heads.num_heads
    pb = model.spec_per_token_blank

    dataset = None
    dl = model.test_dataloader()  # 与 eval.py 同一条数据管线 (batch_size=1)
    agg_sim = [{"drafts": 0, "found": 0, "accepted": 0} for _ in range(K)]
    agg_orc = [{"drafts": 0, "accepted": 0} for _ in range(K)]
    n_done = 0
    for batch_tuple in dl:
        if n_done >= max_samples:
            break
        batch, _texts = batch_tuple
        sim_s, orc_s = analyze_sample(
            inner, batch.features, batch.feature_lengths, blank, K, pb
        )
        for j in range(K):
            for k in ("drafts", "found", "accepted"):
                agg_sim[j][k] += sim_s[j][k]
            for k in ("drafts", "accepted"):
                agg_orc[j][k] += orc_s[j][k]
        n_done += 1
        if n_done % 10 == 0:
            print(f"  .. {n_done}/{max_samples}", flush=True)
    n = n_done

    print("\n=== iter-8 上限诊断 (1 帧流式预算) ===")
    overall_f = overall_a = 0
    overall_d = 0
    for j in range(K):
        s = agg_sim[j]
        o = agg_orc[j]
        cov = s["found"] / max(s["drafts"], 1)
        acc = s["accepted"] / max(s["drafts"], 1)
        oacc = o["accepted"] / max(o["drafts"], 1)
        print(
            f"h{j+1}: sim-found={cov:.3f} sim-accept={acc:.3f} "
            f"(found-and-accepted={s['accepted']}/max{s['drafts']}) "
            f"| oracle-info-ceiling={oacc:.3f}"
        )
        overall_f += s["found"]
        overall_a += s["accepted"]
        overall_d += s["drafts"]
    print(
        f"overall: sim-accept={overall_a / max(overall_d, 1):.3f} "
        f"coverage={overall_f / max(overall_d, 1):.3f}"
    )
    print("注: sim-accept 即「头完美读出 sim 特征」的接受率上限;")
    print("    oracle-info-ceiling 是任何预测头的理论上限 (窗口耗尽除外)。")
    out = {
        "per_head_sim": agg_sim,
        "per_head_oracle": agg_orc,
        "n_samples": n,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
