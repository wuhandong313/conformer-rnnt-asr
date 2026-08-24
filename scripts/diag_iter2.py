#!/usr/bin/env python3
"""iter-2 诊断: teacher-forcing vs draft-forcing head 准确率 + 损失分量。

用途: 定位「CE 下降但接受率不升」的原因 —
  A. 训练条件 (TF) 上 head argmax 就不准 → 目标/容量问题
  B. TF 准但 draft 条件不准 → exposure bias
  C. 损失分量失衡 (kl/accept_ce 压制 ce)
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from common import load_config
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
from conformer_rnnt.medusa import shifted_ce_loss
from conformer_rnnt.spec_decode import (
    _init_decode, _join_argmax, _make_tok, _advance_predictor
)


def main() -> None:
    config = load_config("configs/server_chain.yaml")
    model = LibriSpeechRNNTModule.from_config(config)
    files = sorted(
        glob.glob("exp_iter2/checkpoints/val-*.ckpt"), key=os.path.getmtime
    )
    ckpt = files[-1]
    print("ckpt:", ckpt)
    sd = torch.load(ckpt, map_location="cpu").get("state_dict", {})
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()

    dl = model.test_dataloader()
    blank = model.blank_idx

    # A. teacher-forcing argmax accuracy (per-position, majority vote over t)
    n_ok = [0, 0, 0]
    n_tot = [0, 0, 0]
    ce_vals = []
    with torch.no_grad():
        for idx, (batch, _) in enumerate(dl):
            if idx >= 20:
                break
            feats = batch.features.cuda()
            fl = batch.feature_lengths.cuda()
            tgts = batch.targets.cuda()
            tl = batch.target_lengths.cuda()
            prep = tgts.new_empty((tgts.size(0), tgts.size(1) + 1))
            prep[:, 1:] = tgts
            prep[:, 0] = blank
            logits, src_len, _, base, future = model.model.forward_train_aux(
                feats, fl, prep, tl + 1
            )
            src_len = torch.clamp(src_len, max=logits.size(1))
            for k in range(3):
                f = future[k]
                am = f.argmax(-1)
                for b in range(tgts.size(0)):
                    U = int(tl[b])
                    T = int(src_len[b])
                    for l in range(1, min(U + 1, am.size(2) - k + 1)):
                        t_idx = l + k - 1
                        if t_idx >= U:
                            continue
                        votes = am[b, :T, l].bincount(
                            minlength=f.size(-1)
                        ).argmax()
                        n_tot[k] += 1
                        if int(votes) == int(tgts[b, t_idx]):
                            n_ok[k] += 1
            # CE per head at valid positions (the actual training objective)
            sl = src_len.to(torch.int32)
            ce = shifted_ce_loss(
                future, tgts, [1.0] * 3, tl.to(torch.int32), sl,
                blank_idx=blank, future_state_offset=1,
            )
            ce_vals.append(float(ce))
    print("A. teacher-forcing accuracy (majority over t):")
    for k in range(3):
        print(f"   h{k+1}: {n_ok[k]}/{n_tot[k]} = {n_ok[k]/max(n_tot[k],1):.3f}")
    print(f"   shifted-CE (avg): {sum(ce_vals)/len(ce_vals):.4f}")

    # B. draft-forcing chain accuracy: replay greedy, at each emit roll drafts
    dk_ok = [0, 0, 0]
    dk_tot = [0, 0, 0]
    with torch.no_grad():
        for idx, (batch, _) in enumerate(dl):
            if idx >= 20:
                break
            feats = batch.features.cuda()
            fl = batch.feature_lengths.cuda()
            enc, _ = model.model.transcribe(feats, fl)
            T = enc.size(1)
            g_last, state, one = _init_decode(model.model, blank, "cuda")
            t = 0
            steps = 0
            while t < T and steps < 10000:
                steps += 1
                pred0 = _join_argmax(model.model, enc, t, g_last, one)
                if pred0 == blank:
                    t += 1
                    continue
                # post-emit advance
                g_last, state = _advance_predictor(
                    model.model, _make_tok(pred0, "cuda"), state, one
                )
                # chain drafts with GT check against future oracle tokens
                lookahead = model.model.medusa_heads.lookahead_frames
                src_win = enc[:, t : t + 1 + lookahead]
                chain_states = [g_last]
                g_cur, st_cur = g_last, state
                oracle_rest = _oracle_tail(
                    model.model, enc, t, g_last, state, one, blank, 3
                )
                for j in range(3):
                    _, flog = model.model.predict_heads(
                        src_win, g_last,
                        chain_target_encodings=list(chain_states),
                    )
                    d = int(flog[j].argmax(dim=-1).item())
                    if d == blank:
                        break
                    dk_tot[j] += 1
                    if j < len(oracle_rest) and d == oracle_rest[j]:
                        dk_ok[j] += 1
                    chain_states.append(g_cur)
                    if j < 2:
                        g_cur, st_cur = _advance_predictor(
                            model.model, _make_tok(d, "cuda"), st_cur, one
                        )
                        chain_states[-1] = g_cur
                # advance verifier normally (consume pred0 path done above);
                # continue greedy from current state
                # (this loop only measures drafts; actual decode continues)
                # step t forward to next emit using current state:
                done = False
                while t < T - 1:
                    p = _join_argmax(model.model, enc, t, g_last, one)
                    if p == blank:
                        t += 1
                    else:
                        done = True
                        break
                if not done:
                    break
                g_last, state = _advance_predictor(
                    model.model, _make_tok(p, "cuda"), state, one
                )
    print("B. draft-forcing accuracy (chain, vs oracle tail):")
    for j in range(3):
        print(f"   h{j+1}: {dk_ok[j]}/{dk_tot[j]} = {dk_ok[j]/max(dk_tot[j],1):.3f}")


def _oracle_tail(model, enc, t, g_last, state, one, blank, n):
    """Return next n emitted tokens from the verifier state (t, g_last)."""
    out = []
    g, st, tt = g_last, state, t
    steps = 0
    while tt < enc.size(1) - 1 and steps < 200 and len(out) < n:
        steps += 1
        p = _join_argmax(model, enc, tt, g, one)
        if p == blank:
            tt += 1
            continue
        out.append(p)
        g, st = _advance_predictor(model, _make_tok(p, enc.device), st, one)
    return out


if __name__ == "__main__":
    main()
