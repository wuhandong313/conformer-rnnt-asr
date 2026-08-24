#!/usr/bin/env python3
"""探测 greedy 解码的 emit 帧间距分布 — 校准 spec 验证窗 per_token_blank。

统计每次 emit 相对上一次 emit 前进了多少声学帧 (Δt)。接受率上限约为
P(Δt ≤ w)^k, 因此 w 应覆盖 Δt 分布的高分位 (如 P95)。
"""

from __future__ import annotations

import glob
import os
import sys
from collections import Counter

import torch

from common import checkpoint_dir, load_config, model_ckpt_tag, setup_logging
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
from conformer_rnnt.spec_decode import _init_decode, _join_argmax, _make_tok, _advance_predictor


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/server.yaml"
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    config = load_config(config_path)
    setup_logging(config)
    model = LibriSpeechRNNTModule.from_config(config)

    ckpt_dir = checkpoint_dir(config)
    tag = model_ckpt_tag(config)
    candidates = glob.glob(os.path.join(ckpt_dir, f"val-{tag}-*.ckpt")) or glob.glob(
        os.path.join(ckpt_dir, "val-*.ckpt")
    )
    if not candidates:
        print("no checkpoint found")
        sys.exit(1)
    ckpt_path = max(candidates, key=os.path.getmtime)
    print(f"checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu").get("state_dict", {})
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    blank_idx = model.blank_idx
    dataloader = model.test_dataloader()

    deltas = Counter()
    total_emits = 0
    n = 0
    with torch.no_grad():
        for idx, (batch, _transcripts) in enumerate(dataloader):
            if idx >= max_samples:
                break
            features = batch.features.to(device)
            feature_lengths = batch.feature_lengths.to(device)
            enc, _ = model.model.transcribe(features, feature_lengths)
            T = enc.size(1)
            g_last, state, one = _init_decode(model.model, blank_idx, device)
            last_emit_t = 0
            t = 0
            steps = 0
            while t < T and steps < 100000:
                steps += 1
                pred = _join_argmax(model.model, enc, t, g_last, one)
                if pred == blank_idx:
                    t += 1
                else:
                    total_emits += 1
                    deltas[t - last_emit_t] += 1
                    last_emit_t = t
                    g_last, state = _advance_predictor(
                        model.model, _make_tok(pred, device), state, one
                    )
            n += 1

    print(f"\nsamples={n} total_emits={total_emits}")
    cum = 0
    print("Δt   count   pct    cum_pct")
    for dt in sorted(deltas):
        cnt = deltas[dt]
        cum += cnt
        print(f"{dt:3d}  {cnt:6d}  {100*cnt/total_emits:5.2f}%  {100*cum/total_emits:6.2f}%")
    # 分位数
    cum = 0
    for dt in sorted(deltas):
        cum += deltas[dt]
        for q in (0.5, 0.75, 0.9, 0.95, 0.99):
            pass
    cum = 0
    marks = {0.5: None, 0.75: None, 0.9: None, 0.95: None, 0.99: None}
    for dt in sorted(deltas):
        cum += deltas[dt]
        frac = cum / total_emits
        for q in marks:
            if marks[q] is None and frac >= q:
                marks[q] = dt
    print("quantiles:", {f"P{int(q*100)}": v for q, v in marks.items()})


if __name__ == "__main__":
    main()
