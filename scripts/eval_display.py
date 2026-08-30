#!/usr/bin/env python3
"""投机显示评估器 — 度量「提前量 × 命中率 (− 替换率)」。

iter-9 价值模型: 解码回归纯贪心 RNNT; Medusa 草稿只上屏 (用完即弃),
真实 token 到达时确认或替换。本脚本离线模拟该时间线:

  1. 贪心解码产出参考序列 (真实 token 及其 emit 帧) — 与产品解码完全同构;
     草稿不参与解码 (用完即弃), 故一次解码即可, emit 点状态顺路记录;
  2. 解码完成后, 在每个 emit 点 (draft_after_emit 语义, 状态 = g_{u+1})
     用头链草拟 K 个后续 token, anchor 帧 = 该 emit 帧;
  3. 对齐: 草稿 head-j (0-based) 预测参考序列第 u+1+j 个 token
     (训练侧 shifted_ce: head-k 在列 m 预测 y_{m+k-1}, 此处 m = u+1);
     blank 草稿 = 弃权 (不上屏, 不产生闪烁), 链在此截断。

指标 (每头 h1/h2 + 合并):
  - 显示数 / 弃权数 (blank): 上屏行为统计;
  - 命中率 = 命中 / 显示 (未命中即被真实 token 替换 → 替换率 = 1−命中率);
  - 提前量 = 真实 emit 帧 − 草稿帧 (×帧时长 ms), 只对命中统计有意义
    (命中草稿的提前量 = 观众提前看到正确内容的时间)。

用法:
  python scripts/eval_display.py <config.yaml> [max_samples] [checkpoint]
  # 例: 先对 iter-7 基线跑天花板数字
  #   python scripts/eval_display.py configs/server_display.yaml 100 \
  #       /test21/conformer-rnnt-asr/exp_iter2/BEST_iter7.ckpt
  # 重训后不带 checkpoint 参数则自动选 exp_dir/checkpoints 里最新的
  # val-{tag}-*.ckpt (tag 由配置生成, iter-9 为 ...-disp)。
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import checkpoint_dir, load_config, model_ckpt_tag, setup_logging
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
from conformer_rnnt.spec_decode import (
    _advance_predictor,
    _init_decode,
    _join_argmax,
    _make_tok,
)

MAX_STEPS = 100000


def _build_source_window(
    model, enc: torch.Tensor, t: int
) -> torch.Tensor:
    """构造头草拟用的声学窗 (与 spec_decode 推理路径逐位一致)。

    past=0 布局为 [t .. t+lookahead]; 序列末端不足处零 pad
    (与训练 _anchor_window 的 required = T + lookahead 对齐)。
    """
    heads = model.medusa_heads
    past = getattr(heads, "past_frames", 0)
    lookahead = getattr(heads, "lookahead_frames", 0)
    win_len = past + 1 + lookahead
    if past > 0:
        if t < past:
            src = torch.nn.functional.pad(enc, (0, 0, past - t, 0))
            w_start = 0
        else:
            src = enc
            w_start = t - past
        window = src[:, w_start : w_start + win_len]
    else:
        window = enc[:, t : t + win_len]
    if window.size(1) < win_len:
        window = torch.nn.functional.pad(
            window, (0, 0, 0, win_len - window.size(1))
        )
    return window


def _draft_chain(
    model,
    enc: torch.Tensor,
    t: int,
    g0: torch.Tensor,
    st0: List[torch.Tensor],
    one: torch.Tensor,
    blank_idx: int,
    dev,
) -> Tuple[List[int], int]:
    """在帧 t 以 (g0, st0) 为起点草拟一轮 (chain + post-emit 语义)。

    返回 (drafts, attempted): drafts 为非 blank 草稿 id 列表 (链在首个
    blank 处截断); attempted 为尝试草拟的头数 (含弃权的那个头)。
    逐位复刻 spec_decode.speculative_greedy_decode 的草拟分支, 但不做验证。
    """
    heads = model.medusa_heads
    K = int(heads.num_heads)
    window = _build_source_window(model, enc, t)
    # iter-8 验证器模拟扫描窗 [t .. t+lookahead]: chain_sim 头用它读出
    # 「真 joiner 在本头 rolled state 上的窗口 argmax」— 流式预算内可得,
    # 所以显示评估也应当给它 (否则量的是被阉割的头)。
    sim_window: Optional[torch.Tensor] = None
    if getattr(heads, "chain_sim_enabled", False):
        lookahead = getattr(heads, "lookahead_frames", 0)
        need = 1 + lookahead
        sim_window = enc[:, t : t + need]
        if sim_window.size(1) < need:
            sim_window = torch.nn.functional.pad(
                sim_window, (0, 0, 0, need - sim_window.size(1))
            )
    drafts: List[int] = []
    chain_rolls: List[torch.Tensor] = [g0]
    g_cur, st_cur = g0, st0
    attempted = 0
    for j in range(K):
        attempted += 1
        _, future_logits = model.predict_heads(
            window,
            g_cur,
            chain_drafts=list(drafts),
            chain_states=list(chain_rolls),
            sim_window=sim_window,
        )
        d_j = int(future_logits[j].argmax(dim=-1).item())
        if d_j == blank_idx:
            break
        drafts.append(d_j)
        if j + 1 < K:
            g_cur, st_cur = _advance_predictor(
                model, _make_tok(d_j, dev), st_cur, one
            )
            chain_rolls.append(g_cur)
    return drafts, attempted


def _quantiles(values: List[int], qs=(0.5, 0.9, 0.99)) -> Dict[str, float]:
    """离散值分位数 (升序累计)。"""
    if not values:
        return {f"P{int(q * 100)}": float("nan") for q in qs}
    ordered = sorted(values)
    marks: Dict[str, float] = {}
    for q in qs:
        idx = min(int(q * len(ordered)), len(ordered) - 1)
        marks[f"P{int(q * 100)}"] = float(ordered[idx])
    return marks


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/server_display.yaml"
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ckpt_override = sys.argv[3] if len(sys.argv) > 3 else None

    config = load_config(config_path)
    setup_logging(config)
    model = LibriSpeechRNNTModule.from_config(config)

    if ckpt_override:
        ckpt_path = ckpt_override
    else:
        # 只认本配置 tag 的 ckpt, 或配置里显式指定的 from_checkpoint。
        # 不做 val-* 通配回退: 不同窗口/机制的 ckpt 参数集合不兼容,
        # 抓到最新文件会得到「布局错配」的假数字 (且 strict 加载会直接报错)。
        ckpt_dir = checkpoint_dir(config)
        tag = model_ckpt_tag(config)
        candidates = glob.glob(os.path.join(ckpt_dir, f"val-{tag}-*.ckpt"))
        if candidates:
            ckpt_path = max(candidates, key=os.path.getmtime)
        else:
            ckpt_path = config.get("paths", {}).get("from_checkpoint", "")
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(
                f"no checkpoint found for tag '{tag}' in {ckpt_dir}; "
                "pass one explicitly as argv[3]"
            )
            sys.exit(1)
    print(f"checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu").get("state_dict", {})
    model.load_state_dict(state_dict)

    heads = model.model.medusa_heads
    assert heads is not None, "eval_display 需要 medusa_heads"
    assert getattr(heads, "draft_after_emit", False), (
        "显示时间线按 post-emit 语义设计 (draft_after_emit: true)"
    )
    K = int(heads.num_heads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    blank_idx = model.blank_idx
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    frame_ms = (
        data_cfg.get("hop_length", 160)
        / data_cfg.get("sample_rate", 16000)
        * 1000
        * model_cfg.get("time_reduction_stride", 4)
    )
    print(f"frame_ms={frame_ms:.0f} K={K} max_samples={max_samples}")

    # 每头累计: attempted(尝试) / displayed(非blank上屏) / abstained(blank弃权)
    #           / hits / lead(命中草稿的提前帧数)
    attempted = [0] * K
    displayed = [0] * K
    abstained = [0] * K
    hits = [0] * K
    leads: List[List[int]] = [[] for _ in range(K)]
    unscorable = 0  # 草稿指向句尾之后 (无真实 token 可确认), 不计入命中率
    n_utts = 0
    n_ref_tokens = 0

    with torch.no_grad():
        for _batch_idx, (batch, _transcripts) in enumerate(model.test_dataloader()):
            if n_utts >= max_samples:
                break
            features = batch.features.to(device)
            feature_lengths = batch.feature_lengths.to(device)
            for b in range(features.size(0)):
                if n_utts >= max_samples:
                    break
                enc, _ = model.model.transcribe(
                    features[b : b + 1], feature_lengths[b : b + 1]
                )
                T = enc.size(1)
                g_last, state, one = _init_decode(model.model, blank_idx, device)

                # pass 1: 纯贪心参考序列 + 每个 emit 点的 post-emit 状态快照。
                # 草稿不参与解码 (用完即弃), 所以这次解码就是产品解码本身。
                ref_tokens: List[int] = []
                ref_frames: List[int] = []
                snaps: List[Tuple[int, torch.Tensor, List[torch.Tensor]]] = []
                t = 0
                steps = 0
                while t < T and steps < MAX_STEPS:
                    steps += 1
                    pred = _join_argmax(model.model, enc, t, g_last, one)
                    if pred == blank_idx:
                        t += 1
                        continue
                    ref_tokens.append(pred)
                    ref_frames.append(t)
                    g_last, state = _advance_predictor(
                        model.model, _make_tok(pred, device), state, one
                    )
                    snaps.append((t, g_last, state))

                # pass 2: 在每个 emit 点上屏草稿 (对齐到完整参考序列)。
                for u, (t_emit, g_disp, st_disp) in enumerate(snaps):
                    drafts, att = _draft_chain(
                        model.model, enc, t_emit, g_disp, st_disp,
                        one, blank_idx, device,
                    )
                    for j in range(K):
                        if j >= att:
                            break  # 链在前一头弃权处截断, 更深头未尝试
                        attempted[j] += 1
                        if j >= len(drafts):
                            abstained[j] += 1  # blank 弃权
                            continue
                        displayed[j] += 1
                        target = u + 1 + j
                        if target >= len(ref_tokens):
                            unscorable += 1
                            continue
                        if drafts[j] == ref_tokens[target]:
                            hits[j] += 1
                            leads[j].append(ref_frames[target] - t_emit)
                n_utts += 1
                n_ref_tokens += len(ref_tokens)

    # ---- 汇总 ----
    print(f"\nutterances={n_utts} ref_tokens={n_ref_tokens}")
    total_displayed = sum(displayed)
    total_hits = sum(hits)
    header = (
        f"{'head':<6}{'att':>8}{'disp':>8}{'abst':>8}{'hit':>8}"
        f"{'hit%':>8}{'repl%':>8}{'leadP50':>9}{'leadP90':>9}{'leadP99':>9} (ms)"
    )
    print(header)
    for j in range(K):
        disp = displayed[j]
        hit = hits[j]
        hit_rate = 100.0 * hit / disp if disp else float("nan")
        repl_rate = 100.0 - hit_rate if disp else float("nan")
        q = _quantiles(leads[j])
        lead_ms = {k: v * frame_ms for k, v in q.items()}
        print(
            f"h{j + 1:<5}{attempted[j]:>8}{disp:>8}{abstained[j]:>8}"
            f"{hit:>8}{hit_rate:>8.2f}{repl_rate:>8.2f}"
            f"{lead_ms['P50']:>9.0f}{lead_ms['P90']:>9.0f}{lead_ms['P99']:>9.0f}"
        )
    merged_hit = 100.0 * total_hits / total_displayed if total_displayed else float("nan")
    print(
        f"\nmerged: displayed={total_displayed} hits={total_hits} "
        f"hit%={merged_hit:.2f} unscorable={unscorable}"
    )
    hit_lead_all = [x for lst in leads for x in lst]
    q_all = _quantiles(hit_lead_all)
    print(
        "lead(all hits, ms): "
        + " ".join(f"{k}={v * frame_ms:.0f}" for k, v in q_all.items())
    )


if __name__ == "__main__":
    main()
