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
    (命中草稿的提前量 = 观众提前看到正确内容的时间);
  - **区域拆分**: 复制区 (sim 扫描在可见窗内问到答案, 头照抄) vs 预测区
    (答案在窗外, 头必须真的预测)。总命中率被复制区占比主导, 不可跨头比较;
    头的真实能力看预测区那一列。
  - **LM 先验基线**: predictor 自回归 (喂自己的 argmax)、声学侧固定为句级
    常量。这是预测区的下限参照 — 头若 ≈ 该值, 说明它只是个 LM。

用法:
  python scripts/eval_display.py <config.yaml> [max_samples] [checkpoint] [--no-sim]
  # 例: 先对 iter-7 基线跑天花板数字
  #   python scripts/eval_display.py configs/server_display.yaml 100 \
  #       /test21/conformer-rnnt-asr/exp_iter2/BEST_iter7.ckpt
  # 重训后不带 checkpoint 参数则自动选 exp_dir/checkpoints 里最新的
  # val-{tag}-*.ckpt (tag 由配置生成, iter-9 为 ...-disp)。
  # --no-sim: 屏蔽 sim 注入 (扫描照跑, 只是不喂给头) → 纯预测模式, 用于
  #   量「头在没有复制通道可走时的真实预测准确率」。
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
) -> Tuple[List[int], int, List[Optional[Dict[str, object]]]]:
    """在帧 t 以 (g0, st0) 为起点草拟一轮 (chain + post-emit 语义)。

    返回 (drafts, attempted, sim_reads): drafts 为非 blank 草稿 id 列表
    (链在首个 blank 处截断); attempted 为尝试草拟的头数 (含弃权的那个头);
    sim_reads 为每头的推理端 sim 扫描读出 (chain_sim 关闭时为 None)。
    逐位复刻 spec_decode.speculative_greedy_decode 的草拟分支, 但不做验证。
    """
    heads = model.medusa_heads
    K = int(heads.num_heads)
    window = _build_source_window(model, enc, t)
    # iter-8 验证器模拟扫描窗 [t .. t+lookahead]: chain_sim 头用它读出
    # 「真 joiner 在本头 rolled state 上的窗口 argmax」— 流式预算内可得,
    # 所以显示评估也应当给它 (否则量的是被阉割的头)。
    sim_window: Optional[torch.Tensor] = None
    sim_read: Optional[Dict[str, torch.Tensor]] = None
    if getattr(heads, "chain_sim_enabled", False):
        lookahead = getattr(heads, "lookahead_frames", 0)
        need = 1 + lookahead
        sim_window = enc[:, t : t + need]
        if sim_window.size(1) < need:
            sim_window = torch.nn.functional.pad(
                sim_window, (0, 0, 0, need - sim_window.size(1))
            )
    drafts: List[int] = []
    sim_reads: List[Optional[Dict[str, object]]] = [None] * K
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
        # 本头推理端 sim 扫描读出 (predict_heads 内部已算好): 用来判断
        # 「直注查表到底有没有被点亮」— found=False 时直注不生效。
        scans = getattr(heads, "last_infer_scan", None)
        if scans is not None and j < len(scans) and scans[j] is not None:
            sim_read = {
                "found": bool(scans[j]["found"].item()),
                "sim_id": int(scans[j]["sim_id"].item()),
                "sim_delta": int(scans[j]["sim_delta"].item()),
            }
        else:
            sim_read = None
        sim_reads[j] = sim_read
        if d_j == blank_idx:
            break
        drafts.append(d_j)
        if j + 1 < K:
            g_cur, st_cur = _advance_predictor(
                model, _make_tok(d_j, dev), st_cur, one
            )
            chain_rolls.append(g_cur)
    return drafts, attempted, sim_reads


def _lm_free_run(
    model,
    src_const: torch.Tensor,
    g0: torch.Tensor,
    st0: List[torch.Tensor],
    one: torch.Tensor,
    K: int,
    blank_idx: int,
    dev,
) -> List[Optional[int]]:
    """纯语言先验基线 — predictor 自回归 (喂自己的 argmax), 声学侧固定。

    ``src_const`` 是句级常量 (enc 的时间均值), 抹掉逐帧声学信息但保留
    in-distribution 偏置 (直接置零会让 relu 工作点整体偏移, 基线失真)。
    返回长度 K 的猜测列表, blank 处截断 (与头同规则: blank = 弃权)。

    这是「头的预测区准确率」的下限参照: 若头在预测区的命中率 ≈ 该基线,
    说明头根本没在用声学窗做预测, 只是个 LM; 显著高于它才说明头在干活。
    """
    out: List[Optional[int]] = []
    g_cur, st_cur = g0, st0
    for _ in range(K):
        joint = model.joiner.fuse(src_const, g_cur)  # (1,1,1,D)
        logits = model.joiner.project(joint.squeeze(2))  # (1,1,V)
        tok = int(logits.argmax(dim=-1).item())
        if tok == blank_idx:
            break
        out.append(tok)
        g_cur, st_cur = _advance_predictor(
            model, _make_tok(tok, dev), st_cur, one
        )
    return out


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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_sim = "--no-sim" in sys.argv  # 屏蔽 sim 注入 → 纯预测模式
    config_path = args[0] if args else "configs/server_display.yaml"
    max_samples = int(args[1]) if len(args) > 1 else 100
    ckpt_override = args[2] if len(args) > 2 else None

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
    # 纯预测模式: 扫描照跑 (诊断标签照出), 但不注入读出 → 头只能靠自己猜。
    # 用来分离「复制通道刷出的命中率」与「头真正的预测能力」。
    if no_sim and getattr(heads, "chain_sim_enabled", False):
        heads.sim_inject_enabled = False
        print("sim 注入: 已屏蔽 (纯预测模式)")
    elif no_sim:
        print("sim 注入: 本 ckpt 未启用 chain_sim, --no-sim 无效果")

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
    # chain_sim 诊断: 推理端扫描是否找到 token / 找到的是否就是目标 /
    # 头是否照抄了直注 (sim_id)。三者分离才能定位「信息在但头没用上」。
    sim_seen = [0] * K
    sim_found = [0] * K
    sim_right = [0] * K
    sim_copied = [0] * K
    # 区域拆分: 复制区 (sim 在窗内问到答案, 头只需照抄) vs 预测区 (答案在
    # 窗外, 头必须真的预测)。后者才是头的战场 — 前者是直注查表的功劳。
    disp_copy = [0] * K
    hit_copy = [0] * K
    disp_pred = [0] * K
    hit_pred = [0] * K
    # 纯语言先验基线 (自回归 LM, 声学侧为句级常量): 预测区的下限参照。
    lm_guess = [0] * K
    lm_hit = [0] * K
    lm_guess_pred = [0] * K
    lm_hit_pred = [0] * K
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

                enc_mean = enc.mean(dim=1, keepdim=True)  # (1,1,D) 句级常量

                # pass 2: 在每个 emit 点上屏草稿 (对齐到完整参考序列)。
                for u, (t_emit, g_disp, st_disp) in enumerate(snaps):
                    drafts, att, sim_reads = _draft_chain(
                        model.model, enc, t_emit, g_disp, st_disp,
                        one, blank_idx, device,
                    )
                    lm_drafts = _lm_free_run(
                        model.model, enc_mean, g_disp, st_disp,
                        one, K, blank_idx, device,
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
                        sr = sim_reads[j] if j < len(sim_reads) else None
                        is_copy = bool(sr is not None and sr["found"])
                        if sr is not None:
                            sim_seen[j] += 1
                            if is_copy:
                                sim_found[j] += 1
                                if sr["sim_id"] == ref_tokens[target]:
                                    sim_right[j] += 1
                                if drafts[j] == sr["sim_id"]:
                                    sim_copied[j] += 1
                        ok = drafts[j] == ref_tokens[target]
                        if ok:
                            hits[j] += 1
                            leads[j].append(ref_frames[target] - t_emit)
                        if is_copy:
                            disp_copy[j] += 1
                            hit_copy[j] += ok
                        else:
                            disp_pred[j] += 1
                            hit_pred[j] += ok
                        if j < len(lm_drafts) and lm_drafts[j] is not None:
                            lm_guess[j] += 1
                            lm_ok = lm_drafts[j] == ref_tokens[target]
                            lm_hit[j] += lm_ok
                            if not is_copy:
                                lm_guess_pred[j] += 1
                                lm_hit_pred[j] += lm_ok
                n_utts += 1
                n_ref_tokens += len(ref_tokens)

    # ---- 汇总 ----
    print(f"\nutterances={n_utts} ref_tokens={n_ref_tokens}")
    total_displayed = sum(displayed)
    total_hits = sum(hits)
    header = (
        f"{'head':<6}{'att':>8}{'disp':>8}{'abst':>8}{'hit':>8}"
        f"{'hit%':>8}{'repl%':>8}{'leadP50':>9}{'leadP90':>9}{'leadP99':>9}"
        f"{'leadAvg':>9}{'value':>9} (ms)"
    )
    print(header)
    for j in range(K):
        disp = displayed[j]
        hit = hits[j]
        hit_rate = 100.0 * hit / disp if disp else float("nan")
        repl_rate = 100.0 - hit_rate if disp else float("nan")
        q = _quantiles(leads[j])
        lead_ms = {k: v * frame_ms for k, v in q.items()}
        # 一级指标: 期望感知提前量 = 命中率 × 命中草稿的平均提前量。
        # 未命中的草稿会被真实 token 替换 (闪烁), 提前量记 0。
        mean_lead = sum(leads[j]) / len(leads[j]) if leads[j] else float("nan")
        value = (hit / disp) * mean_lead * frame_ms if disp else float("nan")
        print(
            f"h{j + 1:<5}{attempted[j]:>8}{disp:>8}{abstained[j]:>8}"
            f"{hit:>8}{hit_rate:>8.2f}{repl_rate:>8.2f}"
            f"{lead_ms['P50']:>9.0f}{lead_ms['P90']:>9.0f}{lead_ms['P99']:>9.0f}"
            f"{mean_lead * frame_ms:>9.0f}{value:>9.1f}"
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

    # ---- 区域拆分: 复制通道 vs 预测通道 ----
    # 复制区 = sim 扫描在可见窗内问到答案 (头照抄即可); 预测区 = 答案在窗
    # 外 (头必须真的预测)。头的价值只在预测区, 总命中率被复制区占比主导,
    # 所以 h1 的 90% 与 h2 的 59% 不可直接比较 — 要看预测区那一列。
    if any(disp_copy) or any(disp_pred):
        print("\n区域拆分 (复制区 = 答案在窗内, 预测区 = 答案在窗外):")
        print(
            f"{'head':<6}{'copy_n':>8}{'copyHit%':>10}"
            f"{'pred_n':>8}{'predHit%':>10}"
            f"{'LMhit%':>9}{'LMguess%':>10}"
        )
        for j in range(K):
            n_c, n_p = disp_copy[j], disp_pred[j]
            c_hit = 100.0 * hit_copy[j] / n_c if n_c else float("nan")
            p_hit = 100.0 * hit_pred[j] / n_p if n_p else float("nan")
            # LM 命中率以「全部可打分草稿」为分母, 与 predHit% 同口径可比;
            # LMguess% 是 LM 给出非 blank 猜测的比例 (其余视为弃权)。
            denom = n_c + n_p
            lm_rate = 100.0 * lm_hit[j] / denom if denom else float("nan")
            lm_g = 100.0 * lm_guess[j] / denom if denom else float("nan")
            print(
                f"h{j + 1:<5}{n_c:>8}{c_hit:>10.2f}"
                f"{n_p:>8}{p_hit:>10.2f}"
                f"{lm_rate:>9.2f}{lm_g:>10.2f}"
            )
        if any(lm_guess_pred):
            print("\n预测区 LM 先验 vs 头 (同一批样本):")
            print(f"{'head':<6}{'pred_n':>8}{'headHit%':>10}{'LMhit%':>9}{'gap':>8}")
            for j in range(K):
                n_p = disp_pred[j]
                if not n_p:
                    continue
                p_hit = 100.0 * hit_pred[j] / n_p
                lm_p = 100.0 * lm_hit_pred[j] / n_p
                print(
                    f"h{j + 1:<5}{n_p:>8}{p_hit:>10.2f}{lm_p:>9.2f}"
                    f"{p_hit - lm_p:>+8.2f}"
                )

    if any(sim_seen):
        print("\nchain_sim 推理端诊断 (直注是否被点亮):")
        print(f"{'head':<6}{'scan':>8}{'found%':>9}{'simRight%':>11}{'copied%':>10}")
        for j in range(K):
            seen = sim_seen[j]
            if not seen:
                continue
            print(
                f"h{j + 1:<5}{seen:>8}"
                f"{100.0 * sim_found[j] / seen:>9.2f}"
                f"{100.0 * sim_right[j] / max(sim_found[j], 1):>11.2f}"
                f"{100.0 * sim_copied[j] / max(sim_found[j], 1):>10.2f}"
            )


if __name__ == "__main__":
    main()
