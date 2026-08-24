"""自投机贪心解码 — 无损。

核心思想：Medusa heads 在当前 predictor 末位置草拟 K 个未来 label，用
head-0 (trunk 经 joiner 的 argmax) 在一个帧窗内重放 RNN-T 贪心来验证，
接受与 head-0 一致的最长前缀 m。因验证器 == 原贪心的 argmax，输出与
width=1 oracle **逐 token 完全一致**（无损，PER 不牺牲）。

收益：每接受 m 个 draft 省 m 次外层 (predictor 调用 + Python 往返)，
代价一次 GPU 友好的 joiner 帧窗批量广播。

本实现 eager (非 JIT)，规避 DynamicCompression 的 numpy 不可 script 问题。

三个循环 (greedy / speculative / assert_lossless) 共享同一套状态推进
helper，保证「状态推进严格复刻」是结构性事实而非复制粘贴。
"""

from typing import Dict, List, Optional

import torch
from torch import Tensor


def _init_decode(model, blank_idx: int, dev):
    """初始 predictor 状态: 编码首位 blank, 返回 (g_last, state, one)。"""
    cur_token = torch.tensor([[blank_idx]], device=dev, dtype=torch.long)
    one = torch.tensor([1], device=dev, dtype=torch.int32)
    g0, _, state = model.predict(cur_token, one, None)
    return g0, state, one


def _make_tok(value: int, dev):
    """单 token 张量 (predictor 输入)。"""
    return torch.tensor([[value]], device=dev, dtype=torch.long)


def _join_argmax(model, enc: Tensor, t: int, g_last: Tensor, one) -> int:
    """在 (帧 t, 语言位置 g_last) 处的 joiner argmax (blank 或 label id)。"""
    logits, _, _ = model.join(enc[:, t : t + 1], one, g_last, one)
    return int(logits.argmax(dim=-1).item())


def _advance_predictor(model, tok, state, one):
    """predictor 吃一个 token, 返回新的 g_last 和 state。"""
    g_new, _, state = model.predict(tok, one, state)
    return g_new, state


def greedy_decode(
    model,
    features: Tensor,
    feature_lengths: Tensor,
    blank_idx: int,
    max_steps: int = 100000,
) -> List[int]:
    """标准 RNN-T width=1 贪心解码 (无损 oracle 基线)。

    逐 (t,u) 步：argmax(joiner(enc[t], g[u]))；blank→t+=1，label→emit 并 u+=1。
    """
    dev = features.device
    enc, enc_len = model.transcribe(features, feature_lengths)
    T = enc.size(1)
    g_last, state, one = _init_decode(model, blank_idx, dev)

    out: List[int] = []
    t = 0
    steps = 0
    while t < T and steps < max_steps:
        steps += 1
        pred = _join_argmax(model, enc, t, g_last, one)
        if pred == blank_idx:
            t += 1
        else:
            out.append(pred)
            g_last, state = _advance_predictor(model, _make_tok(pred, dev), state, one)
    return out


def speculative_greedy_decode(
    model,
    features: Tensor,
    feature_lengths: Tensor,
    blank_idx: int,
    max_draft: int = 4,
    per_token_blank: int = 4,
    max_steps: int = 100000,
    max_tokens: Optional[int] = None,
    return_stats: bool = False,
):
    """自投机贪心解码 (链式 draft) — 无损。

    状态推进严格复刻 greedy_decode：emit 后 g_last 推进、t 不变；blank 后
    t 推进、g_last 不变。投机只在"连续 emit"处省 predictor 往返：草拟
    d_1..d_K，逐个用 head-0 joiner argmax 验证(允许 ≤per_token_blank 个
    blank 步到下一次 emit)，命中即连续提交；失配或遇 blank 则本步用标准
    贪心推进。因验证器 == head-0 argmax == oracle argmax ⇒ 逐 token 一致。

    Args:
        max_tokens: 单样本 token 数上限 (幻觉防御)。超限截断并置
            ``truncated`` 供调用方报告；None = 不限制 (仅靠 max_steps)。

    Returns:
        If ``return_stats=False``: ``List[int]`` token ids.
        If ``return_stats=True``: ``(List[int], Dict)`` token ids + stats.
    """
    dev = features.device
    enc, enc_len = model.transcribe(features, feature_lengths)
    T = enc.size(1)
    assert model.medusa_heads is not None, "spec_decode 需 medusa_heads"
    K = min(max_draft, model.medusa_heads.num_heads)
    g_last, state, one = _init_decode(model, blank_idx, dev)

    total_drafts: List[int] = [0] * K  # per head: every proposed token
    accepted: List[int] = [0] * K      # per head: accepted by joiner
    blank_drafts: List[int] = [0] * K
    draft_rounds = 0
    fallback_count = 0

    out: List[int] = []
    t = 0
    steps = 0
    truncated = False

    while t < T and steps < max_steps:
        steps += 1
        if max_tokens is not None and len(out) >= max_tokens:
            # 幻觉防御:未收敛模型会在低置信帧疯狂 emit。截断后停止,标记 truncated。
            truncated = True
            break
        # ---- 本步 head-0 argmax (决定 blank/emit) ----
        pred0 = _join_argmax(model, enc, t, g_last, one)

        if pred0 == blank_idx:
            # blank：t 前进，g_last/state 不变
            t += 1
            continue

        # ---- emit y*=pred0 ----
        out.append(pred0)

        draft_after_emit = getattr(model.medusa_heads, "draft_after_emit", False)

        # In post-emit mode, advance to g_{u+1} before drafting. This makes
        # head k predict y_{u+k} from the same predictor state that the
        # verifier will reach after the current base token.
        if draft_after_emit:
            g_last, state = _advance_predictor(
                model, _make_tok(pred0, dev), state, one
            )

        # ---- 草拟 d_1..d_K ----
        # Legacy mode drafts from g_u; post-emit mode drafts from g_{u+1}.
        draft_rounds += 1
        lookahead = getattr(model.medusa_heads, "lookahead_frames", 0)
        source_window = enc[:, t : t + 1 + lookahead]
        chain_mode = getattr(model.medusa_heads, "chain_states", False)
        drafts: List[int] = []
        if chain_mode:
            # 链式草拟: head-k 的条件状态是「吃掉 d_1..d_{k-1}」后的滚动
            # predictor 状态 g̃ (head-1 即当前 g_last)。draft 全对时
            # g̃ == verifier 验证到该位置时的真实状态, head 只需在正确
            # 前缀上做一步预测, 信息差归零。
            # 逐 head 增量调用: states[0..j] 已知时算出 head-(j+1) 的 logits。
            chain_states: List[Tensor] = [g_last]  # head-1 的条件状态
            g_cur, st_cur = g_last, state
            for j in range(K):
                _, future_logits = model.predict_heads(
                    source_window, g_last,
                    chain_target_encodings=list(chain_states),
                )
                d_j = int(future_logits[j].argmax(dim=-1).item())
                total_drafts[j] += 1
                if d_j == blank_idx:
                    # blank draft 后链式滚动失去意义 (verifier 不会把 blank
                    # 作为 emit 吃掉), 直接停止草拟。
                    blank_drafts[j] += 1
                    break
                drafts.append(d_j)
                if j + 1 < K:
                    # 滚动: predictor 吃 d_j, 作为 head-(j+2) 的条件状态
                    g_cur, st_cur = _advance_predictor(
                        model, _make_tok(d_j, dev), st_cur, one
                    )
                    chain_states.append(g_cur)
            # 未草拟的 head 不计入 drafts 统计
        else:
            _, future_logits = model.predict_heads(source_window, g_last)
            drafts = [int(future_logits[k].argmax(dim=-1).item()) for k in range(K)]
            for j, draft in enumerate(drafts):
                total_drafts[j] += 1
                if draft == blank_idx:
                    blank_drafts[j] += 1

        # Legacy mode consumes pred0 after drafting. In post-emit mode this was
        # done above so the draft and verifier share the same current state.
        if not draft_after_emit:
            g_last, state = _advance_predictor(
                model, _make_tok(pred0, dev), state, one
            )

        # ---- 逐个验证 d_j ----
        # 状态推进与 oracle 完全一致：每个 d_j 允许 ≤per_token_blank 次 blank
        # (t 前进) 后 emit。若 emit == d_j：接受、推进 predictor。若 emit != d_j：
        # 失配但仍须 emit 该 label (与 oracle 一致)，然后停止投机。若 frame 耗尽
        # 未 emit：停止投机 (状态停留，下轮外层从同 t 继续，等价 oracle 走 blank)。
        for j in range(min(K, len(drafts))):
            d_j = drafts[j]
            blanks = 0
            emitted = None
            while blanks < per_token_blank and t < T - 1:
                p = _join_argmax(model, enc, t, g_last, one)
                if p == blank_idx:
                    t += 1
                    blanks += 1
                else:
                    emitted = p
                    break
            if emitted is None:
                # 帧窗耗尽未 emit：停止投机，t 已前进若干 blank (与 oracle 的 blank 前进一致)
                fallback_count += 1
                break
            if emitted != d_j:
                # 失配：但 head-0 在此处确实 emit `emitted`，必须提交以与 oracle 一致
                fallback_count += 1
                out.append(emitted)
                g_last, state = _advance_predictor(model, _make_tok(emitted, dev), state, one)
                break
            # 命中 d_j：提交，推进 predictor (吃 d_j)；emit 后 t 不变
            accepted[j] += 1
            out.append(d_j)
            g_last, state = _advance_predictor(model, _make_tok(d_j, dev), state, one)

    if return_stats:
        stats: Dict = {
            "total_tokens": len(out),
            "total_drafts": sum(total_drafts),
            "total_accepted": sum(accepted),
            "accept_rate": sum(accepted) / max(sum(total_drafts), 1),
            "blank_drafts": sum(blank_drafts),
            "fallbacks": fallback_count,
            "draft_rounds": draft_rounds,
            "truncated": truncated,
            "per_head": [
                {"drafts": total_drafts[i], "accepted": accepted[i],
                 "blank_drafts": blank_drafts[i],
                 "rate": accepted[i] / max(total_drafts[i], 1)}
                for i in range(K)
            ],
        }
        return out, stats
    return out


def assert_lossless(
    model,
    features: Tensor,
    feature_lengths: Tensor,
    blank_idx: int,
    **spec_kwargs,
) -> None:
    """无损断言：spec 输出必须与 greedy oracle 逐 token 一致。

    Raises AssertionError if mismatch. 用于验证投机解码的无损性。
    """
    oracle = greedy_decode(model, features, feature_lengths, blank_idx)
    spec = speculative_greedy_decode(
        model, features, feature_lengths, blank_idx, **spec_kwargs
    )
    assert oracle == spec, (
        f"无损失败: oracle len={len(oracle)} spec len={len(spec)}; "
        f"首个差异 @ {next((i for i, (a, b) in enumerate(zip(oracle, spec)) if a != b), 'n/a')}"
    )
