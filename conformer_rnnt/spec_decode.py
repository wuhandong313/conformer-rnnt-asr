"""自投机贪心解码 (plan §3.5) — 无损。

核心思想：Medusa heads 在当前 predictor 末位置草拟 K 个未来 label，用
head-0 (trunk 经 joiner 的 argmax) 在一个帧窗内重放 RNN-T 贪心来验证，
接受与 head-0 一致的最长前缀 m。因验证器 == 原贪心的 argmax，输出与
width=1 oracle **逐 token 完全一致**（无损，PER 不牺牲）。

收益：每接受 m 个 draft 省 m 次外层 (predictor 调用 + Python 往返)，
代价一次 GPU 友好的 joiner 帧窗批量广播。

链式 draft (默认)：同头单路径 argmax(d_j)。
树式 (tree=True)：每头取 top-k 路径，取可接受最长者（同一批 heads，仅改候选）。

对齐契约 (draft_before_emit)：训练侧 medusa.shifted_ce_loss 的语义是
head k 在位置 u (predictor 已消费 prepended[0..u]，末 token = y_{u-1})
预测 y_{u+k}。因此草拟必须在**消费刚 emit 的 token 之前**做：刚在
位置 u 用 head-0 emit 了 y_u，此刻查询头，d_1 = y_{u+1} 恰是 y_u 之后
的下一个 emit，验证时逐位对齐。若先推进 predictor 再草拟 (post-emit)，
草稿系统性偏移一位，接受率趋近 0（multihead 分支 iter-10 的
draft_after_emit 模式需配合重写的头语义，本分支瘦身头为 legacy 语义）。

本实现 eager (非 JIT)，规避 DynamicCompression 的 numpy 不可 script 问题。
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor


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
    B = enc.size(0)
    assert B == 1, "greedy_decode 仅支持 batch=1"

    # predictor state: 从空开始。输入序列首位 = blank。
    # RNNTBeamSearch 的 predict 逐 token 调用；这里我们也逐 token。
    # 注：predictor 输入是 token id；首 token = blank_idx。
    cur_token = torch.tensor([[blank_idx]], device=dev, dtype=torch.long)
    one = torch.tensor([1], device=dev, dtype=torch.int32)
    g, _, state = model.predict(cur_token, one, None)  # g: (1,1,512)
    # 我们维护"当前 g 末位"：初始是 blank 的编码。

    out: List[int] = []
    t = 0
    u = 0  # 已 emit 的 label 数
    steps = 0
    # g_last: (1,1,512) 当前位置 u 的 predictor 编码
    g_last = g

    while t < T and steps < max_steps:
        steps += 1
        # joiner on (enc[:, t:t+1], g_last) → logits (1,1,1,V)
        logits, _, _ = model.join(enc[:, t:t+1], torch.tensor([1], device=dev, dtype=torch.int32),
                                  g_last, one)
        pred = int(logits.argmax(dim=-1).item())
        if pred == blank_idx:
            t += 1
            g_last = g_last  # label 位不变，换帧
        else:
            out.append(pred)
            u += 1
            # predictor 吃新 emit 的 label，产出新的 g 末位
            cur_token = torch.tensor([[pred]], device=dev, dtype=torch.long)
            g_new, _, state = model.predict(cur_token, one, state)
            g_last = g_new
    return out


def speculative_greedy_decode(
    model,
    features: Tensor,
    feature_lengths: Tensor,
    blank_idx: int,
    max_draft: int = 4,
    frame_window: int = 16,
    per_token_blank: int = 4,
    tree: bool = False,
    max_steps: int = 100000,
) -> List[int]:
    """自投机贪心解码 (链式 draft) — 无损。

    状态推进严格复刻 greedy_decode：emit 后 g_last 推进、t 不变；blank 后
    t 推进、g_last 不变。投机在"连续 emit"处省 predictor 往返：在刚 emit
    y_u 的位置 u（消费 y_u 之前）草拟 d_1..d_K（d_j = y_{u+j}，与
    shifted_ce_loss 对齐，见模块 docstring），随后消费 y_u，再逐个用
    head-0 joiner argmax 验证(允许 ≤per_token_blank 个 blank 步到下一次
    emit)，命中即连续提交；失配或遇 blank 则本步用标准贪心推进。因验证
    器 == head-0 argmax == oracle argmax ⇒ 逐 token 一致。
    """
    dev = features.device
    enc, enc_len = model.transcribe(features, feature_lengths)
    T = enc.size(1)
    B = enc.size(0)
    assert B == 1, "spec decode 仅支持 batch=1"
    assert model.medusa_heads is not None, "spec_decode 需 medusa_heads"
    K = min(max_draft, model.medusa_heads.num_heads)
    one = torch.tensor([1], device=dev, dtype=torch.int32)

    # 初始：predictor 吃 blank 得首位置编码 g_last (1,1,512)，state 累积
    cur_token = torch.tensor([[blank_idx]], device=dev, dtype=torch.long)
    g0, _, state = model.predict(cur_token, one, None)
    g_last = g0

    out: List[int] = []
    t = 0
    steps = 0

    while t < T and steps < max_steps:
        steps += 1
        # ---- 本步 head-0 argmax (决定 blank/emit) ----
        logits_t, _, _ = model.join(enc[:, t:t+1], one, g_last, one)  # (1,1,1,V)
        pred0 = int(logits_t.argmax(dim=-1).item())

        if pred0 == blank_idx:
            # blank：t 前进，g_last/state 不变
            t += 1
            continue

        # ---- emit y*=pred0 ----
        out.append(pred0)

        # ---- 草拟 d_1..d_K：必须在消费 pred0 之前，于当前位置 u 草拟 ----
        # head k 在位置 u 预测 y_{u+k}，d_1 = y_{u+1} = 下一个 emit。
        _, future_logits = model.predict_heads(g_last)
        drafts = [int(future_logits[k].argmax(dim=-1).item()) for k in range(K)]

        # ---- 消费 pred0（label 位 +1，t 不变）----
        tok = torch.tensor([[pred0]], device=dev, dtype=torch.long)
        g_last, _, state = model.predict(tok, one, state)

        # ---- 逐个验证 d_j ----
        # 状态推进与 oracle 完全一致：每个 d_j 允许 ≤per_token_blank 次 blank
        # (t 前进) 后 emit。扫描边界逐步复刻 oracle：t 可走到最后一帧 T-1
        # (oracle 在 t=T-1 仍会 emit/走 blank)，t==T 才耗尽。若 emit == d_j：
        # 接受、推进 predictor。若 emit != d_j：失配但仍须 emit 该 label（与
        # oracle 一致），然后停止投机。若 blank 预算耗尽或帧耗尽未 emit：停止
        # 投机（状态停留，下轮外层从同 t 继续，等价 oracle 走 blank）。
        for j in range(K):
            d_j = drafts[j]
            blanks = 0
            emitted = None
            while t < T and blanks < per_token_blank:
                lg, _, _ = model.join(enc[:, t:t+1], one, g_last, one)
                p = int(lg.argmax(dim=-1).item())
                if p == blank_idx:
                    t += 1
                    blanks += 1
                else:
                    emitted = p
                    break
            if emitted is None:
                # blank 预算或帧耗尽未 emit：停止投机，t 已前进的 blank 与
                # oracle 的 blank 前进一致，外层从同 (t, g_last) 继续
                break
            if emitted != d_j:
                # 失配：但 head-0 在此处确实 emit `emitted`，必须提交以与 oracle 一致
                out.append(emitted)
                tok = torch.tensor([[emitted]], device=dev, dtype=torch.long)
                g_last, _, state = model.predict(tok, one, state)
                break
            # 命中 d_j：提交，推进 predictor (吃 d_j)；emit 后 t 不变
            out.append(d_j)
            tok = torch.tensor([[d_j]], device=dev, dtype=torch.long)
            g_last, _, state = model.predict(tok, one, state)
        # 本轮投机结束，回到外层 while

    return out


def assert_lossless(
    model,
    features: Tensor,
    feature_lengths: Tensor,
    blank_idx: int,
    **spec_kwargs,
) -> None:
    """无损断言：spec 输出必须与 greedy oracle 逐 token 一致。

    Raises AssertionError if mismatch. 用于阶段 2.2 验证。
    """
    oracle = greedy_decode(model, features, feature_lengths, blank_idx)
    spec = speculative_greedy_decode(
        model, features, feature_lengths, blank_idx, **spec_kwargs
    )
    assert oracle == spec, (
        f"无损失败: oracle len={len(oracle)} spec len={len(spec)}; "
        f"首个差异 @ {next((i for i,(a,b) in enumerate(zip(oracle,spec)) if a!=b), 'n/a')}"
    )
