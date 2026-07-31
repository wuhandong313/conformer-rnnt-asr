"""自投机贪心解码 (plan §3.5) — 无损。

核心思想：Medusa heads 在当前 predictor 末位置草拟 K 个未来 label，用
head-0 (trunk 经 joiner 的 argmax) 在一个帧窗内重放 RNN-T 贪心来验证，
接受与 head-0 一致的最长前缀 m。因验证器 == 原贪心的 argmax，输出与
width=1 oracle **逐 token 完全一致**（无损，PER 不牺牲）。

收益：每接受 m 个 draft 省 m 次外层 (predictor 调用 + Python 往返)，
代价一次 GPU 友好的 joiner 帧窗批量广播。

链式 draft (默认)：同头单路径 argmax(d_j)。
树式 (tree=True)：每头取 top-k 路径，取可接受最长者（同一批 heads，仅改候选）。

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
    """自投机贪心解码 (链式 draft)。

    每轮：
      1. 在当前末位置用 heads 草拟 d_1..d_K (head k argmax)。
      2. 在帧窗 [t, t+W] 上用 head-0 joiner argmax 重放贪心验证：
         每个 emitted label 必须依次 == d_j；接受最长前缀 m。
      3. 提交 [d_1..d_m]，状态落在接受末端；失配 m=0 → 走一步标准贪心。

    无损性：验证器 == head-0 argmax == greedy_decode 的 argmax ⇒ 逐 token 一致。
    """
    dev = features.device
    enc, enc_len = model.transcribe(features, feature_lengths)
    T = enc.size(1)
    B = enc.size(0)
    assert B == 1, "spec decode 仅支持 batch=1"
    assert model.medusa_heads is not None, "spec_decode 需 medusa_heads"
    K = min(max_draft, model.medusa_heads.num_heads)
    one = torch.tensor([1], device=dev, dtype=torch.int32)

    # 初始：predictor 吃 blank，得 g_last (1,1,512)，state 累积
    cur_token = torch.tensor([[blank_idx]], device=dev, dtype=torch.long)
    g, _, state = model.predict(cur_token, one, None)
    g_last = g  # (1,1,512)

    out: List[int] = []
    t = 0
    steps = 0

    while t < T and steps < max_steps:
        steps += 1
        # ---- head-0 当前 argmax (作为本步 emit 候选 y*) ----
        # joiner(enc[t], g_last) 的 argmax 决定走 blank 还是 emit
        logits_t, _, _ = model.join(enc[:, t:t+1], one, g_last, one)  # (1,1,1,V)
        pred0 = int(logits_t.argmax(dim=-1).item())

        if pred0 == blank_idx:
            # 当前帧发 blank，前进一帧，label 位不变
            t += 1
            continue

        # ---- emit y*=pred0；先草拟 draft ----
        # heads 在当前 g_last 输出 base + future
        base_logits, future_logits = model.predict_heads(g_last)  # 各 (1,1,V)
        # head k (future[k-1]) 预测 targets[u+k] ⇒ 草拟 d_k
        drafts: List[int] = []
        for k in range(K):
            d = int(future_logits[k].argmax(dim=-1).item())
            drafts.append(d)

        # ---- 验证：在帧窗 [t, t+W] 内重放贪心，接受与 drafts 一致的最长前缀 ----
        # 先提交 y*=pred0（已确认 emit）。然后逐个验证 d_1, d_2, ...
        # 验证每个 d_j：需要从当前 (t, u) 出发，允许 ≤ per_token_blank 个 blank 步
        # 后必须 emit 一个 label；该 label 必须 == d_j。
        accepted: List[int] = []
        # 临时推进状态（用副本，失配要回滚）。注意 predictor state 是 append-only
        # KV cache，回滚 = 用保存的旧 state 切片。
        t_cur = t
        # 我们需要逐个 emit 的 predictor 编码；先算 y* 的 g
        # 注：head-0 argmax 在 g_last 已是 pred0；下面逐步推进。
        # 为安全正确，验证阶段我们用"标准贪心一步"向前模拟：
        cur_g = g_last
        cur_state = state
        rollback_state = state
        rollback_g = g_last
        rollback_t = t
        rollback_out_len = len(out)

        # 提交 y*
        out.append(pred0)
        # predictor 吃 pred0 → 新 g
        tok = torch.tensor([[pred0]], device=dev, dtype=torch.long)
        cur_g, _, cur_state = model.predict(tok, one, cur_state)
        # 现在验证 d_1..: 从 cur_g / cur_state 出发，帧窗内找下一个 emit
        # 帧窗预算：剩余可用帧 = frame_window（从 t 起）。但我们已 emit 0 个 blank 步。
        # 简化：对每个 d_j，允许最多 per_token_blank 次 blank 前进，然后 emit 必须 == d_j。
        ok = True
        for j in range(K):
            d_j = drafts[j]
            # 在帧窗内前进 blank 直到 emit 或超限
            blanks_used = 0
            emitted = None
            while blanks_used < per_token_blank and t_cur < T:
                # 当前 (t_cur, 末 u) 的 argmax
                lg, _, _ = model.join(enc[:, t_cur:t_cur+1], one, cur_g, one)
                p = int(lg.argmax(dim=-1).item())
                if p == blank_idx:
                    t_cur += 1
                    blanks_used += 1
                else:
                    emitted = p
                    break
            if emitted is None:
                # 帧窗耗尽仍未 emit → 不能接受 d_j (及之后)
                ok = False
                break
            if emitted != d_j:
                ok = False
                break
            # 接受 d_j
            accepted.append(d_j)
            out.append(d_j)
            rollback_state = cur_state
            rollback_g = cur_g
            rollback_t = t_cur
            # 推进 predictor：吃 d_j
            tok = torch.tensor([[d_j]], device=dev, dtype=torch.long)
            cur_g, _, cur_state = model.predict(tok, one, cur_state)

        if ok and len(accepted) == K:
            # 全部接受，t_cur 是末 emit 的帧；下一步从 t_cur+1? 不——
            # 我们已 emit，下一步应在新 label 位继续。把 t 指针留 t_cur，
            # 下轮 head-0 会决定 blank/emit。
            t = t_cur
            state = cur_state
            g_last = cur_g
        elif len(accepted) > 0:
            # 部分接受：提交 accepted，状态回滚到最后一个接受处
            t = rollback_t
            state = rollback_state
            g_last = rollback_g
        else:
            # m=0：只提交 y*=pred0（已 append），状态用 pred0 后的 cur_g/state
            # 但要回滚 out 里多加的（accepted 为空，没多加）
            t = t  # t 停在 emit 帧本步；下轮 head-0 在新 g 位继续
            state = cur_state  # pred0 后的状态（已推进一格 label）
            g_last = cur_g
        # 继续下轮

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
