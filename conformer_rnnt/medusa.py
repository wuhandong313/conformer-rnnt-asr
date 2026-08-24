"""Medusa 式多头未来标签预测器。

head 0 继续使用原始 Joiner 的 ``joint(t, u)``，作为无损 verifier 的唯一
打分锚点。未来 head 则可以在同一个 joint 上叠加一个零初始化的、流式受限
context delta。这样新 head 在初始化时仍然等价于 base，但训练后能看到
``tgt[u]``、当前声学帧和有限的 right-context，而不是只从当前 joint 猜
``y[u+k]``。

``stream_local`` 的 acoustic window 只包含当前 encoder frame 和配置允许的
lookahead frame；窗口在序列末尾用零填充，不会泄露 right-context 之外的信息。
future head 另外接收当前 base logits 的 detached 投影，让它明确知道本轮
已经由 verifier 选出的 token 分布，而不会改变 base/verifier 的输出。
在 ``draft_after_emit`` 模式中，draft 从消费当前 token 后的 predictor state
开始，head k 对齐到 ``y[u+k]``，避免用 ``g_u`` 预测本应由 ``g_{u+1}``
条件化的 token。
所有 future head 仍共享 ``joiner.linear``，因此 verifier 的 base logits
和 head 0 使用完全相同的词表投影。
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# ResBlock (零初始化残差)
# ═══════════════════════════════════════════════════════════════════════════


class MedusaResBlock(torch.nn.Module):
    """Linear→SiLU→Linear(零初始化)→+residual。零初始化 ⇒ 初始 ≈ 恒等。"""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.act = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(dim, dim)
        torch.nn.init.zeros_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.linear2(self.act(self.linear1(x)))


# ═══════════════════════════════════════════════════════════════════════════
# MedusaHeads
# ═══════════════════════════════════════════════════════════════════════════


class MedusaHeads(torch.nn.Module):
    r"""K 个 Medusa 残差头 + 共享词表投影。

    ``current_joint`` 保留旧的 joint-only 消融；``predictor_only`` 只给
    context 分支 predictor state；``stream_local`` 使用当前 encoder
    frame、有限 lookahead 和当前 predictor state。context 分支的最后一层
    零初始化，因此所有新 head 初始仍与 base head 完全一致。

    **链式对齐 (iter-2)**: ``chain_states=True`` 时 head-k 的 context
    分支条件于 teacher-forcing 滚动状态 ``tgt_enc[u+k]`` (吃掉
    ``y_{u+1..u+k-1}`` 之后) 而非统一的 ``tgt_enc[u+1]``。这与推理时
    「draft 喂回 predictor 滚动」严格对应: draft 全对时滚动状态 ==
    tgt_enc[u+k],信息差归零,head-k 只需在正确前缀上做一步预测。

    Args:
        input_dim: 特征维度 (=512)，ResBlock 在此维度做残差。
        num_vocab: 词表大小 (含 blank) = num_symbols+1。
        num_heads: K，未来预测头数 (head 1..K)。
        w_vocab: 共享输出投影，必须传 joiner.linear。
        input_mode: ``current_joint`` | ``predictor_only`` | ``stream_local``。
        lookahead_frames: 允许读取的后续 encoder frame 数，不含当前帧。
        chain_states: 链式对齐 (head-k 条件于 tgt_enc[u+k])。
    """

    def __init__(
        self,
        input_dim: int,
        num_vocab: int,
        num_heads: int,
        w_vocab: torch.nn.Linear,
        input_mode: str = "stream_local",
        lookahead_frames: int = 1,
        draft_after_emit: bool = False,
        chain_states: bool = False,
    ) -> None:
        super().__init__()
        if w_vocab.in_features != input_dim or w_vocab.out_features != num_vocab:
            raise ValueError(
                f"w_vocab 形状不匹配: 传入 {w_vocab.in_features}→{w_vocab.out_features}, "
                f"期望 {input_dim}→{num_vocab}"
            )
        if input_mode not in {"current_joint", "predictor_only", "stream_local"}:
            raise ValueError(
                "Unknown Medusa input_mode {!r}; choose current_joint, "
                "predictor_only, or stream_local".format(input_mode)
            )

        self.num_heads = num_heads
        self.input_mode = input_mode
        self.lookahead_frames = max(0, int(lookahead_frames))
        self.draft_after_emit = bool(draft_after_emit)
        self.chain_states = bool(chain_states)
        self.res_blocks = torch.nn.ModuleList(
            [MedusaResBlock(input_dim) for _ in range(num_heads)]
        )

        # Project the acoustic window and predictor state separately before
        # combining them. This keeps full-grid training memory at O(B*T*L*D)
        # instead of materializing O(B*T*L*window*D) concatenations.
        self.context_norm: Optional[torch.nn.LayerNorm]
        self.context_mlp: Optional[torch.nn.Sequential]
        self.context_acoustic: Optional[torch.nn.Linear]
        self.context_predictor: Optional[torch.nn.Linear]
        self.context_base_logits: Optional[torch.nn.Linear]
        if input_mode == "stream_local":
            window_dim = (1 + self.lookahead_frames) * input_dim
            self.context_acoustic = torch.nn.Linear(window_dim, input_dim)
            self.context_predictor = torch.nn.Linear(input_dim, input_dim)
            self.context_base_logits = torch.nn.Linear(num_vocab, input_dim)
            self.context_norm = torch.nn.LayerNorm(input_dim)
            self.context_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
        elif input_mode == "predictor_only":
            self.context_acoustic = None
            self.context_predictor = torch.nn.Linear(input_dim, input_dim)
            self.context_base_logits = torch.nn.Linear(num_vocab, input_dim)
            self.context_norm = torch.nn.LayerNorm(input_dim)
            self.context_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
        else:
            self.context_acoustic = None
            self.context_predictor = None
            self.context_base_logits = None
            self.context_norm = None
            self.context_mlp = None

        if self.context_base_logits is not None:
            # This feature is added to checkpoints that may already contain a
            # trained context MLP. Zero initialization therefore preserves an
            # old head's behavior while allowing the new signal to be learned.
            torch.nn.init.zeros_(self.context_base_logits.weight)
            torch.nn.init.zeros_(self.context_base_logits.bias)

        if self.context_mlp is not None:
            # Preserve the exact base-logit initialization. The context path
            # becomes active only after its final projection learns.
            torch.nn.init.zeros_(self.context_mlp[-1].weight)
            torch.nn.init.zeros_(self.context_mlp[-1].bias)

        # 与 joiner 共享输出投影：head 0 ≡ joiner 精确一致。
        self.w_vocab = w_vocab

    # ------------------------------------------------------------------
    def _context_delta(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor],
        target_encodings: Tensor,
        base_logits: Optional[Tensor],
        chain_target_encodings: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Build a bounded context delta with shape ``(B,T,L,D)``.

        With ``chain_states`` and a full-sequence ``target_encodings`` the
        head-k context uses the teacher-forcing rolled state at ``u+k``
        (training). At inference ``chain_target_encodings`` supplies the
        draft-rolled states directly (one tensor per head).
        """
        if self.input_mode == "current_joint" or self.context_mlp is None:
            return torch.zeros_like(joint)

        B, T, L, _ = joint.shape
        if target_encodings.size(0) != B:
            raise ValueError(
                "target_encodings must match joint batch size: "
                f"joint={tuple(joint.shape)}, targets={tuple(target_encodings.shape)}"
            )

        if self.chain_states:
            # Chain mode: per-head rolled states.
            # Training: target_encodings is (B, L, D) full sequence; head k
            # at L-position l is conditioned on tgt_enc[:, l+k-1, :].
            # Inference: chain_target_encodings is a list of (B, 1, D)
            # tensors from rolling the predictor on accepted drafts (may be
            # shorter than K when drafting stopped early).
            if chain_target_encodings is not None:
                states = [
                    s if s.dim() == 4 else s.unsqueeze(1).unsqueeze(1)
                    for s in chain_target_encodings
                ]
            else:
                K = self.num_heads
                padded = F.pad(target_encodings, (0, 0, 0, K), value=0.0)
                states = [
                    padded[:, k - 1 : k - 1 + L, :].unsqueeze(2)
                    for k in range(1, K + 1)
                ]
            return self._chain_context_delta(
                joint, source_encodings, states, base_logits
            )

        if target_encodings.size(1) != L:
            raise ValueError(
                "target_encodings must match joint on label axis: "
                f"joint={tuple(joint.shape)}, targets={tuple(target_encodings.shape)}"
            )

        predictor = self.context_predictor(target_encodings)
        if base_logits is None or self.context_base_logits is None:
            return torch.zeros_like(joint)
        if base_logits.shape[:3] != (B, T, L):
            raise ValueError(
                "base_logits must match joint on batch/time/label axes: "
                f"joint={tuple(joint.shape)}, base={tuple(base_logits.shape)}"
            )
        # The verifier distribution is an inference-available summary of the
        # current token decision. Detach it so auxiliary head training cannot
        # alter the base model through this shortcut.
        base_context = self.context_base_logits(base_logits.detach())
        if self.input_mode == "predictor_only":
            fused = predictor.unsqueeze(1).expand(-1, T, -1, -1) + base_context
        else:
            if source_encodings is None:
                return torch.zeros_like(joint)
            if source_encodings.size(0) != B:
                raise ValueError(
                    "source_encodings must match joint batch size: "
                    f"joint={tuple(joint.shape)}, source={tuple(source_encodings.shape)}"
                )
            width = 1 + self.lookahead_frames
            source = source_encodings
            required = T + self.lookahead_frames
            if source.size(1) < required:
                source = F.pad(source, (0, 0, 0, required - source.size(1)))
            windows = source.unfold(1, width, 1)
            # Tensor.unfold appends the window axis after D: (B,T,D,W).
            windows = windows.permute(0, 1, 3, 2).contiguous()[:, :T]
            acoustic = self.context_acoustic(windows.flatten(start_dim=2))
            fused = acoustic.unsqueeze(2) + predictor.unsqueeze(1) + base_context

        normalized = self.context_norm(fused)
        return self.context_mlp(normalized)

    def _chain_context_delta(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor],
        states: List[Tensor],
        base_logits: Optional[Tensor],
    ) -> List[Tensor]:
        """Per-head context deltas for chain mode.

        Each entry of ``states`` is the rolled predictor state for the
        corresponding head: ``(B, L, 1, D)`` at training, or ``(B, 1, 1, D)``
        at inference (broadcast over the T axis; L is 1 there). The list may
        be shorter than ``num_heads`` when inference drafting stopped early —
        remaining heads get a zero delta.

        Returns a list of ``num_heads`` tensors, each shaped like ``joint``.
        """
        B, T, L, D = joint.shape
        K = self.num_heads

        # Precompute the shared stream-local acoustic term once (B,T,D).
        acoustic_term: Optional[Tensor] = None
        if self.input_mode == "stream_local":
            if source_encodings is None:
                raise ValueError("stream_local chain mode requires source_encodings")
            if source_encodings.size(0) != B:
                raise ValueError(
                    "source_encodings must match joint batch size: "
                    f"joint={tuple(joint.shape)}, source={tuple(source_encodings.shape)}"
                )
            width = 1 + self.lookahead_frames
            source = source_encodings
            required = T + self.lookahead_frames
            if source.size(1) < required:
                source = F.pad(source, (0, 0, 0, required - source.size(1)))
            windows = source.unfold(1, width, 1)
            windows = windows.permute(0, 1, 3, 2).contiguous()[:, :T]
            acoustic_term = self.context_acoustic(windows.flatten(start_dim=2))

        base_term: Optional[Tensor] = None
        if base_logits is not None and self.context_base_logits is not None:
            base_term = self.context_base_logits(base_logits.detach())

        deltas: List[Tensor] = []
        for k in range(K):
            if k >= len(states):
                deltas.append(torch.zeros_like(joint))
                continue
            state = states[k]
            if state.dim() != 4 or state.size(0) != B:
                raise ValueError(
                    "chain state must be (B,L,1,D) or (B,1,1,D); got "
                    f"{tuple(state.shape)}"
                )
            if state.size(1) == 1 and L != 1:
                state_b = state.expand(B, L, 1, D)
            elif state.size(1) == L:
                state_b = state
            else:
                state_b = F.pad(state, (0, 0, 0, 0, 0, L - state.size(1)))
            predictor = self.context_predictor(state_b.squeeze(2))  # (B,L,D)
            fused = predictor.unsqueeze(1)  # (B,1,L,D)
            if base_term is not None:
                fused = fused + base_term
            if acoustic_term is not None:
                fused = fused + acoustic_term.unsqueeze(2)
            normalized = self.context_norm(fused)
            deltas.append(self.context_mlp(normalized))
        return deltas

    def forward(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor] = None,
        target_encodings: Optional[Tensor] = None,
        chain_target_encodings: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """计算所有 head 的 logits (训练: 全网格; 推理: 单点/小窗口)。

        Args:
            joint: ``(B,T,L,D)`` joiner fused features.
            source_encodings: ``(B,T,D)`` encoder outputs for stream-local
                context. At inference this may be a current+lookahead window.
            target_encodings: ``(B,L,D)`` predictor outputs.
            chain_target_encodings: chain 模式推理时使用 — K 个
                ``(B,1,D)`` draft-rolled predictor states。

        Returns:
            ``(logits_base, logits_future)`` where head 0 is the exact shared
            Joiner projection and each future head predicts ``y[u+k]``.
        """
        logits_base = self.w_vocab(joint)
        if self.chain_states:
            deltas = self._context_delta(
                joint, source_encodings, target_encodings,
                logits_base, chain_target_encodings,
            )
            logits_future: List[Tensor] = []
            for k, block in enumerate(self.res_blocks):
                logits_future.append(self.w_vocab(block(joint + deltas[k])))
            return logits_base, logits_future

        medusa_input = joint + self._context_delta(
            joint, source_encodings, target_encodings, logits_base
        )
        logits_future = []
        for block in self.res_blocks:
            logits_future.append(self.w_vocab(block(medusa_input)))
        return logits_base, logits_future


# ═══════════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════════


def shifted_ce_loss(
    logits_future: List[Tensor],
    targets: Tensor,
    weights: List[float],
    target_lengths: Tensor,
    src_lengths: Tensor,
    blank_idx: Optional[int] = None,
    future_state_offset: int = 0,
) -> Tensor:
    r"""Σ λ_k · CE(head_k 在 (t,u) 的 logits, targets[u+k])。

    ``targets`` 是不含 blank 的真实 label 序列；``blank_idx`` 作为额外
    防线用于屏蔽可能混入的 blank 标签。每个 head 使用自己的
    ``u+k < target_lengths`` mask，所有全 padding 的 batch 不会产生 NaN。
    """
    if not logits_future:
        return targets.new_zeros(())
    future_state_offset = max(0, int(future_state_offset))
    B, T, _, V = logits_future[0].shape
    U = targets.size(1)
    total = targets.new_zeros((), dtype=logits_future[0].dtype)
    tl = target_lengths.to(targets.device)
    sl = src_lengths.to(targets.device)
    t_idx = torch.arange(T, device=targets.device).unsqueeze(0)
    valid_t = t_idx < sl.unsqueeze(1)

    for k_idx, logits_k in enumerate(logits_future):
        k = k_idx + 1
        positions = min(U - k, logits_k.size(2) - future_state_offset)
        if positions <= 0:
            continue
        pred = logits_k[
            :, :, future_state_offset : future_state_offset + positions, :
        ].reshape(-1, V)
        label = targets[:, k : k + positions]
        label = label.unsqueeze(1).expand(-1, T, -1)
        u_idx = torch.arange(positions, device=targets.device).unsqueeze(0)
        valid_u = (u_idx + k) < tl.unsqueeze(1)
        valid = valid_t.unsqueeze(2) & valid_u.unsqueeze(1)
        if blank_idx is not None:
            valid = valid & (label != blank_idx)
        flat_valid = valid.reshape(-1)
        if not flat_valid.any():
            continue
        ce = F.cross_entropy(
            pred[flat_valid], label.reshape(-1)[flat_valid].long(), reduction="mean"
        )
        total = total + weights[k_idx] * ce
    return total


def kl_to_base_loss(
    logits_future: List[Tensor],
    logits_base: Tensor,
    beta: float,
    src_lengths: Tensor,
    target_lengths: Tensor,
    future_state_offset: int = 0,
    chain_align: bool = False,
) -> Tensor:
    r"""β · Σ KL(head_k(u) ‖ base(u+k).detach()).

    Future head ``k`` is aligned with the base distribution at the predictor
    position that corresponds to the same future token. The mask uses
    ``u+k < target_lengths`` independently for every head, so padding and
    unavailable future labels do not dilute the auxiliary loss.

    With ``chain_align=True`` (chain mode), head ``k`` at L-position ``l`` is
    conditioned on the rolled state ``tgt_enc[l+k-1]``, matching the base
    logits at the same L-position (base@l+k-1), not ``l+k``.
    """
    if beta <= 0 or not logits_future:
        return logits_base.new_zeros(())
    future_state_offset = max(0, int(future_state_offset))
    _, T, L, _ = logits_base.shape
    dev = logits_base.device
    sl = src_lengths.to(dev)
    tl = target_lengths.to(dev)
    t_idx = torch.arange(T, device=dev).unsqueeze(0)
    valid_t = t_idx < sl.unsqueeze(1)
    total = logits_base.new_zeros(())

    for k_idx, logits_k in enumerate(logits_future):
        k = k_idx + 1
        positions = min(logits_k.size(2) - future_state_offset, L - k)
        if positions <= 0:
            continue
        u_idx = torch.arange(positions, device=dev).unsqueeze(0)
        valid_u = (u_idx + k) < tl.unsqueeze(1)
        mask = valid_t[:, : logits_k.size(1), None] & valid_u[:, None, :]
        if not bool(mask.any()):
            continue
        future = logits_k[
            :, :, future_state_offset : future_state_offset + positions, :
        ]
        # Chain mode: head-k@l is conditioned on tgt_enc[l+k-1] and predicts
        # targets[l+k-1]; align to the base distribution at l+k-1.
        base_shift = (k - 1) if chain_align else k
        base = logits_base[:, :, base_shift : base_shift + positions, :]
        log_p0 = F.log_softmax(base, dim=-1).detach()
        log_pk = F.log_softmax(future, dim=-1)
        kl = (log_pk.exp() * (log_pk - log_p0)).sum(dim=-1)
        total = total + (kl * mask).sum() / mask.sum()
    return beta * total


def base_next_token_ce_loss(
    logits_future: List[Tensor],
    logits_base: Tensor,
    weights: List[float],
    target_lengths: Tensor,
    src_lengths: Tensor,
    blank_idx: int,
    future_state_offset: int = 1,
    max_blank: int = 4,
    chain_align: bool = False,
) -> Tensor:
    r"""Distill the verifier's next emitted token into future heads.

    In post-emission mode, head ``k`` is queried at predictor state ``u+1``
    while the verifier reaches the corresponding token at state ``u+k``.
    The verifier may emit blank for a few acoustic frames before that token,
    so the target is the first non-blank argmax in the bounded frame window.
    This auxiliary objective matches the actual acceptance event and does not
    backpropagate through the base logits.
    """
    if not logits_future:
        return logits_base.new_zeros(())
    future_state_offset = max(0, int(future_state_offset))
    max_blank = max(0, int(max_blank))
    _, T, L, _ = logits_base.shape
    dev = logits_base.device
    sl = src_lengths.to(dev)
    tl = target_lengths.to(dev)
    valid_t = torch.arange(T, device=dev).unsqueeze(0) < sl.unsqueeze(1)
    base_ids = logits_base.argmax(dim=-1).detach()
    total = logits_base.new_zeros(())

    for k_idx, logits_k in enumerate(logits_future):
        k = k_idx + 1
        positions = min(logits_k.size(2) - future_state_offset, L - k)
        if positions <= 0:
            continue
        # State u+k is the verifier state used to emit the k-th future token.
        # Chain mode: head-k@l predicts targets[l+k-1] conditioned on the
        # rolled state tgt_enc[l+k-1]; the verifier's decision for that token
        # lives at base L-position l+k-1.
        base_shift = (k - 1) if chain_align else k
        candidates = base_ids[:, :, base_shift : base_shift + positions]
        chosen = torch.full_like(candidates, blank_idx)
        found = torch.zeros_like(candidates, dtype=torch.bool)
        for delta in range(max_blank + 1):
            if delta == 0:
                shifted = candidates
            else:
                shifted = F.pad(
                    candidates[:, delta:, :],
                    (0, 0, 0, delta),
                    value=blank_idx,
                )
            nonblank = shifted != blank_idx
            take = nonblank & ~found
            chosen = torch.where(take, shifted, chosen)
            found = found | nonblank

        u_idx = torch.arange(positions, device=dev).unsqueeze(0)
        valid_u = (u_idx + k) < tl.unsqueeze(1)
        valid = valid_t.unsqueeze(2) & valid_u.unsqueeze(1) & found
        flat_valid = valid.reshape(-1)
        if not flat_valid.any():
            continue
        pred = logits_k[
            :, :, future_state_offset : future_state_offset + positions, :
        ].reshape(-1, logits_k.size(-1))
        ce = F.cross_entropy(pred[flat_valid], chosen.reshape(-1)[flat_valid].long())
        total = total + weights[k_idx] * ce
    return total
