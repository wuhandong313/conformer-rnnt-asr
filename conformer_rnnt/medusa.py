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
            # iter-6: residual capacity expansion. context_mlp keeps the
            # exact iter-4 two-layer structure (warm-start loads it fully);
            # context_mlp_extra adds a parallel deeper path whose output
            # projection is zero-initialized — at step 0 the module is
            # bit-identical to iter-4.
            self.context_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
            self.context_mlp_extra = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
            torch.nn.init.zeros_(self.context_mlp_extra[-1].weight)
            torch.nn.init.zeros_(self.context_mlp_extra[-1].bias)
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

        # Draft-conditioned chaining (iter-3): head-k (k>=2) additionally
        # consumes the embedding of the previous head's draft token. Training
        # uses the previous head's argmax (detached) on the same grid, so the
        # train and inference conditions are exactly isomorphic — no LSTM
        # rolling and no emit-gap sequence dependence.
        self.chain_draft_embed: Optional[torch.nn.Embedding]
        self.chain_draft_proj: Optional[torch.nn.Linear]
        self.chain_state_proj: Optional[torch.nn.Linear]
        if self.chain_states and num_heads > 1:
            self.chain_draft_embed = torch.nn.Embedding(num_vocab, 128)
            self.chain_draft_proj = torch.nn.Linear(128, input_dim)
            torch.nn.init.zeros_(self.chain_draft_proj.weight)
            torch.nn.init.zeros_(self.chain_draft_proj.bias)
            # iter-4: rolled-language-state input. Training uses the
            # teacher-forcing tgt_enc[l+k-1]; inference uses the predictor
            # rolled on accepted drafts (bit-exact when drafts hit).
            self.chain_state_proj = torch.nn.Linear(input_dim, input_dim)
            torch.nn.init.zeros_(self.chain_state_proj.weight)
            torch.nn.init.zeros_(self.chain_state_proj.bias)
            # iter-5 (abandoned): frame-level attention over the acoustic
            # window. Training destabilized (PER 0.07 → 0.50, greedy output
            # corrupted) — the modules are kept for checkpoint compatibility
            # but disabled via win_attn_enabled=False.
            self.win_attn_enabled = False
            self.win_attn_key = torch.nn.Linear(input_dim, 128)
            self.win_attn_val = torch.nn.Linear(input_dim, input_dim)
            self.win_attn_query = torch.nn.Linear(input_dim, 128)
            self.win_attn_out = torch.nn.Linear(input_dim, input_dim)
            torch.nn.init.zeros_(self.win_attn_out.weight)
            torch.nn.init.zeros_(self.win_attn_out.bias)
        else:
            self.chain_draft_embed = None
            self.chain_draft_proj = None
            self.chain_state_proj = None

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
            # Draft-conditioned chaining (iter-3) — handled in forward(),
            # which builds heads sequentially. This path is unreachable in
            # production; raise to catch stale callers.
            raise RuntimeError(
                "chain mode builds heads in forward(); _context_delta "
                "should not be called with chain_states=True"
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
        extra = getattr(self, "context_mlp_extra", None)
        return self.context_mlp(normalized) + (extra(normalized) if extra is not None else 0.0)

    def _chain_shared_terms(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor],
        target_encodings: Tensor,
        base_logits: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Shared chain inputs.

        Returns (predictor, flat_acoustic, base_term, attn_frames) where
        ``attn_frames`` is ``(B, T, W, D)`` — the raw per-frame window tensor
        for the iter-5 attention (None when not stream_local).
        """
        B, T, L, D = joint.shape
        predictor = self.context_predictor(target_encodings)  # (B,L,D)
        acoustic_term: Optional[Tensor] = None
        attn_frames: Optional[Tensor] = None
        if self.input_mode == "stream_local":
            if source_encodings is None:
                raise ValueError("stream_local chain mode requires source_encodings")
            width = 1 + self.lookahead_frames
            source = source_encodings
            required = T + self.lookahead_frames
            if source.size(1) < required:
                source = F.pad(source, (0, 0, 0, required - source.size(1)))
            windows = source.unfold(1, width, 1)
            windows = windows.permute(0, 1, 3, 2).contiguous()[:, :T]  # (B,T,W,D)
            attn_frames = windows
            acoustic_term = self.context_acoustic(windows.flatten(start_dim=2))
        base_term: Optional[Tensor] = None
        if base_logits is not None and self.context_base_logits is not None:
            base_term = self.context_base_logits(base_logits.detach())
        return predictor, acoustic_term, base_term, attn_frames

    def _chain_head_delta(
        self,
        shared: Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]],
        joint: Tensor,
        k: int,
        draft_ids: Optional[Tensor] = None,
        prev_logits: Optional[Tensor] = None,
        rolled_state: Optional[Tensor] = None,
    ) -> Tensor:
        """Context delta for chain head ``k`` (0-based).

        ``draft_ids`` (inference, (B,1,1) long) or ``prev_logits`` (training,
        (B,T,L) detached) supply the previous head's draft embedding. For
        ``k == 0`` neither is used — the head is the plain non-chain head.

        ``rolled_state`` (iter-4) is the rolled language state for head k:
        ``(B, L, D)`` at training (tgt_enc shifted by k) or ``(B, 1, D)`` at
        inference (predictor advanced on accepted drafts). It carries the LM
        context the verifier will have when reaching this token. It also
        serves as the attention query (iter-5) over the acoustic window.
        """
        predictor, acoustic_term, base_term, attn_frames = shared
        fused = predictor.unsqueeze(1)  # (B,1,L,D)
        if base_term is not None:
            fused = fused + base_term
        if acoustic_term is not None:
            fused = fused + acoustic_term.unsqueeze(2)
        if rolled_state is not None and self.chain_state_proj is not None:
            state_term = self.chain_state_proj(rolled_state)
            if state_term.dim() == 3:
                state_term = state_term.unsqueeze(1)  # (B,1,L,D) or (B,1,1,D)
            fused = fused + state_term
            # iter-5: frame attention — query from this head's rolled state,
            # keys/values over the (B,T,W,D) window frames. Output mixes
            # additively into the fused context.
            if attn_frames is not None and getattr(self, "win_attn_enabled", False):
                B, T, L2, D = fused.shape
                frames = attn_frames  # (B,T,W,D)
                W = frames.size(2)
                keys = self.win_attn_key(frames)          # (B,T,W,128)
                vals = self.win_attn_val(frames)          # (B,T,W,D)
                # query: (B, L, D) → (B, L, 128); broadcast over T
                q = self.win_attn_query(rolled_state)     # (B,L,128) or (B,1,128)
                scores = torch.einsum(
                    "bld,btwd->bltw", q, keys
                ) / (128 ** 0.5)                          # (B,L,T,W)
                weights = F.softmax(scores, dim=-1)
                attended = torch.einsum(
                    "bltw,btwd->bltd", weights, vals
                )                                         # (B,L,T,D)
                out = self.win_attn_out(attended)         # (B,L,T,D)
                fused = fused + out.permute(0, 2, 1, 3)   # → (B,T,L,D)
        if k >= 1:
            if draft_ids is not None:
                ids = draft_ids
            elif prev_logits is not None:
                ids = prev_logits.argmax(dim=-1).detach()  # (B,T,L)
            else:
                ids = None
            if ids is not None:
                if ids.dim() == 4:
                    ids = ids.squeeze(-1)
                emb = self.chain_draft_proj(self.chain_draft_embed(ids))
                if emb.dim() == 3:
                    emb = emb.unsqueeze(1)  # (B,1,1,D) inference → broadcast
                fused = fused + emb
        normalized = self.context_norm(fused)
        extra = getattr(self, "context_mlp_extra", None)
        return self.context_mlp(normalized) + (extra(normalized) if extra is not None else 0.0)

    def forward(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor] = None,
        target_encodings: Optional[Tensor] = None,
        chain_drafts: Optional[List[int]] = None,
        chain_states: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """计算所有 head 的 logits (训练: 全网格; 推理: 单点/小窗口)。

        Args:
            joint: ``(B,T,L,D)`` joiner fused features.
            source_encodings: ``(B,T,D)`` encoder outputs for stream-local
                context. At inference this may be a current+lookahead window.
            target_encodings: ``(B,L,D)`` predictor outputs.
            chain_drafts: chain 模式推理时使用 — 已确定的前序 draft id 列表
                (d_1..d_{k-1}); head-k 消费 chain_drafts[k-1]。
            chain_states: chain 模式推理时使用 (iter-4) — K 个 (B,1,D)
                predictor 滚动状态; head-k 消费 chain_states[k]
                (k=0 为 draft 点状态本身)。

        Returns:
            ``(logits_base, logits_future)`` where head 0 is the exact shared
            Joiner projection and each future head predicts ``y[u+k]``.
        """
        logits_base = self.w_vocab(joint)
        if self.chain_states:
            # Draft-conditioned chain: heads are built sequentially — head-k
            # consumes the previous head's draft (training: actual head
            # logits' argmax over the grid, detached; inference: fixed ids
            # from chain_drafts). _chain_shared_terms is precomputed once.
            K = self.num_heads
            shared = self._chain_shared_terms(
                joint, source_encodings, target_encodings, logits_base
            )
            logits_future: List[Tensor] = []
            inference = chain_drafts is not None
            # Rolled states: inference passes them via chain_states; training
            # shifts tgt_enc so head-k sees the state after consuming
            # y_l..y_{l+k-1} (teacher forcing; bit-exact with inference when
            # drafts are accepted).
            rolled: List[Optional[Tensor]] = [None] * K
            if inference:
                for k in range(K):
                    if chain_states is not None and k < len(chain_states):
                        rolled[k] = chain_states[k]
            else:
                L = joint.size(2)
                padded = F.pad(target_encodings, (0, 0, 0, K), value=0.0)
                for k in range(K):
                    rolled[k] = padded[:, k : k + L, :]
            for k in range(K):
                draft_tok: Optional[Tensor] = None
                prev = None
                if k >= 1:
                    if inference:
                        if k - 1 < len(chain_drafts):
                            draft_tok = torch.full(
                                (joint.size(0), 1, 1), int(chain_drafts[k - 1]),
                                dtype=torch.long, device=joint.device,
                            )
                        # else: incremental inference — this head's draft is
                        # not fixed yet; its logits are unused by the caller.
                        # Fall back to the training-style previous head argmax
                        # so the graph stays well-formed.
                        else:
                            prev = logits_future[k - 1].detach()
                    else:
                        prev = logits_future[k - 1].detach()
                delta_k = self._chain_head_delta(
                    shared, joint, k,
                    draft_ids=draft_tok,
                    prev_logits=prev,
                    rolled_state=rolled[k],
                )
                logits_k = self.w_vocab(self.res_blocks[k](joint + delta_k))
                logits_future.append(logits_k)
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
