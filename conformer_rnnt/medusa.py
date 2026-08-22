"""Medusa 式多头未来标签预测器。

head 0 继续使用原始 Joiner 的 ``joint(t, u)``，作为无损 verifier 的唯一
打分锚点。未来 head 则可以在同一个 joint 上叠加一个零初始化的、流式受限
context delta。这样新 head 在初始化时仍然等价于 base，但训练后能看到
``tgt[u]``、当前声学帧和有限的 right-context，而不是只从当前 joint 猜
``y[u+k]``。

``stream_local`` 的 acoustic window 只包含当前 encoder frame 和配置允许的
lookahead frame；窗口在序列末尾用零填充，不会泄露 right-context 之外的信息。
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
    context 分支 predictor state；默认 ``stream_local`` 使用当前 encoder
    frame、有限 lookahead 和当前 predictor state。context 分支的最后一层
    零初始化，因此所有新 head 初始仍与 base head 完全一致。

    Args:
        input_dim: 特征维度 (=512)，ResBlock 在此维度做残差。
        num_vocab: 词表大小 (含 blank) = num_symbols+1。
        num_heads: K，未来预测头数 (head 1..K)。
        w_vocab: 共享输出投影，必须传 joiner.linear。
        input_mode: ``current_joint`` | ``predictor_only`` | ``stream_local``。
        lookahead_frames: 允许读取的后续 encoder frame 数，不含当前帧。
    """

    def __init__(
        self,
        input_dim: int,
        num_vocab: int,
        num_heads: int,
        w_vocab: torch.nn.Linear,
        input_mode: str = "stream_local",
        lookahead_frames: int = 1,
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
        if input_mode == "stream_local":
            window_dim = (1 + self.lookahead_frames) * input_dim
            self.context_acoustic = torch.nn.Linear(window_dim, input_dim)
            self.context_predictor = torch.nn.Linear(input_dim, input_dim)
            self.context_norm = torch.nn.LayerNorm(input_dim)
            self.context_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
        elif input_mode == "predictor_only":
            self.context_acoustic = None
            self.context_predictor = torch.nn.Linear(input_dim, input_dim)
            self.context_norm = torch.nn.LayerNorm(input_dim)
            self.context_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
        else:
            self.context_acoustic = None
            self.context_predictor = None
            self.context_norm = None
            self.context_mlp = None

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
        target_encodings: Optional[Tensor],
    ) -> Tensor:
        """Build a bounded context delta with shape ``(B,T,L,D)``."""
        if self.input_mode == "current_joint" or self.context_mlp is None:
            return torch.zeros_like(joint)
        if target_encodings is None:
            # Keep direct ``heads(joint)`` diagnostics usable; production paths
            # always pass predictor outputs so stream-local training is active.
            return torch.zeros_like(joint)

        B, T, L, _ = joint.shape
        if target_encodings.size(0) != B or target_encodings.size(1) != L:
            raise ValueError(
                "target_encodings must match joint on batch/label axes: "
                f"joint={tuple(joint.shape)}, targets={tuple(target_encodings.shape)}"
            )

        predictor = self.context_predictor(target_encodings)
        if self.input_mode == "predictor_only":
            fused = predictor.unsqueeze(1).expand(-1, T, -1, -1)
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
            fused = acoustic.unsqueeze(2) + predictor.unsqueeze(1)

        normalized = self.context_norm(fused)
        return self.context_mlp(normalized)

    def forward(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor] = None,
        target_encodings: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """计算所有 head 的 logits (训练: 全网格; 推理: 单点/小窗口)。

        Args:
            joint: ``(B,T,L,D)`` joiner fused features.
            source_encodings: ``(B,T,D)`` encoder outputs for stream-local
                context. At inference this may be a current+lookahead window.
            target_encodings: ``(B,L,D)`` predictor outputs.

        Returns:
            ``(logits_base, logits_future)`` where head 0 is the exact shared
            Joiner projection and each future head predicts ``y[u+k]``.
        """
        logits_base = self.w_vocab(joint)
        medusa_input = joint + self._context_delta(
            joint, source_encodings, target_encodings
        )
        logits_future: List[Tensor] = []
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
) -> Tensor:
    r"""Σ λ_k · CE(head_k 在 (t,u) 的 logits, targets[u+k])。

    ``targets`` 是不含 blank 的真实 label 序列；``blank_idx`` 作为额外
    防线用于屏蔽可能混入的 blank 标签。每个 head 使用自己的
    ``u+k < target_lengths`` mask，所有全 padding 的 batch 不会产生 NaN。
    """
    if not logits_future:
        return targets.new_zeros(())
    B, T, _, V = logits_future[0].shape
    U = targets.size(1)
    total = targets.new_zeros((), dtype=logits_future[0].dtype)
    tl = target_lengths.to(targets.device)
    sl = src_lengths.to(targets.device)
    t_idx = torch.arange(T, device=targets.device).unsqueeze(0)
    valid_t = t_idx < sl.unsqueeze(1)

    for k_idx, logits_k in enumerate(logits_future):
        k = k_idx + 1
        positions = min(U - k, logits_k.size(2))
        if positions <= 0:
            continue
        pred = logits_k[:, :, :positions, :].reshape(-1, V)
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
) -> Tensor:
    r"""β · Σ KL(head_k(u) ‖ base(u+k).detach()).

    Future head ``k`` is aligned with the base distribution at the predictor
    position that corresponds to the same future token. The mask uses
    ``u+k < target_lengths`` independently for every head, so padding and
    unavailable future labels do not dilute the auxiliary loss.
    """
    if beta <= 0 or not logits_future:
        return logits_base.new_zeros(())
    _, T, L, _ = logits_base.shape
    dev = logits_base.device
    sl = src_lengths.to(dev)
    tl = target_lengths.to(dev)
    t_idx = torch.arange(T, device=dev).unsqueeze(0)
    valid_t = t_idx < sl.unsqueeze(1)
    total = logits_base.new_zeros(())

    for k_idx, logits_k in enumerate(logits_future):
        k = k_idx + 1
        positions = min(logits_k.size(2), L - k)
        if positions <= 0:
            continue
        u_idx = torch.arange(positions, device=dev).unsqueeze(0)
        valid_u = (u_idx + k) < tl.unsqueeze(1)
        mask = valid_t[:, : logits_k.size(1), None] & valid_u[:, None, :]
        if not bool(mask.any()):
            continue
        future = logits_k[:, :, :positions, :]
        base = logits_base[:, :, k : k + positions, :]
        log_p0 = F.log_softmax(base, dim=-1).detach()
        log_pk = F.log_softmax(future, dim=-1)
        kl = (log_pk.exp() * (log_pk - log_p0)).sum(dim=-1)
        total = total + (kl * mask).sum() / mask.sum()
    return beta * total
