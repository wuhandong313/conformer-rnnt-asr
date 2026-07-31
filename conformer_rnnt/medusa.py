"""Medusa 式多头未来标签预测器 (plan §3.3)。

挂在受限因果自注意力 predictor trunk 之上：
    g (B, U+1, 512)  ← predictor trunk 输出
    head 0 (base) : logits_0 = W_vocab(g)                 预测 targets[u]   (下一标签)
    head k (1..K) : logits_k = W_vocab(ResBlock_k(g))     预测 targets[u+k] (未来第 k 标签)

K 个 ResBlock 对应 head 1..K (均做 aux CE)；head 0 = trunk 直投影 (不经 ResBlock)，
不单独训 CE —— 它由 RNN-T loss (经 joiner) 训 —— 只作 KL 的 detach 锚。

零初始化：每个 ResBlock 的第二个 Linear 初始化为 0 ⇒ 初始 ResBlock(g)≈g ⇒
head k 初始 logits == head 0 logits (draft 初始 = base，训练稳定；Medusa 关键技巧)。

共享 W_vocab (512→V，V=num_symbols+1 含 blank) 对应 Medusa 共享 output_proj。
参数开销：K=4 ⇒ 4×0.53M + 38K ≈ 2.1M。
"""

from typing import List, Tuple

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
        # 零初始化第二个 Linear：初始 linear2(act(linear1(x))) = 0 ⇒ out = x
        torch.nn.init.zeros_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.linear2(self.act(self.linear1(x)))


# ═══════════════════════════════════════════════════════════════════════════
# MedusaHeads
# ═══════════════════════════════════════════════════════════════════════════


class MedusaHeads(torch.nn.Module):
    r"""K 个 Medusa 残差头 + 共享词表投影。

    Args:
        input_dim: trunk 输出维度 (=512)，ResBlock 在此维度做残差。
        num_vocab: 词表大小 (含 blank) = num_symbols+1。
        num_heads: K，未来预测头数 (head 1..K)。0 = 关闭 (单头消融)。
    """

    def __init__(self, input_dim: int = 512, num_vocab: int = 75, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.res_blocks = torch.nn.ModuleList(
            [MedusaResBlock(input_dim) for _ in range(num_heads)]
        )
        # 共享词表投影 (head 0 与 head 1..K 共用)
        self.w_vocab = torch.nn.Linear(input_dim, num_vocab)

    # ------------------------------------------------------------------
    def forward(self, g: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """计算所有 head 的 logits。

        Args:
            g: (B, L, input_dim) trunk 输出序列 (L=U+1，含 blank 起始位)。

        Returns:
            (logits_base, logits_future):
                logits_base   : (B, L, num_vocab)  head 0 = W_vocab(g)，预测下一标签
                logits_future : List[K] 个 (B, L, num_vocab)，head k 预测 targets[u+k]
        """
        logits_base = self.w_vocab(g)
        logits_future: List[Tensor] = []
        for block in self.res_blocks:
            logits_future.append(self.w_vocab(block(g)))
        return logits_base, logits_future


# ═══════════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════════


def shifted_ce_loss(
    logits_future: List[Tensor],
    targets: Tensor,
    weights: List[float],
) -> Tensor:
    r"""Σ_{k=1..K} λ_k · CE(head_k 位置 u 的 logits, targets[u+k])。

    偏移按本仓库 blank 前置布局钉死：
        prepended_targets = [blank, y_0, ..., y_{U-1}]  (长度 U+1)
        g 位置 u 输入 prepended_targets[u] ⇒ head k 在位 u 预测 targets[u+k]
        有效位 u ∈ [0, U-1-k] (共 U-k 个)；label = targets[k:U]

    Args:
        logits_future: List[K] 个 (B, L, V)，L≥U+1 (trunk 序列长)。
        targets: (B, U) 真实标签 id (0..V-2，不含 blank)。
        weights: [λ_1, ..., λ_K]。
    """
    U = targets.size(1)
    V = logits_future[0].size(-1)
    total = targets.new_zeros(())
    for k_idx, logits_k in enumerate(logits_future):  # head k = k_idx+1
        k = k_idx + 1
        if U - k <= 0:
            continue  # 序列太短，该 head 无有效位置
        pred = logits_k[:, : U - k, :].reshape(-1, V)   # (B*(U-k), V)
        label = targets[:, k: U].reshape(-1)            # (B*(U-k),)
        ce = F.cross_entropy(pred, label, reduction="mean", ignore_index=-1)
        total = total + weights[k_idx] * ce
    return total


def kl_to_base_loss(
    logits_future: List[Tensor],
    logits_base: Tensor,
    beta: float,
) -> Tensor:
    r"""β · Σ_{k=1..K} KL(softmax(head_k) ‖ softmax(head_0).detach())。

    同位置分布正则：防 draft 头漂离 base 分布先验。head 0 detach (不回传)。
    """
    if beta <= 0 or len(logits_future) == 0:
        return logits_base.new_zeros(())
    log_p0 = F.log_softmax(logits_base, dim=-1).detach()
    total = logits_base.new_zeros(())
    for logits_k in logits_future:
        log_pk = F.log_softmax(logits_k, dim=-1)
        # KL(p_k || p_0) = Σ p_k (log p_k - log p_0)
        kl = (log_pk.exp() * (log_pk - log_p0)).sum(dim=-1).mean()
        total = total + kl
    return beta * total


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════


def medusa_heads_from_config(config: dict) -> "MedusaHeads":
    """从 config['model']['medusa'] 段构建。num_heads=0 返回 None (关闭)。"""
    m = config.get("model", config)
    med = m.get("medusa", {})
    num_heads = med.get("num_heads", 4)
    if num_heads == 0:
        return None
    num_vocab = m.get("num_symbols", 74) + 1  # +blank
    encoding_dim = m.get("encoding_dim", 512)
    return MedusaHeads(input_dim=encoding_dim, num_vocab=num_vocab, num_heads=num_heads)