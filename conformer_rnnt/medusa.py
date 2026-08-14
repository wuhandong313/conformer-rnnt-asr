"""Medusa 式多头未来标签预测器。

挂在 joiner 融合特征之上：
    joint[t][u] = activation(src_enc[t] + tgt_enc[u])  (B, T, U+1, 512)
    head 0 (base) : logits_0 = W_vocab(joint)            仅作 KL 锚 (不训 CE)
    head k (1..K) : logits_k = W_vocab(ResBlock_k(joint)) 预测 y_{u+k} (后续 token)

K 个 ResBlock 对应 head 1..K (均做 aux CE)；head 0 = 直投影 (不经 ResBlock)，
不单独训 CE —— 只作 KL 的 detach 锚。

零初始化：每个 ResBlock 的第二个 Linear 初始化为 0 ⇒ 初始 ResBlock(x)≈x ⇒
head k 初始 logits == head 0 logits (draft 初始 = base，训练稳定；Medusa 关键技巧)。

共享 W_vocab (512→V，V=num_symbols+1 含 blank) 对应 Medusa 共享 output_proj。
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

    输入为 joiner 融合特征 activation(src_enc + tgt_enc)，与 joiner 输入一致。

    Args:
        input_dim: 特征维度 (=512)，ResBlock 在此维度做残差。
        num_vocab: 词表大小 (含 blank) = num_symbols+1。
        num_heads: K，未来预测头数 (head 1..K)。0 = 关闭 (单头消融)。
        activation: 融合后激活函数名，与 joiner 保持一致。
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_vocab: int = 75,
        num_heads: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.res_blocks = torch.nn.ModuleList(
            [MedusaResBlock(input_dim) for _ in range(num_heads)]
        )
        self.w_vocab = torch.nn.Linear(input_dim, num_vocab)
        # 跟 joiner 一致的激活
        self.activation = torch.nn.ReLU() if activation == "relu" else (
            torch.nn.Tanh() if activation == "tanh" else (
                torch.nn.GELU() if activation == "gelu" else torch.nn.SiLU()
            )
        )

    # ------------------------------------------------------------------
    def forward(self, src_enc: Tensor, tgt_enc: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """计算所有 head 的 logits (训练: 全网格; 推理: 单点)。

        Args:
            src_enc: (B, T, input_dim) encoder 输出 (推理时 T=1)。
            tgt_enc: (B, L, input_dim) predictor 输出 (L=U+1; 推理时 L=1)。

        Returns:
            (logits_base, logits_future):
                logits_base   : (B, T, L, num_vocab)  head 0 = W_vocab(joint)
                logits_future : List[K] 个 (B, T, L, num_vocab)
        """
        # 跟 joiner 一致的融合: src_enc.unsqueeze(2) + tgt_enc.unsqueeze(1) → (B,T,L,D)
        joint = src_enc.unsqueeze(2) + tgt_enc.unsqueeze(1)
        joint = self.activation(joint)
        logits_base = self.w_vocab(joint)
        logits_future: List[Tensor] = []
        for block in self.res_blocks:
            logits_future.append(self.w_vocab(block(joint)))
        return logits_base, logits_future


# ═══════════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════════


def shifted_ce_loss(
    logits_future: List[Tensor],
    targets: Tensor,
    weights: List[float],
    target_lengths: Tensor,
) -> Tensor:
    r"""Σ_{k=1..K} λ_k · CE(head_k 在 (t,u) 的 logits, targets[u+k])。

    joiner 在 (t,u) 预测 y_u (blank/emit 决策)，head k 预测下一个 token y_{u+k}。
    标签对每个帧 t 相同（只依赖 u），在 T 维上广播。padding 位置
    （u+k ≥ target_lengths[b]，标签落在短样本填充区）置 -1 被 ignore。

    Args:
        logits_future: List[K] 个 (B, T, L, V)，L≥U+1。
        targets: (B, U) 真实标签 id (0..V-2，不含 blank)。
        weights: [λ_1, ..., λ_K]。
        target_lengths: (B,) 各样本有效 token 数，用于 mask padding。
    """
    U = targets.size(1)
    _, T, _, V = logits_future[0].shape
    total = targets.new_zeros(())
    tl = target_lengths.to(targets.device)
    for k_idx, logits_k in enumerate(logits_future):  # head k = k_idx+1
        k = k_idx + 1
        if U - k <= 0:
            continue  # 序列太短，该 head 无有效位置
        # 取 u=0..U-k-1 共 U-k 个有效位置；T 维广播
        pred = logits_k[:, :, : U - k, :].reshape(-1, V)   # (B*T*(U-k), V)
        label = targets[:, k: U]                            # (B, U-k)
        label = label.unsqueeze(1).expand(-1, T, -1).reshape(-1).long()  # (B*T*(U-k),)
        # mask：仅当 u+k < target_lengths[b] 时 label 是有效 token，否则 -1 (ignore)
        u_idx = torch.arange(U - k, device=targets.device).unsqueeze(0)  # (1, U-k)
        valid = (u_idx + k) < tl.unsqueeze(1)                            # (B, U-k)
        valid = valid.unsqueeze(1).expand(-1, T, -1).reshape(-1)         # (B*T*(U-k),)
        label = label.masked_fill(~valid, -1)
        ce = F.cross_entropy(pred, label, reduction="mean", ignore_index=-1)
        total = total + weights[k_idx] * ce
    return total


def kl_to_base_loss(
    logits_future: List[Tensor],
    logits_base: Tensor,
    beta: float,
) -> Tensor:
    r"""β · Σ_{k=1..K} KL(softmax(head_k) ‖ softmax(head_0).detach())。

    同位置分布正则：防 draft 头漂离 base 分布先验。对所有 (t,u) 位置取均值。
    head_0 detach (不回传)。
    """
    if beta <= 0 or len(logits_future) == 0:
        return logits_base.new_zeros(())
    log_p0 = F.log_softmax(logits_base, dim=-1).detach()
    total = logits_base.new_zeros(())
    for logits_k in logits_future:
        log_pk = F.log_softmax(logits_k, dim=-1)
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
    activation = m.get("joiner", {}).get("activation", "relu")
    return MedusaHeads(
        input_dim=encoding_dim,
        num_vocab=num_vocab,
        num_heads=num_heads,
        activation=activation,
    )