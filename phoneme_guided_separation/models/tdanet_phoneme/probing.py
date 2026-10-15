"""Linear phoneme probing head (Softmax left to CE loss) + ArtV 回归头。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProbingNet(nn.Module):
    """Frame-level phoneme logits and optional ArtV from hidden rep.

    Input:  [B, C, T] or [B*n_src, C, T]
    Logits: [B, T, K]
    ArtV:   [B, T, D]
    """

    def __init__(
        self,
        d_model: int = 128,
        num_phonemes: int = 80,
        artv_dim: int = 24,
    ):
        super().__init__()
        self.proj = nn.Linear(d_model, num_phonemes)
        self.artv_proj = nn.Linear(d_model, artv_dim)
        self.artv_dim = artv_dim

    def forward(
        self, h: torch.Tensor, return_artv: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # h: [B, C, T] -> [B, T, C]
        ht = h.transpose(1, 2)
        logits = self.proj(ht)
        if return_artv:
            return logits, self.artv_proj(ht)
        return logits
