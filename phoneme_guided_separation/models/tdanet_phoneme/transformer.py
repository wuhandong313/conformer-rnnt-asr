"""Post-separation Transformer (shared across speakers).

Input/output: [B, C, T] with C == d_model (default 128).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        t = x.size(1)
        return x + self.pe[:, :t]


class PostSepTransformer(nn.Module):
    """Shared Transformer after per-source 128-d channels."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 10000,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] or [B*n_src, C, T]
        Returns:
            same shape as x
        """
        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x.transpose(1, 2)
