"""残差下采样 + Transformer + 瓶颈 + Transformer + 上采样（实验 5）。

输入/输出通道均为 ``[B, C, T]``。探针挂在瓶颈输出上（时间维约 ``T / ratio``）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import PostSepTransformer


def _pad_to_multiple(x: torch.Tensor, ratio: int) -> tuple[torch.Tensor, int]:
    """右侧 pad 使时间维可被 ratio 整除。返回 ``(padded, pad)``。"""
    t = x.size(-1)
    pad = (ratio - t % ratio) % ratio
    if pad:
        x = F.pad(x, (0, pad))
    return x, pad


class ResidualDownsample(nn.Module):
    """stride=ratio 的卷积 + 平均池化残差。"""

    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.ratio = int(ratio)
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=self.ratio, stride=self.ratio
        )
        self.pool = nn.AvgPool1d(self.ratio, stride=self.ratio)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        orig_t = x.size(-1)
        x, pad = _pad_to_multiple(x, self.ratio)
        y = self.conv(x) + self.pool(x)
        return y, pad, orig_t


class ResidualUpsample(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.ratio = int(ratio)
        self.up = nn.ConvTranspose1d(
            channels, channels, kernel_size=self.ratio, stride=self.ratio
        )

    def forward(self, x: torch.Tensor, target_t: int) -> torch.Tensor:
        y = self.up(x)
        t = y.size(-1)
        if t < target_t:
            y = F.pad(y, (0, target_t - t))
        elif t > target_t:
            y = y[..., :target_t]
        return y


class ChannelBottleneck(nn.Module):
    """1×1 压缩再还原，残差；输出与输入同形状，供探针使用。"""

    def __init__(self, channels: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(channels // 4, 8)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.PReLU(),
            nn.Conv1d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PostSepUNetTransformer(nn.Module):
    """下采样 TF → 瓶颈 → TF 上采样。``forward`` 返回全分辨率残差分支与瓶颈特征。"""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        downsample_ratio: int = 8,
        max_len: int = 10000,
    ):
        super().__init__()
        self.d_model = d_model
        self.downsample_ratio = int(downsample_ratio)
        self.down = ResidualDownsample(d_model, self.downsample_ratio)
        self.tf_down = PostSepTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        self.bottleneck = ChannelBottleneck(d_model)
        self.tf_up = PostSepTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        self.up = ResidualUpsample(d_model, self.downsample_ratio)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, C, T]``
        Returns:
            tf_out: ``[B, C, T]`` 与输入对齐，供 ``alpha * tf_out`` 残差
            bottleneck: ``[B, C, T_bn]`` 探针入口
        """
        z, _pad, orig_t = self.down(x)
        z = self.tf_down(z)
        bn = self.bottleneck(z)
        z = self.tf_up(bn)
        tf_out = self.up(z, orig_t)
        return tf_out, bn
