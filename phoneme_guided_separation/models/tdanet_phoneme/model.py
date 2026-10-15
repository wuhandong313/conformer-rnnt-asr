"""TDANetPhoneme: TDANet + post-sep U-Net Transformer + phoneme / ArtV probing.

实验 5：波形 ``mask = ReLU( mask_net(sm) + alpha * Δ(Channel+TF) )``。
``alpha=0`` 时与 TDANet 同一条 ``mask_net``。``return_phoneme=True`` 时返回
``(waveforms, phoneme_logits, artv_pred)``，探针挂在时间下采样后的瓶颈上。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..tdanet import TDANet
from .probing import ProbingNet
from .unet_transformer import PostSepUNetTransformer


class TDANetPhoneme(TDANet):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        num_sources=2,
        sample_rate=16000,
        # phoneme / transformer extras
        num_phonemes=80,
        tf_num_layers=2,
        tf_nhead=4,
        tf_dim_feedforward=512,
        tf_dropout=0.1,
        downsample_ratio=8,
        artv_dim=24,
        alpha_init=0.0,
    ):
        super().__init__(
            out_channels=out_channels,
            in_channels=in_channels,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            enc_kernel_size=enc_kernel_size,
            num_sources=num_sources,
            sample_rate=sample_rate,
        )
        self.num_phonemes = num_phonemes
        self.artv_dim = int(artv_dim)
        self.downsample_ratio = int(downsample_ratio)
        self.tf_num_layers = tf_num_layers
        self.tf_nhead = tf_nhead
        self.tf_dim_feedforward = tf_dim_feedforward
        self.tf_dropout = tf_dropout
        self.alpha_init = float(alpha_init)

        # 保留父类 mask_net：α=0 时波形与实验 3 TDANet 同一条
        # mask_net(sm) + α · Δ(Channel+TF)。Δ 末层零初始化，α 拉到 0.5 的瞬间波形不跳。
        self.channel_gen = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, num_sources * out_channels, 1),
        )
        self.post_sep_tf = PostSepUNetTransformer(
            d_model=out_channels,
            nhead=tf_nhead,
            num_layers=tf_num_layers,
            dim_feedforward=tf_dim_feedforward,
            dropout=tf_dropout,
            downsample_ratio=self.downsample_ratio,
        )
        self.probing = ProbingNet(
            d_model=out_channels,
            num_phonemes=num_phonemes,
            artv_dim=self.artv_dim,
        )
        self.mask_from_hidden = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, self.enc_num_basis, 1),
        )
        nn.init.zeros_(self.mask_from_hidden[-1].weight)
        if self.mask_from_hidden[-1].bias is not None:
            nn.init.zeros_(self.mask_from_hidden[-1].bias)
        self.register_buffer(
            "alpha", torch.tensor(float(alpha_init), dtype=torch.float32)
        )

    def set_alpha(self, value: float) -> None:
        self.alpha.fill_(float(value))

    def forward(self, input_wav, return_phoneme: bool = False):
        """混合波形 → 分离通道 → U-Net TF 残差 →（探针 / mask 解码）。

        Args:
            input_wav: `[T]` / `[B, T]` / `[B, 1, T]`
            return_phoneme: 是否把瓶颈上的探针一起返回。
                True（训练 CE/ArtV、dump CSV）：返回
                ``(waveforms, phoneme_logits, artv_pred)``。
                False（默认；val PIT / 测试 SI-SNR）：只返回波形，不算探针头。
                ``alpha=0`` 且不算探针时不跑 TF，波形与 TDANet 的 ``mask_net`` 相同。
        """
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 3:
            input_wav = input_wav.squeeze(1)

        x, rest = self.pad_input(
            input_wav, self.enc_kernel_size, self.enc_kernel_size // 4
        )
        x = self.encoder(x.unsqueeze(1))
        s = x.clone()

        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        b, c, t = x.shape
        n_src = self.num_sources
        # 与 TDANet.forward 相同：PReLU+Conv1d(C→n_src·F)
        mask_logits = self.mask_net(x)

        alpha = self.alpha.to(dtype=x.dtype)
        need_tf = bool(return_phoneme) or float(alpha.detach()) != 0.0
        phoneme_logits = None
        artv_pred = None
        if need_tf:
            channels = self.channel_gen(x).view(b, n_src, c, t)
            ch = channels.reshape(b * n_src, c, t)
            tf_out, bn = self.post_sep_tf(ch)
            h = ch + alpha * tf_out
            if return_phoneme:
                logits, artv = self.probing(bn, return_artv=True)
                t_bn = logits.size(1)
                phoneme_logits = logits.view(b, n_src, t_bn, self.num_phonemes)
                artv_pred = artv.view(b, n_src, t_bn, self.artv_dim)
            delta = self.mask_from_hidden(h).reshape(b, n_src * self.enc_num_basis, t)
            mask_logits = mask_logits + alpha * delta

        masks = self.mask_nl_class(mask_logits)
        masks = masks.view(b, n_src, self.enc_num_basis, t)
        masked = masks * s.unsqueeze(1)

        estimated_waveforms = self.decoder(masked.view(b, -1, t))
        estimated_waveforms = estimated_waveforms[
            :,
            :,
            self.enc_kernel_size
            - self.enc_kernel_size
            // 4 : -(rest + self.enc_kernel_size - self.enc_kernel_size // 4),
        ].contiguous()

        if was_one_d:
            estimated_waveforms = estimated_waveforms.squeeze(0)
            if return_phoneme:
                phoneme_logits = phoneme_logits.squeeze(0)
                artv_pred = artv_pred.squeeze(0)

        if return_phoneme:
            return estimated_waveforms, phoneme_logits, artv_pred
        return estimated_waveforms

    def get_model_args(self):
        return {
            "n_src": self.num_sources,
            "num_phonemes": self.num_phonemes,
            "downsample_ratio": self.downsample_ratio,
            "artv_dim": self.artv_dim,
            "alpha_init": float(self.alpha.item())
            if torch.is_tensor(self.alpha)
            else float(self.alpha_init),
            "tf_num_layers": self.tf_num_layers,
            "tf_nhead": self.tf_nhead,
            "tf_dim_feedforward": self.tf_dim_feedforward,
            "tf_dropout": self.tf_dropout,
        }
