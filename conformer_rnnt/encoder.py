"""Conformer encoder for RNN-T ASR.

Architecture:
    input → input_linear → time_reduction
    → ConformerBlock1 (N₁ layers)
    → [DynamicCompression]   ← optional, configurable
    → ConformerBlock2 (N₂ layers)
    → output_linear → layer_norm

The two-stage design lets us insert dynamic compression between blocks:
Block 1 captures local patterns; compression selects key frames at change
points; Block 2 models longer-range dependencies on the reduced sequence.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torchaudio.models import Conformer

from conformer_rnnt.compression import DynamicCompression


# ═══════════════════════════════════════════════════════════════════════════
# Time reduction
# ═══════════════════════════════════════════════════════════════════════════


class TimeReduction(torch.nn.Module):
    """Coalesce consecutive frames: ``(B, T, D) → (B, T//S, D*S)``.

    Args:
        stride: Number of frames to merge into one output frame.
    """

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, input: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, D = input.shape
        num_frames = T - (T % self.stride)
        input = input[:, :num_frames, :]
        lengths = lengths.div(self.stride, rounding_mode="trunc")
        T_out = num_frames // self.stride
        output = input.reshape(B, T_out, D * self.stride).contiguous()
        return output, lengths


# ═══════════════════════════════════════════════════════════════════════════
# Conformer encoder
# ═══════════════════════════════════════════════════════════════════════════


class ConformerEncoder(torch.nn.Module):
    """Two-stage Conformer encoder with optional inter-layer compression.

    All constructor arguments are keyword-only for clarity.

    Args:
        input_dim: Mel filterbank dimension (e.g. 80).
        output_dim: Output encoding dimension passed to the joint network.
        right_context_length: Right-context frames appended to each utterance.
        time_reduction_input_dim: Dimension after input projection.
        time_reduction_stride: Frame merging factor (4 → 200→50 fps).
        num_heads: Attention heads per Conformer layer.
        ffn_dim: Feed-forward hidden dimension.
        num_layers_block1: Layers before compression. Default: 2.
        num_layers_block2: Layers after compression. Default: 8.
        dropout: Dropout probability. Default: 0.1.
        depthwise_conv_kernel_size: Depthwise conv kernel. Default: 31.
        compression_filter: ``"none"`` | ``"fir"`` | ``"savgol"`` |
            ``"savgol_hanning"`` | ``"hanning"`` | ``"naive"``.
        hanning_window_size: Window size for Hanning smoothing. Default: 5.
        weight_predictor_mode: ``"single_frame"`` | ``"diff5"`` |
            ``"five_frames"``. Default: ``"diff5"``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        right_context_length: int,
        time_reduction_input_dim: int,
        time_reduction_stride: int,
        compression_right_context_length: Optional[int] = None,
        num_heads: int,
        ffn_dim: int,
        num_layers_block1: int = 2,
        num_layers_block2: int = 8,
        dropout: float = 0.1,
        depthwise_conv_kernel_size: int = 31,
        compression_filter: str = "hanning",
        hanning_window_size: int = 5,
        weight_predictor_mode: str = "diff5",
        compression_detector: Optional[dict] = None,
        compression_mixer: Optional[dict] = None,
        compression_keep_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.right_context_length = right_context_length
        self.time_reduction_stride = time_reduction_stride

        dim = time_reduction_input_dim * time_reduction_stride  # encoder dim
        rc_frames = right_context_length // time_reduction_stride    # reduced right-ctx
        compression_rc_frames = (
            right_context_length
            if compression_right_context_length is None
            else compression_right_context_length
        )

        # Frontend
        self.input_linear = torch.nn.Linear(input_dim, time_reduction_input_dim, bias=False)
        self.time_reduction = TimeReduction(time_reduction_stride)

        # Stage 1 — local patterns
        self.conformer1 = _make_conformer(
            dim, num_heads, ffn_dim, num_layers_block1,
            depthwise_conv_kernel_size, dropout,
        )

        # Inter-stage compression (new detector/mixer axes take precedence
        # over the legacy filter enum)
        if compression_detector is not None or compression_mixer is not None:
            det_cfg = compression_detector or {"type": "raw"}
            mix_cfg = compression_mixer or {"type": "hard"}
            self.compression = DynamicCompression(
                right_context_length=compression_rc_frames,
                detector=det_cfg,
                mixer=mix_cfg,
                feature_dim=dim,
                keep_rate=compression_keep_rate,
                hanning_window_size=hanning_window_size,
                weight_predictor_mode=weight_predictor_mode,
            )
        else:
            self.compression = DynamicCompression.from_legacy(
                right_context_length=compression_rc_frames,
                compression_filter=compression_filter,
                hanning_window_size=hanning_window_size,
                feature_dim=dim,
                weight_predictor_mode=weight_predictor_mode,
            )

        # Stage 2 — long-range dependencies
        self.conformer2 = _make_conformer(
            dim, num_heads, ffn_dim, num_layers_block2,
            depthwise_conv_kernel_size, dropout,
        )

        # Output
        self.output_linear = torch.nn.Linear(dim, output_dim)
        self.output_norm = torch.nn.LayerNorm(output_dim)

    # ------------------------------------------------------------------
    def forward(self, input: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Training / non-streaming forward pass.

        Args:
            input: ``(B, T + right_context_length, D)``.
            lengths: Valid frame counts (without right context), ``(B,)``.

        Returns:
            (encoded, out_lengths): ``(B, T//stride, output_dim)``, ``(B,)``.
        """
        rc_frames = self.right_context_length // self.time_reduction_stride

        # Frontend
        x, x_len = self.time_reduction(self.input_linear(input), lengths)
        x, x_len = self.conformer1(x, x_len + rc_frames)

        # Compression
        if self.compression.passthrough:
            x, x_len = x, x_len
        else:
            x, x_len = self.compression(x, x_len)

        # Stage 2 + output
        x, x_len = self.conformer2(x, x_len)
        return self.output_norm(self.output_linear(x)), x_len

    # ------------------------------------------------------------------
    @torch.jit.export
    def infer(
        self,
        input: Tensor,
        lengths: Tensor,
        states: Optional[List[List[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, List[List[Tensor]]]:
        """Streaming-compatible forward (delegates to forward, returns empty state)."""
        output, out_len = self.forward(input, lengths)
        return output, out_len, []


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_conformer(
    dim: int, heads: int, ffn_dim: int, num_layers: int,
    kernel_size: int, dropout: float,
) -> Conformer:
    """Create a Conformer block with consistent arguments."""
    return Conformer(
        dim, heads, ffn_dim,
        num_layers=num_layers,
        depthwise_conv_kernel_size=kernel_size,
        dropout=dropout,
    )
