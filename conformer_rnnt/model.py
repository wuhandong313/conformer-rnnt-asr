"""RNN-T model assembly.

Combines ConformerEncoder + Predictor + Joiner into a complete transducer.
Provides the canonical ``conformer_rnnt_model`` factory.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from conformer_rnnt.encoder import ConformerEncoder
from conformer_rnnt.predictor import Predictor
from conformer_rnnt.joiner import Joiner


class ConformerRNNT(torch.nn.Module):
    """Conformer-based Recurrent Neural Network Transducer for ASR.

    Components
    ----------
    transcriber : ConformerEncoder
        Acoustic encoder — two-stage Conformer with optional compression.
    predictor : Predictor
        Autoregressive language model — stacked layer-norm LSTM.
    joiner : Joiner
        Joint network — additive fusion → activation → linear → logits.
    medusa_heads : MedusaHeads | None
        Optional Medusa-style future-label heads over the predictor trunk;
        ``None`` for single-head / ablation.
    """

    def __init__(
        self,
        transcriber: ConformerEncoder,
        predictor,
        joiner: Joiner,
        medusa_heads=None,
    ) -> None:
        super().__init__()
        self.transcriber = transcriber
        self.predictor = predictor
        self.joiner = joiner
        self.medusa_heads = medusa_heads

    # ── Training forward ────────────────────────────────────────────────
    def forward_train_aux(
        self,
        sources: Tensor,
        source_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[List[Tensor]]]:
        """Training forward producing RNNT logits + Medusa head logits.

        Joint fusion is computed once and shared by the joiner and the Medusa
        heads (head 0's logits are the joiner's own logits — same projection).

        Returns:
            (logits, src_len, tgt_len, logits_base, logits_future)
            logits_base/future are None when medusa_heads is None.
        """
        src_enc, src_len = self.transcriber(sources, source_lengths)
        tgt_enc, tgt_len, _ = self.predictor(targets, target_lengths, None)
        joint = self.joiner.fuse(src_enc, tgt_enc)
        logits = self.joiner.project(joint)
        if self.medusa_heads is not None:
            logits_base, logits_future = self.medusa_heads(
                joint,
                source_encodings=src_enc,
                target_encodings=tgt_enc,
            )
        else:
            logits_base, logits_future = None, None
        return logits, src_len, tgt_len, logits_base, logits_future

    # ── Component-level APIs (for beam search) ──────────────────────────
    @torch.jit.export
    def transcribe(self, sources: Tensor, source_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode acoustics (non-streaming)."""
        return self.transcriber(sources, source_lengths)

    @torch.jit.export
    def predict(
        self, targets: Tensor, target_lengths: Tensor, state: Optional[List[List[Tensor]]]
    ) -> Tuple[Tensor, Tensor, List[List[Tensor]]]:
        """Encode target tokens."""
        return self.predictor(targets, target_lengths, state)

    @torch.jit.export
    def join(
        self,
        source_encodings: Tensor, source_lengths: Tensor,
        target_encodings: Tensor, target_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Fuse acoustic and language encodings."""
        return self.joiner(source_encodings, source_lengths, target_encodings, target_lengths)

    def predict_heads(
        self,
        source_encodings: Tensor,
        target_encodings: Tensor,
        chain_drafts: Optional[List[int]] = None,
    ) -> Tuple[Optional[Tensor], List[Tensor]]:
        """Apply Medusa heads to joiner-fused features (for spec decode).

        Args:
            source_encodings: (B, T, D) encoder output. In ``stream_local``
                mode this is the current frame followed by the bounded
                lookahead window; only the first frame is scored.
            target_encodings: (B, L, D) from ``predict`` (L=1 at inference).
            chain_drafts: chain 模式推理时使用 — 已确定的前序 draft id;
                head-k 消费 chain_drafts[k-1] 的 embedding。

        Returns:
            (logits_base, logits_future) — both empty/None when no heads.
        """
        if self.medusa_heads is None:
            return None, []
        if source_encodings.size(1) == 0:
            raise ValueError("predict_heads requires at least one source frame")
        # The verifier is always anchored at the current frame. The complete
        # source window is passed separately so stream-local heads can inspect
        # only the configured right-context lookahead.
        current = source_encodings[:, :1]
        joint = self.joiner.fuse(current, target_encodings)
        return self.medusa_heads(
            joint,
            source_encodings=source_encodings,
            target_encodings=target_encodings,
            chain_drafts=chain_drafts,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════════════════


def conformer_rnnt_model(
    *,
    # Encoder
    input_dim: int,
    encoding_dim: int,
    right_context_length: int,
    time_reduction_input_dim: int,
    time_reduction_stride: int,
    num_heads: int,
    ffn_dim: int,
    num_layers_block1: int = 2,
    num_layers_block2: int = 8,
    dropout: float = 0.1,
    depthwise_conv_kernel_size: int = 31,
    # Compression
    compression_filter: str = "hanning",
    hanning_window_size: int = 5,
    weight_predictor_mode: str = "diff5",
    # Predictor
    num_symbols: int,
    symbol_embedding_dim: int = 128,
    num_lstm_layers: int = 2,
    lstm_hidden_dim: int = 128,
    lstm_layer_norm: bool = True,
    lstm_layer_norm_epsilon: float = 1e-3,
    lstm_dropout: float = 0.3,
    # Joiner
    joiner_activation: str = "relu",
    # Medusa
    medusa_num_heads: int = 0,
    medusa_input_mode: str = "stream_local",
    medusa_lookahead_frames: Optional[int] = None,
    medusa_draft_after_emit: bool = False,
    medusa_chain_states: bool = False,
) -> ConformerRNNT:
    """Build a Conformer-RNNT model from explicit keyword arguments.

    All parameters are keyword-only to prevent positional confusion.
    """
    predictor = Predictor(
        num_symbols=num_symbols,
        output_dim=encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )

    joiner = Joiner(
        input_dim=encoding_dim, output_dim=num_symbols, activation=joiner_activation
    )

    medusa_heads = None
    if medusa_num_heads > 0:
        from conformer_rnnt.medusa import MedusaHeads
        max_lookahead = max(0, right_context_length // time_reduction_stride)
        if medusa_lookahead_frames is None:
            medusa_lookahead_frames = max_lookahead
        # Offline (non-streaming) decoding encodes the full utterance up
        # front, so a wider draft-side acoustic window is legitimate — the
        # clamp to right-context only applies to streaming constraints.
        medusa_lookahead_frames = max(0, int(medusa_lookahead_frames))
        # 共享 joiner 的输出投影: head 0 ≡ joiner 精确一致,draft 与验证器
        # 用同一个打分标准 (对应原版 Medusa 的共享 output_proj)。
        medusa_heads = MedusaHeads(
            input_dim=encoding_dim,
            num_vocab=num_symbols,
            num_heads=medusa_num_heads,
            w_vocab=joiner.linear,
            input_mode=medusa_input_mode,
            lookahead_frames=medusa_lookahead_frames,
            draft_after_emit=medusa_draft_after_emit,
            chain_states=medusa_chain_states,
        )

    return ConformerRNNT(
        transcriber=ConformerEncoder(
            input_dim=input_dim,
            output_dim=encoding_dim,
            right_context_length=right_context_length,
            time_reduction_input_dim=time_reduction_input_dim,
            time_reduction_stride=time_reduction_stride,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers_block1=num_layers_block1,
            num_layers_block2=num_layers_block2,
            dropout=dropout,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            compression_filter=compression_filter,
            hanning_window_size=hanning_window_size,
            weight_predictor_mode=weight_predictor_mode,
        ),
        predictor=predictor,
        joiner=joiner,
        medusa_heads=medusa_heads,
    )
