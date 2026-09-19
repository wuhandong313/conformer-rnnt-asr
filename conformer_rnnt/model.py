"""RNN-T model assembly.

Combines ConformerEncoder + Predictor + Joiner into a complete transducer.
Provides both manual and config-driven factory functions.
"""

from typing import Any, Dict, List, Optional, Tuple

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
    predictor : Predictor | PredictorAttn
        Autoregressive language model — LSTM (default) or restricted causal
        self-attention trunk (``predictor.type="attn"``).
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
    def forward(
        self,
        sources: Tensor,
        source_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
        predictor_state: Optional[List[List[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[List[Tensor]]]:
        """Training forward pass.

        Returns:
            (logits, src_len, tgt_len, pred_state)
                logits: ``(B, T_enc, U_enc, num_symbols)``
        """
        src_enc, src_len = self.transcriber(sources, source_lengths)
        tgt_enc, tgt_len, pred_state = self.predictor(targets, target_lengths, predictor_state)
        logits, src_len, tgt_len = self.joiner(src_enc, src_len, tgt_enc, tgt_len)
        return logits, src_len, tgt_len, pred_state

    # ── Training forward with Medusa aux outputs ────────────────────────
    def forward_train_aux(
        self,
        sources: Tensor,
        source_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, List[List[Tensor]], Optional[Tensor], Optional[List[Tensor]]]:
        """Forward producing RNNT logits + Medusa head logits for aux loss.

        Returns:
            (logits, src_len, tgt_len, pred_state, logits_base, logits_future)
            logits_base/future are None when medusa_heads is None.
        """
        src_enc, src_len = self.transcriber(sources, source_lengths)
        tgt_enc, tgt_len, pred_state = self.predictor(targets, target_lengths, None)
        logits, src_len, tgt_len = self.joiner(src_enc, src_len, tgt_enc, tgt_len)
        if self.medusa_heads is not None:
            logits_base, logits_future = self.medusa_heads(tgt_enc)
        else:
            logits_base, logits_future = None, None
        return logits, src_len, tgt_len, pred_state, logits_base, logits_future

    # ── Component-level APIs (for beam search) ──────────────────────────
    @torch.jit.export
    def transcribe(self, sources: Tensor, source_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode acoustics (non-streaming)."""
        return self.transcriber(sources, source_lengths)

    @torch.jit.export
    def transcribe_streaming(
        self, sources: Tensor, source_lengths: Tensor, state: Optional[List[List[Tensor]]]
    ) -> Tuple[Tensor, Tensor, List[List[Tensor]]]:
        """Encode acoustics (streaming-compatible)."""
        return self.transcriber.infer(sources, source_lengths, state)

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

    @torch.jit.export
    def predict_heads(
        self, target_encodings: Tensor
    ) -> Tuple[Optional[Tensor], List[Tensor]]:
        """Apply Medusa heads to predictor trunk encodings (for spec decode).

        Args:
            target_encodings: (B, L, D) from ``predict`` (trunk output).

        Returns:
            (logits_base, logits_future) — both empty/None when no heads.
        """
        if self.medusa_heads is not None:
            return self.medusa_heads(target_encodings)
        empty: List[Tensor] = []
        return None, empty


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
    # Predictor — attention trunk (predictor_type="attn")
    predictor_type: str = "lstm",
    attn_d_model: int = 256,
    attn_num_layers: int = 4,
    attn_num_heads: int = 4,
    attn_ffn_dim: int = 1024,
    attn_left_context: int = 1024,
    attn_dropout: float = 0.1,
    attn_max_len: int = 4096,
    # Joiner
    joiner_activation: str = "relu",
    # Medusa
    medusa_num_heads: int = 0,
) -> ConformerRNNT:
    """Build a Conformer-RNNT model from explicit keyword arguments.

    All parameters are keyword-only to prevent positional confusion.
    """
    if predictor_type == "attn":
        from conformer_rnnt.predictor_attn import PredictorAttn
        predictor = PredictorAttn(
            num_symbols=num_symbols,
            output_dim=encoding_dim,
            symbol_embedding_dim=symbol_embedding_dim,
            d_model=attn_d_model,
            num_layers=attn_num_layers,
            num_heads=attn_num_heads,
            ffn_dim=attn_ffn_dim,
            left_context=attn_left_context,
            dropout=attn_dropout,
            max_len=attn_max_len,
        )
    else:
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

    medusa_heads = None
    if medusa_num_heads > 0:
        from conformer_rnnt.medusa import MedusaHeads
        medusa_heads = MedusaHeads(
            input_dim=encoding_dim, num_vocab=num_symbols, num_heads=medusa_num_heads
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
        joiner=Joiner(input_dim=encoding_dim, output_dim=num_symbols, activation=joiner_activation),
        medusa_heads=medusa_heads,
    )


def conformer_rnnt_model_from_config(config: Dict[str, Any]) -> ConformerRNNT:
    """Build a Conformer-RNNT model from a config dictionary.

    Expects the ``model`` section of ``configs/default.yaml``.
    """
    m = config.get("model", config)
    c = m.get("compression", {})
    p = m.get("predictor", {})
    a = p.get("attn", {})
    j = m.get("joiner", {})
    med = m.get("medusa", {})

    return conformer_rnnt_model(
        input_dim=m.get("input_dim", 80),
        encoding_dim=m.get("encoding_dim", 512),
        num_symbols=m.get("num_symbols", 74) + 1,  # +blank
        right_context_length=m.get("right_context_length", 4),
        time_reduction_input_dim=m.get("time_reduction_input_dim", 128),
        time_reduction_stride=m.get("time_reduction_stride", 4),
        num_heads=m.get("num_heads", 8),
        ffn_dim=m.get("ffn_dim", 1024),
        num_layers_block1=m.get("num_layers_block1", 2),
        num_layers_block2=m.get("num_layers_block2", 8),
        dropout=m.get("dropout", 0.1),
        depthwise_conv_kernel_size=m.get("depthwise_conv_kernel_size", 31),
        compression_filter=c.get("filter", "hanning"),
        hanning_window_size=c.get("hanning_window_size", 5),
        weight_predictor_mode=c.get("weight_predictor_mode", "diff5"),
        symbol_embedding_dim=p.get("symbol_embedding_dim", 128),
        num_lstm_layers=p.get("num_lstm_layers", 2),
        lstm_hidden_dim=p.get("lstm_hidden_dim", 128),
        lstm_layer_norm=p.get("lstm_layer_norm", True),
        lstm_layer_norm_epsilon=p.get("lstm_layer_norm_epsilon", 1e-3),
        lstm_dropout=p.get("lstm_dropout", 0.3),
        predictor_type=p.get("type", "lstm"),
        attn_d_model=a.get("d_model", 256),
        attn_num_layers=a.get("num_layers", 4),
        attn_num_heads=a.get("num_heads", 4),
        attn_ffn_dim=a.get("ffn_dim", 1024),
        attn_left_context=a.get("left_context", 1024),
        attn_dropout=a.get("dropout", 0.1),
        attn_max_len=a.get("max_len", 4096),
        joiner_activation=j.get("activation", "relu"),
        medusa_num_heads=med.get("num_heads", 0),
    )


def conformer_rnnt_base(num_symbols: int) -> ConformerRNNT:
    """Quick factory with reasonable defaults for prototyping."""
    return conformer_rnnt_model(
        input_dim=80, encoding_dim=512, num_symbols=num_symbols,
        right_context_length=4, time_reduction_input_dim=128, time_reduction_stride=4,
        num_heads=8, ffn_dim=1024,
        num_layers_block1=2, num_layers_block2=8,
        dropout=0.1, depthwise_conv_kernel_size=31,
        compression_filter="hanning", hanning_window_size=5, weight_predictor_mode="diff5",
        symbol_embedding_dim=128, num_lstm_layers=2,
        lstm_layer_norm=True, lstm_layer_norm_epsilon=1e-3, lstm_dropout=0.3,
    )
