"""RNN-T model assembly.

Combines ConformerEncoder + Predictor + Joiner into a complete transducer.
Provides both manual and config-driven factory functions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from conformer_rnnt.encoder import ConformerEncoder
from conformer_rnnt.staged_encoder import StagedConformerEncoder
from conformer_rnnt.compression.stages import placement_options_from_config
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
    """

    def __init__(self, transcriber: Union[ConformerEncoder, StagedConformerEncoder], predictor: Predictor, joiner: Joiner) -> None:
        super().__init__()
        self.transcriber = transcriber
        self.predictor = predictor
        self.joiner = joiner

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
    compression_right_context_length: Optional[int] = None,
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
    compression_detector: Optional[dict] = None,
    compression_mixer: Optional[dict] = None,
    compression_keep_rate: Optional[float] = None,
    sequence_mode: str = "legacy",
    encoder_num_layers: Optional[int] = None,
    compression_stages: Optional[List[dict]] = None,
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
) -> ConformerRNNT:
    """Build a Conformer-RNNT model from explicit keyword arguments.

    All parameters are keyword-only to prevent positional confusion.
    """
    if compression_stages is not None:
        if (sequence_mode != "offline" or right_context_length != 0
                or compression_right_context_length is not None or compression_keep_rate is not None):
            raise ValueError("Staged compression is offline, without context partitions or a target rate")
        if compression_detector is not None or compression_mixer is not None:
            raise ValueError("Explicit stages cannot be combined with a single-stage detector or mixer")
        transcriber = StagedConformerEncoder(
            input_dim=input_dim, output_dim=encoding_dim,
            time_reduction_input_dim=time_reduction_input_dim,
            time_reduction_stride=time_reduction_stride,
            num_heads=num_heads, ffn_dim=ffn_dim, num_layers=encoder_num_layers,
            compression_stages=compression_stages, dropout=dropout,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        )
    else:
        if encoder_num_layers is not None:
            raise ValueError("encoder_num_layers requires explicit compression_stages")
        transcriber = ConformerEncoder(
            input_dim=input_dim,
            output_dim=encoding_dim,
            right_context_length=right_context_length,
            time_reduction_input_dim=time_reduction_input_dim,
            time_reduction_stride=time_reduction_stride,
            compression_right_context_length=compression_right_context_length,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers_block1=num_layers_block1,
            num_layers_block2=num_layers_block2,
            dropout=dropout,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            compression_filter=compression_filter,
            hanning_window_size=hanning_window_size,
            weight_predictor_mode=weight_predictor_mode,
            compression_detector=compression_detector,
            compression_mixer=compression_mixer,
            compression_keep_rate=compression_keep_rate,
            sequence_mode=sequence_mode,
        )
    return ConformerRNNT(
        transcriber=transcriber,
        predictor=Predictor(
            num_symbols=num_symbols,
            output_dim=encoding_dim,
            symbol_embedding_dim=symbol_embedding_dim,
            num_lstm_layers=num_lstm_layers,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layer_norm=lstm_layer_norm,
            lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
            lstm_dropout=lstm_dropout,
        ),
        joiner=Joiner(input_dim=encoding_dim, output_dim=num_symbols, activation=joiner_activation),
    )


def sequence_options_from_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Reject obsolete controls in explicit offline recipes instead of ignoring them."""
    compression = model_config.get("compression", {})
    mode = model_config.get("sequence_mode", "legacy")
    if mode == "offline":
        if ("right_context_length" in model_config or "right_context_length" in compression
                or "keep_rate" in compression):
            raise ValueError("Offline configuration must omit right_context_length and keep_rate")
        return dict(sequence_mode=mode, right_context_length=0, compression_right_context_length=None)
    return dict(
        sequence_mode=mode, right_context_length=model_config.get("right_context_length", 4),
        compression_right_context_length=compression.get("right_context_length"),
    )


def conformer_rnnt_model_from_config(config: Dict[str, Any]) -> ConformerRNNT:
    """Build a Conformer-RNNT model from a config dictionary.

    Expects the ``model`` section of ``configs/default.yaml``.
    """
    m = config.get("model", config)
    c = m.get("compression", {})
    p = m.get("predictor", {})
    j = m.get("joiner", {})
    num_symbols = m.get("num_symbols")
    if num_symbols is None:
        tokenizer = config.get("tokenizer", {})
        tokenizer_type = tokenizer.get("type", "phone69")
        if tokenizer_type == "phone69":
            from phone69_tokenizer import DEFAULT_PHONE69_VOCAB, FixedPhone69Tokenizer
            num_symbols = FixedPhone69Tokenizer(tokenizer.get("vocab_path") or DEFAULT_PHONE69_VOCAB).get_piece_size()
        elif tokenizer_type == "ipa":
            from ipa_tokenizer import DEFAULT_VOCAB, FixedIPATokenizer
            num_symbols = FixedIPATokenizer(tokenizer.get("vocab_path") or DEFAULT_VOCAB).get_piece_size()
        else:
            raise ValueError("Word-BPE model factory requires model.num_symbols")

    return conformer_rnnt_model(
        input_dim=m.get("input_dim", 80),
        encoding_dim=m.get("encoding_dim", 512),
        num_symbols=num_symbols + 1,  # +blank
        **sequence_options_from_config(m),
        **placement_options_from_config(m),
        time_reduction_input_dim=m.get("time_reduction_input_dim", 128),
        time_reduction_stride=m.get("time_reduction_stride", 4),
        num_heads=m.get("num_heads", 8),
        ffn_dim=m.get("ffn_dim", 1024),
        num_layers_block1=m.get("num_layers_block1", 2),
        num_layers_block2=m.get("num_layers_block2", 8),
        dropout=m.get("dropout", 0.1),
        depthwise_conv_kernel_size=m.get("depthwise_conv_kernel_size", 31),
        compression_filter=c.get("filter", "none" if "stages" in c else "hanning"),
        hanning_window_size=c.get("hanning_window_size", 5),
        weight_predictor_mode=c.get("weight_predictor_mode", "five_frames" if "stages" in c else "diff5"),
        compression_detector=c.get("detector"),
        compression_mixer=c.get("mixer"),
        compression_keep_rate=c.get("keep_rate"),
        symbol_embedding_dim=p.get("symbol_embedding_dim", 128),
        num_lstm_layers=p.get("num_lstm_layers", 2),
        lstm_hidden_dim=p.get("lstm_hidden_dim", 128),
        lstm_layer_norm=p.get("lstm_layer_norm", True),
        lstm_layer_norm_epsilon=p.get("lstm_layer_norm_epsilon", 1e-3),
        lstm_dropout=p.get("lstm_dropout", 0.3),
        joiner_activation=j.get("activation", "relu"),
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
