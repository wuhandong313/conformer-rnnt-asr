
# Allow direct execution from the repository root.
import sys as _sys
from pathlib import Path as _Path
_repo_root = str(_Path(__file__).resolve().parents[1])
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

# tests_tiny_model.py — 小随机模型工厂（CPU 可跑）
import torch
from conformer_rnnt.model import conformer_rnnt_model


def make_tiny_model(*, compression_filter: str = "none", medusa_heads: int = 0):
    torch.manual_seed(0)
    return conformer_rnnt_model(
        input_dim=80, encoding_dim=64, right_context_length=4,
        time_reduction_input_dim=32, time_reduction_stride=4,
        num_heads=4, ffn_dim=128, num_layers_block1=1, num_layers_block2=1,
        depthwise_conv_kernel_size=15, dropout=0.0,
        compression_filter=compression_filter, hanning_window_size=3,
        weight_predictor_mode="diff5", num_symbols=20,
        symbol_embedding_dim=32, num_lstm_layers=1, lstm_hidden_dim=32,
        lstm_dropout=0.0, joiner_activation="relu",
        medusa_num_heads=medusa_heads, medusa_input_mode="stream_local",
        medusa_lookahead_frames=1,
    ).eval()