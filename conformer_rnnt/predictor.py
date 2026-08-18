"""RNN-T prediction network (language model component).

A stack of layer-normalized LSTM layers with an embedding lookup table,
producing target-side encodings for the joint network.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor


class CustomLSTM(torch.nn.Module):
    """LSTM cell with optional layer normalization on gates and cell state.

    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Hidden state dimension.
        layer_norm (bool): If ``True``, apply LayerNorm to gates and cell.
        layer_norm_epsilon (float): Epsilon for LayerNorm.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_norm: bool = False,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.x2g = torch.nn.Linear(input_dim, 4 * hidden_dim, bias=(not layer_norm))
        self.p2g = torch.nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        if layer_norm:
            self.c_norm = torch.nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon)
            self.g_norm = torch.nn.LayerNorm(4 * hidden_dim, eps=layer_norm_epsilon)
        else:
            self.c_norm = torch.nn.Identity()
            self.g_norm = torch.nn.Identity()
        self.hidden_dim = hidden_dim

    def forward(
        self, input: Tensor, state: Optional[List[Tensor]]
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Forward pass.

        Args:
            input (Tensor): shape ``(T, B, D)``.
            state: List of ``[h, c]`` tensors, each shape ``(B, hidden_dim)``,
                or ``None`` to use zero-initialized state.

        Returns:
            (Tensor, List[Tensor]):
                - Output, shape ``(T, B, hidden_dim)``.
                - New state ``[h, c]``.
        """
        if state is None:
            B = input.size(1)
            h = torch.zeros(B, self.hidden_dim, device=input.device, dtype=input.dtype)
            c = torch.zeros(B, self.hidden_dim, device=input.device, dtype=input.dtype)
        else:
            h, c = state

        gated_input = self.x2g(input)
        outputs = []
        for gates in gated_input.unbind(0):
            gates = gates + self.p2g(h)
            gates = self.g_norm(gates)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()
            cell_gate = cell_gate.tanh()
            output_gate = output_gate.sigmoid()
            c = forget_gate * c + input_gate * cell_gate
            c = self.c_norm(c)
            h = output_gate * c.tanh()
            outputs.append(h)

        output = torch.stack(outputs, dim=0)
        state = [h, c]
        return output, state


class Predictor(torch.nn.Module):
    r"""RNN-T prediction network.

    Maps target token sequences to continuous encodings via embedding lookup
    followed by stacked layer-norm LSTM layers.

    Args:
        num_symbols (int): Vocabulary size (including blank).
        output_dim (int): Output encoding dimension.
        symbol_embedding_dim (int): Token embedding dimension.
        num_lstm_layers (int): Number of LSTM layers.
        lstm_hidden_dim (int): LSTM hidden dimension.
        lstm_layer_norm (bool): Whether to use layer-norm LSTM.
        lstm_layer_norm_epsilon (float): Epsilon for LayerNorm in LSTM.
        lstm_dropout (float): Dropout probability after each LSTM layer.
    """

    def __init__(
        self,
        num_symbols: int,
        output_dim: int,
        symbol_embedding_dim: int,
        num_lstm_layers: int,
        lstm_hidden_dim: int,
        lstm_layer_norm: bool = False,
        lstm_layer_norm_epsilon: float = 1e-5,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_symbols, symbol_embedding_dim)
        self.input_layer_norm = torch.nn.LayerNorm(symbol_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                CustomLSTM(
                    symbol_embedding_dim if idx == 0 else lstm_hidden_dim,
                    lstm_hidden_dim,
                    layer_norm=lstm_layer_norm,
                    layer_norm_epsilon=lstm_layer_norm_epsilon,
                )
                for idx in range(num_lstm_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(p=lstm_dropout)
        self.linear = torch.nn.Linear(lstm_hidden_dim, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(
        self,
        input: Tensor,
        lengths: Tensor,
        state: Optional[List[List[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, List[List[Tensor]]]:
        r"""Forward pass.

        Args:
            input (Tensor): Target token sequences, shape ``(B, U)``, with
                values in ``[0, num_symbols)``.
            lengths (Tensor): Valid token counts, shape ``(B,)``.
            state: Previous LSTM states, or ``None``.

        Returns:
            (Tensor, Tensor, List[List[Tensor]]):
                - Output encodings, shape ``(B, U, output_dim)``.
                - Output lengths (same as input), shape ``(B,)``.
                - New LSTM states.
        """
        input_tb = input.permute(1, 0)
        embedding_out = self.embedding(input_tb)
        x = self.input_layer_norm(embedding_out)

        state_out: List[List[Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            x, lstm_state_out = lstm(
                x, None if state is None else state[layer_idx]
            )
            x = self.dropout(x)
            state_out.append(lstm_state_out)

        x = self.linear(x)
        x = self.output_layer_norm(x)
        return x.permute(1, 0, 2), lengths, state_out
