"""RNN-T joint network.

Fuses source (acoustic) and target (language) encodings via additive fusion:
    joint = activation(source_encodings + target_encodings)
    logits = linear(joint)
"""

from typing import Dict, Tuple

import torch
from torch import Tensor

# Registry: activation name → module class (extensible)
_ACTIVATIONS: Dict[str, type] = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
}


class Joiner(torch.nn.Module):
    """RNN-T joint network.

    Args:
        input_dim: Dimension of source and target encodings (must match).
        output_dim: Vocabulary size (number of target symbols).
        activation: Non-linearity name. Supported: ``"relu"``, ``"tanh"``,
            ``"gelu"``, ``"silu"``. Default: ``"relu"``.
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu") -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from: {list(_ACTIVATIONS)}."
            )
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.activation = _ACTIVATIONS[activation]()

    def forward(
        self,
        source_encodings: Tensor,
        source_lengths: Tensor,
        target_encodings: Tensor,
        target_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Fuse encoder and predictor outputs into vocabulary logits.

        Args:
            source_encodings: Encoder output, shape ``(B, T, D)``.
            source_lengths: Valid source frames, shape ``(B,)``.
            target_encodings: Predictor output, shape ``(B, U, D)``.
            target_lengths: Valid target tokens, shape ``(B,)``.

        Returns:
            (logits, source_lengths, target_lengths):
                - logits: ``(B, T, U, output_dim)``
                - source_lengths: pass-through ``(B,)``
                - target_lengths: pass-through ``(B,)``
        """
        # Broadcast: (B,T,1,D) + (B,1,U,D) → (B,T,U,D)
        joint = source_encodings.unsqueeze(2) + target_encodings.unsqueeze(1)
        logits = self.linear(self.activation(joint))
        return logits, source_lengths, target_lengths
