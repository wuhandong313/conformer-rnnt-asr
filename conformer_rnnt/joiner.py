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


def get_activation(name: str) -> torch.nn.Module:
    """Build an activation module by name, raising on unknown names."""
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(_ACTIVATIONS)}."
        )
    return _ACTIVATIONS[name]()


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
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.activation = get_activation(activation)

    def fuse(
        self,
        source_encodings: Tensor,
        target_encodings: Tensor,
    ) -> Tensor:
        """Broadcast-add source and target encodings, then apply activation.

        The fused ``joint`` is the single source of truth shared by the joiner
        logits and the Medusa heads — keeping head 0 ≡ joiner exact.
        """
        # (B,T,1,D) + (B,1,U,D) → (B,T,U,D)
        joint = source_encodings.unsqueeze(2) + target_encodings.unsqueeze(1)
        return self.activation(joint)

    def project(self, joint: Tensor) -> Tensor:
        """Project fused features to vocabulary logits."""
        return self.linear(joint)

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
        logits = self.project(self.fuse(source_encodings, target_encodings))
        return logits, source_lengths, target_lengths
