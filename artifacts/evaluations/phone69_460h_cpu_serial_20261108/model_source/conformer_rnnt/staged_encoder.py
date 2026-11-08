"""Whole-utterance Conformer with explicitly positioned compression modules."""

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from conformer_rnnt.compression.stages import CompressionSchedule
from conformer_rnnt.encoder import TimeReduction, _make_conformer


class StagedConformerEncoder(torch.nn.Module):
    """Layer 0 compresses model-input Mel; other stages follow global layers.

    This is opt-in. Existing two-block encoders keep their construction, state
    keys, random initialization order and forward path.
    """

    def __init__(
        self, *, input_dim: int, output_dim: int, time_reduction_input_dim: int,
        time_reduction_stride: int, num_heads: int, ffn_dim: int, num_layers: int,
        compression_stages: List[dict], dropout: float = 0.1,
        depthwise_conv_kernel_size: int = 31,
    ) -> None:
        super().__init__()
        self.sequence_mode = "offline"
        self.time_reduction_stride = time_reduction_stride
        dim = time_reduction_input_dim * time_reduction_stride
        self.input_linear = torch.nn.Linear(input_dim, time_reduction_input_dim, bias=False)
        self.time_reduction = TimeReduction(time_reduction_stride, pad_to_stride=True)
        self.compression = CompressionSchedule(compression_stages, num_layers, input_dim, dim)
        self.block_ends = [
            stage["after_layer"] for stage in self.compression.definitions if stage["after_layer"] > 0
        ] + [num_layers]
        starts = [0] + self.block_ends[:-1]
        self.block_sizes = [end - start for start, end in zip(starts, self.block_ends)]
        self.conformer_blocks = torch.nn.ModuleList([
            _make_conformer(dim, num_heads, ffn_dim, count, depthwise_conv_kernel_size, dropout)
            for count in self.block_sizes
        ])
        self.output_linear = torch.nn.Linear(dim, output_dim)
        self.output_norm = torch.nn.LayerNorm(output_dim)
        self.placement_definition = {
            "num_layers": num_layers, "layers": list(self.block_sizes),
            "compression_stages": deepcopy(self.compression.definitions),
            "mel_input": "after_existing_data_pipeline_before_projection_and_stacking",
        }

    def forward(self, input: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        self.compression.start_batch(lengths)
        valid = torch.arange(input.size(1), device=input.device)[None, :] < lengths[:, None]
        x = input.masked_fill(~valid[:, :, None], 0.0)
        x, lengths = self.compression.at_layer(0, x, lengths)
        x, lengths = self.time_reduction(self.input_linear(x), lengths)
        self.compression.last_lengths["encoder_input"] = lengths.tolist()
        self.compression.last_lengths["stride"] = self.time_reduction_stride
        for end, block in zip(self.block_ends, self.conformer_blocks):
            x, lengths = block(x, lengths)
            x, lengths = self.compression.at_layer(end, x, lengths)
        self.compression.last_lengths["output"] = lengths.tolist()
        return self.output_norm(self.output_linear(x)), lengths

    @torch.jit.export
    def infer(
        self, input: Tensor, lengths: Tensor,
        states: Optional[List[List[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, List[List[Tensor]]]:
        """Decoding adapter; computation remains whole-utterance and non-streaming."""
        output, out_lengths = self.forward(input, lengths)
        return output, out_lengths, []
