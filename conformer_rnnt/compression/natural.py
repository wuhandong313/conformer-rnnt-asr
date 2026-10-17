"""Offline, natural peak/valley compression of the entire valid sequence.

There is deliberately no rate argument, quota, or special tail-frame segment.
Legacy equal-rate/context-aware experiments remain in ``core.py``.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from conformer_rnnt.compression.detectors import build_detector
from conformer_rnnt.compression.mixers import build_mixer


class NaturalPeakCompression(torch.nn.Module):
    def __init__(
        self,
        detector: Optional[dict] = None,
        mixer: Optional[dict] = None,
        feature_dim: int = 128,
        window_size: int = 5,
        weight_predictor_mode: str = "diff5",
        passthrough: bool = False,
    ) -> None:
        super().__init__()
        self.passthrough = passthrough
        self.last_lengths = None
        if passthrough:
            return
        detector = detector or {"type": "cosine_diff"}
        if detector["type"] not in {"raw", "savgol", "fir", "cosine_diff"}:
            raise ValueError("Offline natural compression requires a peak/valley detector")
        self.detector = build_detector(detector)
        self.mixer = build_mixer(
            mixer or {"type": "learned"}, feature_dim, window_size, weight_predictor_mode
        )

    def forward(self, input: Tensor, lengths: Tensor):
        if input.dim() != 3 or lengths.dim() != 1 or input.size(0) != lengths.size(0):
            raise ValueError("Expected (B,T,F) features and (B,) lengths")
        sizes = lengths.tolist()
        if any(n < 0 or n > input.size(1) for n in sizes):
            raise ValueError("Sequence length exceeds available frames")
        if self.passthrough:
            return input, lengths
        output = []
        for frames, n in zip(input, sizes):
            frames = frames[:n]
            if n <= 1:
                output.append(frames)
                continue
            diff = (frames[1:] - frames[:-1]).abs().sum(dim=-1)
            with torch.no_grad():
                indices, _ = self.detector.detect(frames, diff, diff.detach().cpu().numpy(), n)
                indices = torch.unique(indices[(indices >= 0) & (indices < n)], sorted=True)
                if indices.numel() == 0:
                    indices = torch.zeros(1, dtype=torch.long, device=input.device)
            output.append(self.mixer.mix(frames, indices, diff, n))
        output_sizes = [x.size(0) for x in output]
        self.last_lengths = dict(input=sizes, output=output_sizes)
        return (
            pad_sequence(output, batch_first=True),
            lengths.new_tensor(output_sizes),
        )
