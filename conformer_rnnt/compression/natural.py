"""Offline, natural peak/valley compression of the entire valid sequence.

There is deliberately no rate argument, quota, or special tail-frame segment.
Legacy equal-rate/context-aware experiments remain in ``core.py``.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from conformer_rnnt.compression.detectors import build_detector, CosineDiffPeaksDetector
from conformer_rnnt.compression.mixers import build_mixer, LearnedWeightMixer
from conformer_rnnt.compression.batched_cosine import batched_natural_cosine_fusion


class NaturalPeakCompression(torch.nn.Module):
    def __init__(
        self,
        detector: Optional[dict] = None,
        mixer: Optional[dict] = None,
        feature_dim: int = 128,
        window_size: int = 5,
        weight_predictor_mode: str = "diff5",
        passthrough: bool = False,
        neighborhood_radius: int = 0,
        execution_backend: str = "reference",
    ) -> None:
        super().__init__()
        self.passthrough = passthrough
        if execution_backend not in {"reference", "batched"}:
            raise ValueError(f"Unknown compression execution backend: {execution_backend}")
        self.execution_backend = execution_backend
        self.last_lengths = None
        if type(neighborhood_radius) is not int or neighborhood_radius < 0:
            raise ValueError("neighborhood_radius must be a non-negative integer")
        if neighborhood_radius and (mixer or {}).get("type") != "hard":
            raise ValueError("Neighborhood retention requires the hard mixer")
        self.neighborhood_radius = neighborhood_radius
        self.last_candidate_lengths = None
        self.record_indices = False
        self.last_indices = None
        if passthrough:
            if execution_backend != "reference":
                raise ValueError("Batched backend requires active cosine compression")
            return
        detector = detector or {"type": "cosine_diff"}
        if detector["type"] not in {"raw", "savgol", "fir", "cosine_diff"}:
            raise ValueError("Offline natural compression requires a peak/valley detector")
        self.detector = build_detector(detector)
        self.mixer = build_mixer(
            mixer or {"type": "learned"}, feature_dim, window_size, weight_predictor_mode
        )
        if execution_backend == "batched":
            if not isinstance(self.detector, CosineDiffPeaksDetector) or self.detector.preprocess != "none":
                raise ValueError("Batched backend requires unsmoothed cosine_diff")
            if not isinstance(self.mixer, LearnedWeightMixer) or self.mixer.weight_predictor_mode not in {"five_frames", "nine_frames"}:
                raise ValueError("Batched backend requires a five_frames or nine_frames learned mixer")
            expected = 5 if self.mixer.weight_predictor_mode == "five_frames" else 9
            if self.mixer.window_size != expected or self.neighborhood_radius:
                raise ValueError("Batched backend requires matching window and no neighborhood retention")

    def forward(self, input: Tensor, lengths: Tensor):
        if input.dim() != 3 or lengths.dim() != 1 or input.size(0) != lengths.size(0):
            raise ValueError("Expected (B,T,F) features and (B,) lengths")
        if self.execution_backend == "batched":
            output, result_lengths, selected = batched_natural_cosine_fusion(input, lengths, self.mixer)
            stats = torch.stack((lengths.to(result_lengths.device), result_lengths)).detach().cpu().tolist()
            self.last_lengths = dict(input=stats[0], output=stats[1])
            self.last_candidate_lengths = stats[1]
            self.last_indices = None
            if self.record_indices:
                selected_cpu = selected.detach().cpu()
                retained = [torch.where(row)[0].tolist() for row in selected_cpu]
                self.last_indices = {"centres": retained, "retained": retained}
            return output, result_lengths
        sizes = lengths.tolist()
        if any(n < 0 or n > input.size(1) for n in sizes):
            raise ValueError("Sequence length exceeds available frames")
        if self.passthrough:
            return input, lengths
        output = []
        candidate_lengths = []
        centre_trace, retained_trace = [], []
        for frames, n in zip(input, sizes):
            frames = frames[:n]
            if n <= 1:
                output.append(frames)
                candidate_lengths.append(n)
                if self.record_indices:
                    centre_trace.append(list(range(n)))
                    retained_trace.append(list(range(n)))
                continue
            diff = (frames[1:] - frames[:-1]).abs().sum(dim=-1)
            with torch.no_grad():
                indices, _ = self.detector.detect(frames, diff, diff.detach().cpu().numpy(), n)
                indices = torch.unique(indices[(indices >= 0) & (indices < n)], sorted=True)
                if indices.numel() == 0:
                    indices = torch.zeros(1, dtype=torch.long, device=input.device)
                candidate_lengths.append(indices.numel())
                if self.record_indices:
                    centre_trace.append(indices.tolist())
                if self.neighborhood_radius:
                    offsets = torch.arange(
                        -self.neighborhood_radius, self.neighborhood_radius + 1, device=input.device
                    )
                    expanded = (indices[:, None] + offsets).flatten()
                    indices = torch.unique(expanded[(expanded >= 0) & (expanded < n)], sorted=True)
                if self.record_indices:
                    retained_trace.append(indices.tolist())
            output.append(self.mixer.mix(frames, indices, diff, n))
        output_sizes = [x.size(0) for x in output]
        self.last_lengths = dict(input=sizes, output=output_sizes)
        self.last_candidate_lengths = candidate_lengths
        self.last_indices = (
            {"centres": centre_trace, "retained": retained_trace} if self.record_indices else None
        )
        return (
            pad_sequence(output, batch_first=True),
            lengths.new_tensor(output_sizes),
        )
