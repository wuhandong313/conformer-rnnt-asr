"""Frame-mixing backends for inter-block dynamic compression.

Contract
--------
Every mixer implements::

    mix(frames: Tensor, indices: Tensor, diff: Tensor, main_len: int) -> Tensor

returning the compressed main part ``(N, F)`` where ``N == len(indices)``
(``indices`` sorted ascending, all in ``[0, main_len)``).

Mixers
------
    HardSelectMixer     — index_select (legacy hard selection).
    SegmentAvgMixer     — replace each kept frame by the mean of the segment
                          it starts (true avg-pool when paired with uniform
                          segments).
    LearnedWeightMixer  — learnable Hanning-window weighted blending around
                          each boundary (legacy ``hanning`` path); owns the
                          weight-predictor MLP (registered in the parent module).
"""

import torch
from torch import Tensor


class HardSelectMixer:
    def mix(self, frames: Tensor, indices: Tensor, diff: Tensor, main_len: int) -> Tensor:
        return frames.index_select(0, indices)


class SegmentAvgMixer:
    def mix(self, frames: Tensor, indices: Tensor, diff: Tensor, main_len: int) -> Tensor:
        # Segment i = [indices[i], indices[i+1]) (last segment ends at main_len).
        bounds = torch.cat([indices, torch.tensor([main_len], device=indices.device)])
        rows = []
        for i in range(len(indices)):
            s, e = int(bounds[i]), int(bounds[i + 1])
            rows.append(frames[s:e].mean(dim=0))
        return torch.stack(rows)


class LearnedWeightMixer(torch.nn.Module):
    """Legacy learnable Hanning-window blending (moved verbatim from
    ``DynamicCompression._hanning_smooth`` + weight-predictor MLP)."""

    def __init__(
        self,
        feature_dim: int,
        window_size: int = 5,
        weight_predictor_mode: str = "diff5",
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.weight_predictor_mode = weight_predictor_mode

        if weight_predictor_mode == "diff5":
            self.weight_predictor = torch.nn.Sequential(
                torch.nn.Linear(4, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, window_size),
            )
        elif weight_predictor_mode == "five_frames":
            self.weight_predictor = torch.nn.Sequential(
                torch.nn.Linear(5 * feature_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, window_size),
            )
        elif weight_predictor_mode == "single_frame":
            self.weight_predictor = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, window_size),
            )
        else:
            raise ValueError(
                f"Unknown weight_predictor_mode '{weight_predictor_mode}'"
            )

    def mix(self, frames: Tensor, indices: Tensor, diff: Tensor, main_len: int) -> Tensor:
        half_win = self.window_size // 2
        N = len(indices)

        if self.weight_predictor_mode == "diff5":
            offsets = torch.tensor([-2, -1, 0, 1, 2], device=diff.device)
            idx5 = (indices.unsqueeze(-1) + offsets).clamp(0, len(diff) - 1)
            five = diff[idx5]  # (N, 5)
            peak_features = torch.abs(five[:, 1:] - five[:, :-1])  # (N, 4)
        elif self.weight_predictor_mode == "five_frames":
            offsets = torch.tensor([-2, -1, 0, 1, 2], device=frames.device)
            idx5 = (indices.unsqueeze(-1) + offsets).clamp(0, main_len - 1)
            peak_features = frames[idx5].reshape(N, -1)  # (N, 5*F)
        else:  # single_frame
            peak_features = frames[indices]  # (N, F)

        all_weights = torch.softmax(self.weight_predictor(peak_features), dim=-1)

        smoothed = []
        for i, idx in enumerate(indices):
            start = max(0, int(idx) - half_win)
            end = min(main_len, int(idx) + half_win + 1)
            w_start = start - (int(idx) - half_win)
            w_end = self.window_size - ((int(idx) + half_win + 1) - end)
            weights = all_weights[i, w_start:w_end]
            weights = weights / weights.sum()
            smoothed.append((frames[start:end] * weights.unsqueeze(-1)).sum(dim=0))

        return torch.stack(smoothed)


MIXER_REGISTRY = {
    "hard": HardSelectMixer,
    "seg_avg": SegmentAvgMixer,
    "learned": LearnedWeightMixer,
}


def build_mixer(cfg: dict, feature_dim: int, window_size: int, weight_predictor_mode: str):
    """Build a mixer from ``{type, mode}`` config dict."""
    cfg = dict(cfg or {})
    mtype = cfg.get("type", "hard")
    if mtype not in MIXER_REGISTRY:
        raise ValueError(f"Unknown mixer type '{mtype}'; known: {sorted(MIXER_REGISTRY)}")
    if mtype == "learned":
        return LearnedWeightMixer(
            feature_dim=feature_dim,
            window_size=window_size,
            weight_predictor_mode=cfg.get("mode", weight_predictor_mode),
        )
    return MIXER_REGISTRY[mtype]()
