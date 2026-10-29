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
        # Vectorized: map each frame to its segment id, accumulate with
        # index_add, divide by segment length (numerically equivalent to the
        # original per-segment Python loop).
        N = len(indices)
        bounds = torch.cat([indices, torch.tensor([main_len], device=indices.device)])
        lengths = bounds[1:] - bounds[:-1]
        seg_id = torch.repeat_interleave(
            torch.arange(N, device=frames.device), lengths
        )
        acc = torch.zeros(
            N, frames.shape[-1], device=frames.device, dtype=frames.dtype
        )
        # First segment starts at indices[0]; frames before it belong to no
        # segment (the original per-segment loop never touched them either).
        acc.index_add_(0, seg_id, frames[indices[0]:main_len])
        return acc / lengths.unsqueeze(-1).to(frames.dtype)


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
        elif weight_predictor_mode == "nine_frames":
            if window_size != 9:
                raise ValueError("nine_frames requires window_size=9")
            self.weight_predictor = torch.nn.Sequential(
                torch.nn.Linear(9 * feature_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 9),
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
        elif self.weight_predictor_mode == "nine_frames":
            offsets = torch.arange(-4, 5, device=frames.device)
            idx9 = (indices.unsqueeze(-1) + offsets).clamp(0, main_len - 1)
            peak_features = frames[idx9].reshape(N, -1)  # (N, 9*F)
        else:  # single_frame
            peak_features = frames[indices]  # (N, F)

        all_weights = torch.softmax(self.weight_predictor(peak_features), dim=-1)

        # Vectorized windowed blend: gather the full (possibly clipped) window
        # for every kept frame, zero out weights of out-of-range positions
        # (equivalent to the original w_start/w_end slice + renormalize), then
        # weighted-sum in one shot.
        offsets = torch.arange(-half_win, half_win + 1, device=frames.device)
        pos = indices.unsqueeze(-1) + offsets  # (N, W)
        valid = (pos >= 0) & (pos < main_len)
        weights = all_weights * valid.to(all_weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        windows = frames[pos.clamp(0, main_len - 1)]  # (N, W, F)
        return (windows * weights.unsqueeze(-1)).sum(dim=1)


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
