"""Frame-selection detectors for inter-block dynamic compression.

Contract
--------
Every detector implements::

    detect(frames: Tensor, diff: Tensor, diff_np: np.ndarray, main_len: int)
        -> (indices: LongTensor, scores: Tensor)

where ``frames`` is ``(main_len, F)`` (main part only, right context excluded),
``diff`` is the inter-frame L1 difference signal ``(main_len - 1,)`` with
``diff[i] = |frames[i+1] - frames[i]|`` summed over channels, and ``main_len``
is the number of compressible frames.

Returns:
    indices — sorted 1-D LongTensor of selected frame positions in
        ``[0, main_len)``. Must be non-empty.
    scores  — 1-D Tensor aligned with ``indices``, used by the universal
        equal-rate top-k mechanism in ``DynamicCompression``.

Rate model
----------
Detectors declare ``fixed_rate``:
    - ``fixed_rate=True``  — the returned index count already realises the
      configured keep rate (uniform, random, energy, topk-diff, max/avg-pool,
      MMDP). The universal top-k must NOT be applied again.
    - ``fixed_rate=False`` — dynamic peak-style detectors (find-peaks family,
      PLE, cosine-diff). When ``DynamicCompression.keep_rate`` is set, the
      top-k equal-rate mechanism picks the highest-scoring indices.

New baselines for the A4 equal-rate study (research-plan §5 I4):
PLE (DTM-Codec cumulative cosine distance) and CosineDiffPeaks (Audio Token
Compression style), plus fixed-rate selectors uniform / random / energy /
topk-diff / maxpool / avgpool / MMDP.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks, savgol_filter, firwin, filtfilt
from torch import Tensor


class Detector:
    """Base class. Subclasses set ``fixed_rate`` and implement ``detect``."""

    fixed_rate: bool = False

    def detect(
        self,
        frames: Tensor,
        diff: Tensor,
        diff_np: np.ndarray,
        main_len: int,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


def _diff_at(diff: Tensor, indices: Tensor, main_len: int) -> Tensor:
    """Score for frame t = |frames[t] - frames[t-1]|; frame 0 scores 0."""
    return torch.cat([diff.new_zeros(1), diff])[indices.clamp(0, main_len - 1)]


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic peak-style detectors (fixed_rate=False)
# ═══════════════════════════════════════════════════════════════════════════


class FindPeaksDetector(Detector):
    """Legacy peak/valley detection family.

    preprocess:
        "raw"    — raw diff → find_peaks  (legacy ``naive`` / ``hanning``)
        "savgol" — Savitzky-Golay smooth → find_peaks (legacy ``savgol``)
        "fir"    — FIR lowpass → coarse peaks → refine ±2 on raw (legacy ``fir``)
    """

    fixed_rate = False

    _FIR_CACHE = {}

    def __init__(self, preprocess: str = "raw") -> None:
        if preprocess not in ("raw", "savgol", "fir"):
            raise ValueError(f"Unknown preprocess '{preprocess}'")
        self.preprocess = preprocess

    def detect(self, frames, diff, diff_np, main_len):
        if self.preprocess == "savgol":
            smoothed = diff_np.copy()
            if len(smoothed) > 1:
                wlen = min(9, len(smoothed))
                if wlen % 2 == 0:
                    wlen -= 1
                if wlen > 3:
                    smoothed = savgol_filter(smoothed, window_length=wlen, polyorder=4)
            peaks, _ = find_peaks(smoothed)
            valleys, _ = find_peaks(-smoothed)
            merged = np.unique(np.concatenate([peaks + 1, valleys]))
        elif self.preprocess == "fir":
            smoothed = diff_np.copy()
            if len(smoothed) > 1:
                num_taps = min(21, len(smoothed) // 8)
                if num_taps >= 5:
                    if num_taps % 2 == 0:
                        num_taps -= 1
                    if num_taps not in self._FIR_CACHE:
                        nyquist = 0.5 * 200  # frame_rate = 200
                        self._FIR_CACHE[num_taps] = firwin(
                            num_taps, 50 / nyquist, pass_zero="lowpass"
                        )
                    smoothed = filtfilt(self._FIR_CACHE[num_taps], 1.0, smoothed)
            peaks_approx, _ = find_peaks(smoothed)
            valleys_approx, _ = find_peaks(-smoothed)
            radius, raw_len = 2, len(diff_np)
            peaks_refined = [
                max(0, p - radius) + np.argmax(diff_np[max(0, p - radius):min(raw_len, p + radius + 1)])
                for p in peaks_approx
            ]
            valleys_refined = [
                max(0, v - radius) + np.argmin(diff_np[max(0, v - radius):min(raw_len, v + radius + 1)])
                for v in valleys_approx
            ]
            merged = np.unique(
                np.concatenate([np.array(peaks_refined) + 1, np.array(valleys_refined)])
            )
        else:  # raw
            peaks, _ = find_peaks(diff_np)
            valleys, _ = find_peaks(-diff_np)
            merged = np.unique(np.concatenate([peaks + 1, valleys]))

        device = diff.device
        indices = torch.as_tensor(merged, device=device).long()
        indices = indices[indices < main_len]
        if len(indices) == 0:
            indices = torch.zeros(1, dtype=torch.long, device=device)
        return indices, _diff_at(diff, indices, main_len)


class PLEDetector(Detector):
    """Prediction-Length Encoder style boundary detection (DTM-Codec).

    Walks the cumulative cosine distance between adjacent frames; every time
    the accumulator exceeds ``threshold``, the frame that caused the jump is
    marked as a segment boundary and the accumulator resets.
    """

    fixed_rate = False

    def __init__(self, threshold: float = 6.0) -> None:
        self.threshold = threshold

    def detect(self, frames, diff, diff_np, main_len):
        f = torch.nn.functional.normalize(frames, dim=-1)
        cos_dist = 1.0 - (f[1:] * f[:-1]).sum(dim=-1)  # (main_len - 1,)

        indices = [0]
        scores = [0.0]
        cum = 0.0
        for i, c in enumerate(cos_dist.tolist()):
            cum += c
            if cum >= self.threshold:
                indices.append(i + 1)
                scores.append(c)
                cum = 0.0
        return (
            torch.tensor(indices, dtype=torch.long, device=frames.device),
            torch.tensor(scores, device=frames.device),
        )


class CosineDiffPeaksDetector(Detector):
    """Audio Token Compression style: peak of per-frame cosine distance.

    ``c_t = 1 - cos(frames[t], frames[t+1])``; a boundary is kept at ``t+1``
    when ``c_t`` is a local maximum (find_peaks), valley positions at ``t``
    when ``c_t`` is a local minimum.
    """

    fixed_rate = False

    def detect(self, frames, diff, diff_np, main_len):
        f = torch.nn.functional.normalize(frames, dim=-1)
        c = (1.0 - (f[1:] * f[:-1]).sum(dim=-1)).cpu().numpy()
        if len(c) < 3:
            return (
                torch.zeros(1, dtype=torch.long, device=frames.device),
                torch.zeros(1, device=frames.device),
            )
        peaks, _ = find_peaks(c)
        valleys, _ = find_peaks(-c)
        merged = np.unique(np.concatenate([peaks + 1, valleys]))
        merged = merged[merged < main_len]
        if len(merged) == 0:
            merged = np.array([0])
        device = frames.device
        indices = torch.as_tensor(merged, device=device).long()
        scores = torch.as_tensor(c, device=device, dtype=torch.float32)[
            indices.clamp(0, len(c) - 1)
        ]
        return indices, scores


# ═══════════════════════════════════════════════════════════════════════════
# Fixed-rate detectors (fixed_rate=True)
# ═══════════════════════════════════════════════════════════════════════════


def _num_keep(main_len: int, rate: float) -> int:
    return max(1, round(main_len * rate))


class UniformDetector(Detector):
    """Evenly spaced frames at ``rate``. Fixed rate."""

    fixed_rate = True

    def __init__(self, rate: float) -> None:
        if not 0 < rate <= 1:
            raise ValueError(f"rate must be in (0, 1], got {rate}")
        self.rate = rate

    def detect(self, frames, diff, diff_np, main_len):
        k = _num_keep(main_len, self.rate)
        indices = torch.linspace(0, main_len - 1, k, device=frames.device).round().long()
        indices = torch.unique(indices)
        return indices, _diff_at(diff, indices, main_len)


class RandomDetector(Detector):
    """Random frames at ``rate``. Deterministic per utterance length
    (same-length utterances share the draw — documented baseline weakness)."""

    fixed_rate = True

    def __init__(self, rate: float, seed: int = 0) -> None:
        self.rate = rate
        self.seed = seed

    def detect(self, frames, diff, diff_np, main_len):
        k = _num_keep(main_len, self.rate)
        rng = np.random.default_rng(self.seed + main_len)
        idx = np.sort(rng.choice(main_len, size=min(k, main_len), replace=False))
        indices = torch.as_tensor(idx, device=frames.device).long()
        return indices, _diff_at(diff, indices, main_len)


class EnergyTopKDetector(Detector):
    """Top-``rate`` frames by L2 energy."""

    fixed_rate = True

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def detect(self, frames, diff, diff_np, main_len):
        k = _num_keep(main_len, self.rate)
        scores = frames[:main_len].norm(dim=-1)
        _, indices = torch.topk(scores, k=min(k, main_len))
        indices, _ = torch.sort(indices)
        return indices, scores[indices]


class TopKDiffDetector(Detector):
    """Top-``rate`` frames by inter-frame difference magnitude."""

    fixed_rate = True

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def detect(self, frames, diff, diff_np, main_len):
        k = _num_keep(main_len, self.rate)
        padded = torch.cat([diff.new_zeros(1), diff])  # (main_len,)
        _, indices = torch.topk(padded, k=min(k, main_len))
        indices, _ = torch.sort(indices)
        return indices, padded[indices]


class _SegmentedDetector(Detector):
    """Shared segmentation helper: fixed-rate segments of size
    ``round(1/rate)``; each detector picks one representative per segment."""

    fixed_rate = True

    def __init__(self, rate: float) -> None:
        self.rate = rate
        self.seg_size = max(1, round(1.0 / rate))

    def _segments(self, main_len: int):
        bounds = list(range(0, main_len, self.seg_size))
        if bounds[0] != 0:
            bounds.insert(0, 0)
        return [(s, min(s + self.seg_size, main_len)) for s in bounds]


class MaxPoolDetector(_SegmentedDetector):
    """Per-segment argmax of inter-frame difference."""

    def detect(self, frames, diff, diff_np, main_len):
        padded = torch.cat([diff.new_zeros(1), diff])
        indices, scores = [], []
        for s, e in self._segments(main_len):
            local = padded[s:e]
            i = s + int(torch.argmax(local).item())
            indices.append(i)
            scores.append(padded[i].item())
        return (
            torch.tensor(indices, dtype=torch.long, device=frames.device),
            torch.tensor(scores, device=frames.device),
        )


class AvgPoolDetector(_SegmentedDetector):
    """Per-segment first frame as representative; scores = segment mean diff.

    Intended to pair with :class:`~conformer_rnnt.compression.mixers.SegmentAvgMixer`
    (true avg-pool = uniform segments + segment-average mixing); with the hard
    mixer it degrades to uniform-on-segment-boundaries.
    """

    def detect(self, frames, diff, diff_np, main_len):
        padded = torch.cat([diff.new_zeros(1), diff])
        indices, scores = [], []
        for s, e in self._segments(main_len):
            indices.append(s)
            scores.append(padded[s:e].mean().item())
        return (
            torch.tensor(indices, dtype=torch.long, device=frames.device),
            torch.tensor(scores, device=frames.device),
        )


class MMDPDetector(_SegmentedDetector):
    """Per-segment medoid-style diversity pick: the frame with maximum cosine
    distance to its segment mean (multi-modal diversity pruning heuristic)."""

    def detect(self, frames, diff, diff_np, main_len):
        indices, scores = [], []
        for s, e in self._segments(main_len):
            seg = torch.nn.functional.normalize(frames[s:e], dim=-1)
            mean = seg.mean(dim=0, keepdim=True)
            dist = 1.0 - (seg * mean).sum(dim=-1)
            i = s + int(torch.argmax(dist).item())
            indices.append(i)
            scores.append(float(dist.max().item()))
        return (
            torch.tensor(indices, dtype=torch.long, device=frames.device),
            torch.tensor(scores, device=frames.device),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

# type -> (class, param key mapping from detector.params)
DETECTOR_REGISTRY = {
    "raw": FindPeaksDetector,
    "fir": FindPeaksDetector,
    "savgol": FindPeaksDetector,
    "ple": PLEDetector,
    "cosine_diff": CosineDiffPeaksDetector,
    "uniform": UniformDetector,
    "random": RandomDetector,
    "energy": EnergyTopKDetector,
    "topk_diff": TopKDiffDetector,
    "max_pool": MaxPoolDetector,
    "avg_pool": AvgPoolDetector,
    "mmdp": MMDPDetector,
}


def build_detector(cfg: dict) -> Detector:
    """Build a detector from ``{type, params}`` config dict."""
    cfg = dict(cfg or {})
    dtype = cfg.get("type", "raw")
    params = dict(cfg.get("params", {}))
    if dtype in ("fir", "savgol"):
        params.setdefault("preprocess", dtype)
    elif dtype == "raw":
        params.setdefault("preprocess", "raw")
    if dtype not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector type '{dtype}'; known: {sorted(DETECTOR_REGISTRY)}")
    if dtype == "fir":
        params.pop("preprocess", None)  # FindPeaksDetector(fir) drives FIR itself
        return FindPeaksDetector("fir")
    cls = DETECTOR_REGISTRY[dtype]
    kwargs = {}
    if cls is FindPeaksDetector:
        kwargs["preprocess"] = params.get("preprocess", "raw")
    else:
        kwargs.update(params)
    return cls(**kwargs)
