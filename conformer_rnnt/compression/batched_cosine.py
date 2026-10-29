"""Batched offline cosine extrema and frame-weighted fusion.

The detector preserves the reference SciPy peak and valley semantics, including
plateau midpoints, peak+1 mapping, natural cardinality and fallback frame 0.
No target retention rate, threshold, or context partition is introduced.
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def frame_mask_from_cosine_distances(distances: Tensor, lengths: Tensor) -> Tensor:
    """Return selected [B,T] frame mask for raw distances [B,T-1].

    ``lengths`` are feature-frame lengths, so a length n has n-1 distances.
    This helper deliberately retains NaN comparison semantics of find_peaks.
    """
    batch, width = distances.shape
    device = distances.device
    frame_width = width + 1
    scores_length = (lengths.to(device=device, dtype=torch.long) - 1).clamp(min=0, max=width)
    if width == 0:
        return (lengths.to(device=device) > 0).unsqueeze(1)

    position = torch.arange(width, device=device).expand(batch, -1)
    # Every maximal equal-value plateau has its first and last index.  cummax
    # and reverse cummin propagate those bounds independently for each row.
    changed = torch.ones((batch, width), dtype=torch.bool, device=device)
    changed[:, 1:] = distances[:, 1:] != distances[:, :-1]
    starts = torch.where(changed, position, torch.zeros_like(position)).cummax(dim=1).values
    last = torch.ones((batch, width), dtype=torch.bool, device=device)
    last[:, :-1] = distances[:, :-1] != distances[:, 1:]
    ends = torch.where(last, position, torch.full_like(position, width - 1)).flip(1).cummin(dim=1).values.flip(1)
    left = distances.gather(1, (starts - 1).clamp(min=0))
    right = distances.gather(1, (ends + 1).clamp(max=width - 1))
    midpoint = position == ((starts + ends) // 2)
    # SciPy's find_peaks ignores sequence endpoints and cannot use padded
    # scores as neighbours. Rows with <3 distances have no extrema.
    interior = midpoint & (starts > 0) & (ends + 1 < scores_length[:, None]) & (position < scores_length[:, None])
    peak = interior & (distances > left) & (distances > right)
    valley = interior & (distances < left) & (distances < right)
    frame_mask = F.pad(valley, (0, 1)) | F.pad(peak, (1, 0))
    fallback = (lengths.to(device=device) > 0) & ~frame_mask.any(dim=1)
    # Non-in-place expression also avoids surprising aliases for autograd.
    frame_mask = frame_mask | (fallback[:, None] & (torch.arange(frame_width, device=device)[None, :] == 0))
    return frame_mask


def batched_natural_cosine_fusion(input: Tensor, lengths: Tensor, mixer):
    """Return compressed features, lengths, and selected mask.

    A single batch-level scalar synchronization is used to allocate only the
    observed maximum retained length. It is not a retention budget or quota.
    Both five_frames and nine_frames share the original predictor parameters.
    """
    if input.ndim != 3 or lengths.ndim != 1 or input.shape[0] != lengths.shape[0]:
        raise ValueError('Expected (B,T,F) features and (B,) lengths')
    batch, time, feature = input.shape
    if batch == 0:
        raise ValueError('Expected nonempty batch')
    if time == 0:
        if torch.any(lengths != 0):
            raise ValueError('Sequence length exceeds available frames')
        return input, lengths.new_zeros((batch,)), torch.zeros((batch,0),device=input.device,dtype=torch.bool)
    if mixer.weight_predictor_mode not in ('five_frames', 'nine_frames'):
        raise ValueError('Batched cosine fusion requires five_frames or nine_frames')
    window = 5 if mixer.weight_predictor_mode == 'five_frames' else 9
    if mixer.window_size != window:
        raise ValueError('Batched cosine fusion requires matching predictor and fusion windows')
    # Length checks run on CPU once for consistent error behavior and are
    # included in performance measurements; no per-utterance transfer.
    length_cpu = lengths.detach().cpu()
    if torch.any(length_cpu < 0) or torch.any(length_cpu > time):
        raise ValueError('Sequence length exceeds available frames')
    with torch.no_grad():
        normalized = F.normalize(input, dim=-1)
        distances = 1.0 - (normalized[:, 1:] * normalized[:, :-1]).sum(dim=-1)
        frame_mask = frame_mask_from_cosine_distances(distances, lengths)
        counts = frame_mask.sum(dim=1)
        max_count = int(counts.max().item())
        if max_count == 0:
            return input[:, :0], lengths.new_zeros((batch,)), frame_mask
        positions = torch.arange(time, device=input.device).expand(batch, -1)
        # Unique, ordered frame positions, with sentinel-valued batch padding.
        centres = torch.where(frame_mask, positions, time).sort(dim=1).values[:, :max_count]
        selected = centres < time
        centres = centres.clamp(max=time - 1)
        offsets = torch.arange(-(window // 2), window // 2 + 1, device=input.device)
        pos = centres[:, :, None] + offsets
        n = lengths.to(device=input.device, dtype=torch.long)
        valid_window = selected[:, :, None] & (pos >= 0) & (pos < n[:, None, None])
        # Predictor boundary rule: replicate closest valid frame, exactly as
        # the reference predictor does. Zero-length rows are masked later.
        clipped = torch.minimum(pos.clamp(min=0), (n - 1).clamp(min=0)[:, None, None])
    rows = torch.arange(batch, device=input.device)[:, None, None]
    windows = input[rows, clipped]
    windows = torch.where(selected[:, :, None, None], windows, torch.zeros_like(windows))
    logits = mixer.weight_predictor(windows.reshape(batch, max_count, window * feature))
    weights = torch.softmax(logits, dim=-1) * valid_window.to(logits.dtype)
    denominator = weights.sum(dim=-1, keepdim=True)
    # Reference normalisation is kept for every real centre; only batch padding
    # receives denominator 1 to avoid creating artificial 0/0.
    denominator = torch.where(selected[:, :, None], denominator, torch.ones_like(denominator))
    output = (windows * (weights / denominator).unsqueeze(-1)).sum(dim=2)
    # The reference returns the single original frame without calling its mixer.
    bypass = (n == 1)[:, None, None] & selected[:, :, None]
    centre_frame = input[torch.arange(batch, device=input.device)[:, None], centres]
    output = torch.where(bypass, centre_frame, output)
    return output, counts.to(dtype=lengths.dtype, device=lengths.device), frame_mask
