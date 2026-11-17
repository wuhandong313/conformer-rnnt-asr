"""Fixed temporal smoothing of encoder features used only for detection."""

import math
from numbers import Real
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def validate_smoothing_alpha(alpha: float) -> float:
    """Validate the strength of identity/binomial interpolation."""
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("feature_smoothing_alpha must be a finite number in [0, 1]")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("feature_smoothing_alpha must be a finite number in [0, 1]")
    return alpha


def smooth_feature_frames(
    frames: Tensor, alpha: float, lengths: Optional[Tensor] = None,
) -> Tensor:
    """Smooth (T,D) or (B,T,D) along time with one kernel for all channels.

    The interior kernel is [alpha/4, 1-alpha/2, alpha/4]. Missing or padded
    neighbours are excluded and weights renormalized; padded outputs are zero.
    This operation neither selects frames nor changes sequence lengths.
    """
    alpha = validate_smoothing_alpha(alpha)
    if frames.ndim not in (2, 3) or not frames.is_floating_point():
        raise ValueError("Expected floating-point (T,D) or (B,T,D) features")
    unbatched = frames.ndim == 2
    x = frames.unsqueeze(0) if unbatched else frames
    batch, time, _ = x.shape
    if lengths is None:
        valid = torch.ones((batch, time), dtype=torch.bool, device=x.device)
    else:
        if lengths.ndim != 1 or lengths.numel() != batch:
            raise ValueError("Expected one length per feature sequence")
        valid = torch.arange(time, device=x.device)[None, :] < lengths.to(x.device)[:, None]
    x = x.masked_fill(~valid[:, :, None], 0.0)
    if alpha == 0.0 or time == 0:
        return x.squeeze(0) if unbatched else x

    side = alpha / 4.0
    centre = 1.0 - alpha / 2.0
    left = F.pad(x[:, :-1], (0, 0, 1, 0))
    right = F.pad(x[:, 1:], (0, 0, 0, 1))
    weights = valid.to(x.dtype)
    left_valid = F.pad(weights[:, :-1], (1, 0))
    right_valid = F.pad(weights[:, 1:], (0, 1))
    denominator = centre * weights + side * (left_valid + right_valid)
    denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    output = (centre * x + side * (left + right)) / denominator[:, :, None]
    output = output.masked_fill(~valid[:, :, None], 0.0)
    return output.squeeze(0) if unbatched else output
