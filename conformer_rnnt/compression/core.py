"""Dynamic inter-frame compression: detector × mixer composition.

Architecture (research-plan v2.1, detector × mixer orthogonal axes):

    DynamicCompression(right_context_length,
                       detector=..., mixer=..., keep_rate=None)

    forward(x, lengths):
        per sample b:
            main part / right-context split (unchanged)
            indices, scores = detector.detect(main_part, diff, main_len)
            if keep_rate is set and not detector.fixed_rate:
                indices = top-k(scores, round(main_len * keep_rate))   # equal-rate
            out = mixer.mix(main_part, indices, diff, main_len)

Legacy ``compression.filter`` enum maps onto the new axes via
:meth:`DynamicCompression.from_legacy`:

    none           -> passthrough (encoder bypasses)
    naive          -> (FindPeaksDetector raw,   HardSelectMixer)
    fir            -> (FindPeaksDetector fir,   HardSelectMixer)
    savgol         -> (FindPeaksDetector savgol, HardSelectMixer)
    hanning        -> (FindPeaksDetector raw,   LearnedWeightMixer)
    savgol_hanning -> (FindPeaksDetector savgol, LearnedWeightMixer)

New-style config schema::

    model.compression:
        detector: {type: savgol, params: {}}     # see detectors.DETECTOR_REGISTRY
        mixer:    {type: learned, mode: diff5}   # see mixers.MIXER_REGISTRY
        keep_rate: null                           # null = legacy dynamic rate
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from conformer_rnnt.compression.detectors import (
    Detector,
    FindPeaksDetector,
    build_detector,
)
from conformer_rnnt.compression.mixers import (
    HardSelectMixer,
    LearnedWeightMixer,
    build_mixer,
)

# Legacy filter enum -> (detector type, mixer type)
_LEGACY_MAP = {
    "naive": ("raw", "hard"),
    "fir": ("fir", "hard"),
    "savgol": ("savgol", "hard"),
    "hanning": ("raw", "learned"),
    "savgol_hanning": ("savgol", "learned"),
}


class DynamicCompression(torch.nn.Module):
    r"""Equal-rate-capable inter-block frame compression.

    Args:
        right_context_length: Right-context frames excluded from compression.
        detector: ``{type, params}`` detector config dict.
        mixer: ``{type, mode}`` mixer config dict.
        feature_dim: Encoder feature dim (used by the learned mixer MLP).
        keep_rate: Universal equal-rate knob. When set (0 < r <= 1) and the
            detector is dynamic (``fixed_rate=False``), detected indices are
            re-ranked by score and top-``round(main_len * r)`` kept (frame 0
            always retained). Ignored for fixed-rate detectors.
        hanning_window_size: Window size for the learned mixer.
        weight_predictor_mode: Learned-mixer MLP input mode.
        passthrough: When True the module is an identity (legacy ``none``).
    """

    def __init__(
        self,
        right_context_length: int,
        detector: Optional[Dict[str, Any]] = None,
        mixer: Optional[Dict[str, Any]] = None,
        feature_dim: int = 512,
        keep_rate: Optional[float] = None,
        hanning_window_size: int = 5,
        weight_predictor_mode: str = "diff5",
        passthrough: bool = False,
    ) -> None:
        super().__init__()
        self.right_context_length = right_context_length
        self.keep_rate = keep_rate
        self.passthrough = bool(passthrough)

        detector = dict(detector or {"type": "raw"})
        mixer = dict(mixer or {"type": "hard"})
        self.detector_cfg = detector
        self.mixer_cfg = mixer

        # Human-readable filter-style tag, e.g. "savgol" / "savgol+learned".
        mix_type = mixer.get("type", "hard")
        self.compression_filter = (
            detector["type"]
            if mix_type == "hard"
            else f"{detector['type']}+{mix_type}"
        )

        if self.passthrough:
            self.detector = None
            self.mixer = None
            return

        self.detector: Detector = build_detector(detector)
        self.mixer = build_mixer(
            mixer, feature_dim, hanning_window_size, weight_predictor_mode
        )

        if keep_rate is not None and not 0 < keep_rate <= 1:
            raise ValueError(f"keep_rate must be in (0, 1], got {keep_rate}")

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_legacy(
        cls,
        right_context_length: int,
        compression_filter: str = "hanning",
        hanning_window_size: int = 5,
        feature_dim: int = 512,
        weight_predictor_mode: str = "diff5",
    ) -> "DynamicCompression":
        """Build from the legacy ``compression.filter`` enum (bit-exact path)."""
        if compression_filter == "none":
            return cls(
                right_context_length=right_context_length,
                feature_dim=feature_dim,
                passthrough=True,
            )
        if compression_filter not in _LEGACY_MAP:
            raise ValueError(f"Unknown compression_filter '{compression_filter}'")
        det_type, mix_type = _LEGACY_MAP[compression_filter]
        return cls(
            right_context_length=right_context_length,
            detector={"type": det_type},
            mixer={"type": mix_type, "mode": weight_predictor_mode},
            feature_dim=feature_dim,
            hanning_window_size=hanning_window_size,
            weight_predictor_mode=weight_predictor_mode,
        )

    @classmethod
    def from_config(
        cls,
        right_context_length: int,
        compression_cfg: Dict[str, Any],
        feature_dim: int = 512,
    ) -> "DynamicCompression":
        """Build from a ``model.compression`` config dict.

        New-style keys (``detector``/``mixer``/``keep_rate``) take precedence;
        falls back to the legacy ``filter`` enum when absent.
        """
        comp = dict(compression_cfg or {})
        if "detector" in comp:
            return cls(
                right_context_length=right_context_length,
                detector=comp.get("detector"),
                mixer=comp.get("mixer", {"type": "hard"}),
                feature_dim=feature_dim,
                keep_rate=comp.get("keep_rate"),
                hanning_window_size=comp.get("hanning_window_size", 5),
                weight_predictor_mode=comp.get("weight_predictor_mode", "diff5"),
            )
        return cls.from_legacy(
            right_context_length=right_context_length,
            compression_filter=comp.get("filter", "hanning"),
            hanning_window_size=comp.get("hanning_window_size", 5),
            feature_dim=feature_dim,
            weight_predictor_mode=comp.get("weight_predictor_mode", "diff5"),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Compress input sequences by frame selection / mixing.

        Args:
            input: ``(B, T, F)``.
            lengths: Valid frame counts (including right context), ``(B,)``.

        Returns:
            (compressed, new_lengths).
        """
        if self.passthrough:
            return input, lengths

        if input.dim() != 3:
            raise ValueError(f"Input must be 3D (B, T, F), got {input.dim()}D")
        if lengths.dim() != 1:
            raise ValueError(f"Lengths must be 1D (B,), got {lengths.dim()}D")
        if input.shape[0] != lengths.shape[0]:
            raise ValueError(
                f"Batch size mismatch: input={input.shape[0]}, lengths={lengths.shape[0]}"
            )

        B = input.shape[0]
        device = input.device
        compressed_features = []

        # Inter-frame absolute difference (B, T-1)
        diff = torch.abs(input[:, 1:, :] - input[:, :-1, :]).sum(dim=-1)

        for b in range(B):
            total_len = lengths[b].item()
            if total_len <= 1:
                compressed_features.append(input[b, :total_len, :])
                continue

            main_len = total_len - self.right_context_length
            if main_len <= 0:
                compressed_features.append(input[b, :total_len, :])
                continue

            main_part = input[b, :main_len, :]
            right_context_part = input[b, main_len:total_len, :]

            diff_b = diff[b, : main_len - 1]

            with torch.no_grad():
                diff_np_raw = diff_b.cpu().numpy()
                indices, scores = self.detector.detect(
                    main_part, diff_b, diff_np_raw, main_len
                )
                indices = indices[indices < main_len]
                if len(indices) == 0:
                    indices = torch.zeros(1, dtype=torch.long, device=device)
                elif not self.detector.fixed_rate and self.keep_rate is not None:
                    # Universal equal-rate mechanism: top-k by detector score.
                    k = max(1, round(main_len * self.keep_rate))
                    k = min(k, len(indices))
                    _, top = torch.topk(scores, k=k)
                    indices, _ = torch.sort(indices[top])

            compressed_main = self.mixer.mix(main_part, indices, diff_b, main_len)

            compressed_features.append(
                torch.cat([compressed_main, right_context_part], dim=0)
            )

        compressed_lengths = torch.tensor(
            [feat.shape[0] for feat in compressed_features],
            device=device,
            dtype=torch.int32,
        )
        compressed_output = pad_sequence(
            compressed_features, batch_first=True, padding_value=0.0
        )
        return compressed_output, compressed_lengths
