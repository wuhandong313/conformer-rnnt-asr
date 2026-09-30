# FROZEN pre-refactor reference implementation.
# Verbatim copy of conformer_rnnt/compression.py @ commit e67507d.
# Used by tests/test_parity.py — DO NOT MODIFY.
"""Dynamic inter-frame compression for sequence length reduction.

Provides peak/valley detection on inter-frame difference signals with optional
filtering (FIR lowpass, Savitzky-Golay, Hanning smoothing) and a learnable
weight predictor for adaptive frame blending.

Compression modes:
  - "none":     Bypass compression entirely (identity pass-through).
  - "naive":    Raw inter-frame diff → find_peaks, no filtering.
  - "hanning":  Raw diff → find_peaks → learnable Hanning-weighted blending.
  - "fir":      FIR lowpass → find approximate peaks → refine on raw signal.
  - "savgol":   Savitzky-Golay smoothing → find peaks on smoothed signal.
  - "savgol_hanning": SG smoothing + learnable Hanning-weighted blending.

Weight predictor modes (for hanning / savgol_hanning):
  - "single_frame":  MLP(peak_frame_features) → window weights.
  - "diff5":         MLP(4 differences of 5 surrounding diff values) → window weights.
  - "five_frames":   MLP(concatenated 5-frame features) → window weights.
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from scipy.signal import find_peaks, savgol_filter, firwin, filtfilt


class DynamicCompression(torch.nn.Module):
    r"""Dynamic frame compression via inter-frame difference peak/valley detection.

    For each sample, computes the per-frame absolute difference sum, detects
    peaks and valleys (sudden changes), and selects key frames at those points.
    Right context frames are excluded from compression and passed through unchanged.

    Args:
        right_context_length (int): Number of right-context frames to preserve
            without compression.
        compression_filter (str): Filter mode. One of
            ``"none"``, ``"fir"``, ``"savgol"``, ``"savgol_hanning"``,
            ``"hanning"``, ``"naive"``.
        hanning_window_size (int): Window size for Hanning-weighted smoothing.
            Only used when compression_filter includes ``"hanning"``.
        feature_dim (int): Feature dimension of input frames. Used by the
            weight predictor MLPs.
        weight_predictor_mode (str): Input mode for the weight predictor.
            One of ``"single_frame"``, ``"diff5"``, ``"five_frames"``.
    """

    def __init__(
        self,
        right_context_length: int,
        compression_filter: str = "hanning",
        hanning_window_size: int = 5,
        feature_dim: int = 512,
        weight_predictor_mode: str = "diff5",
    ) -> None:
        super().__init__()
        self.right_context_length = right_context_length
        self.compression_filter = compression_filter
        self.hanning_window_size = hanning_window_size
        self.weight_predictor_mode = weight_predictor_mode

        # Build weight predictor MLP when Hanning smoothing is active
        if compression_filter in ("hanning", "savgol_hanning"):
            if weight_predictor_mode == "diff5":
                # Input: 4 differences computed from 5 surrounding diff values
                self.weight_predictor = torch.nn.Sequential(
                    torch.nn.Linear(4, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, hanning_window_size),
                )
            elif weight_predictor_mode == "five_frames":
                # Input: 5 concatenated frames around each peak/valley
                self.weight_predictor = torch.nn.Sequential(
                    torch.nn.Linear(5 * feature_dim, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, hanning_window_size),
                )
            else:  # "single_frame"
                # Input: single frame at each peak/valley position
                self.weight_predictor = torch.nn.Sequential(
                    torch.nn.Linear(feature_dim, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, hanning_window_size),
                )

        # Cache FIR filter taps to avoid recomputation
        self._fir_taps_cache = {}

    # ------------------------------------------------------------------
    # Peak / valley detection
    # ------------------------------------------------------------------
    def _detect_peaks_valleys(
        self,
        diff_np_raw: np.ndarray,
    ) -> np.ndarray:
        """Apply configured filter and return merged peak+valley indices.

        Returns 0-based frame indices (not diff indices).
        """
        diff_len = len(diff_np_raw)

        if self.compression_filter in ("savgol", "savgol_hanning"):
            # Savitzky-Golay smoothing — use filtered positions directly
            diff_np = diff_np_raw.copy()
            if diff_len > 1:
                wlen = min(9, diff_len)
                if wlen % 2 == 0:
                    wlen -= 1
                if wlen > 3:
                    diff_np = savgol_filter(diff_np, window_length=wlen, polyorder=4)
            peaks, _ = find_peaks(diff_np)
            valleys, _ = find_peaks(-diff_np)
            # Convert diff indices → frame indices:
            #   diff[i] = |frame[i+1] - frame[i]|
            #   peak at diff[k]  →  frame index k+1
            #   valley at diff[k] →  frame index k
            merged = np.unique(np.concatenate([peaks + 1, valleys]))
            return merged

        elif self.compression_filter == "fir":
            # FIR lowpass → coarse peak detection → refine on raw signal
            diff_np = diff_np_raw.copy()
            if diff_len > 1:
                num_taps = min(21, diff_len // 8)
                if num_taps >= 5:
                    if num_taps % 2 == 0:
                        num_taps -= 1
                    cache_key = num_taps
                    if cache_key not in self._fir_taps_cache:
                        nyquist = 0.5 * 200  # frame_rate = 200
                        cutoff_norm = 50 / nyquist
                        self._fir_taps_cache[cache_key] = firwin(
                            num_taps, cutoff_norm, pass_zero="lowpass"
                        )
                    taps = self._fir_taps_cache[cache_key]
                    diff_np = filtfilt(taps, 1.0, diff_np)

            peaks_approx, _ = find_peaks(diff_np)
            valleys_approx, _ = find_peaks(-diff_np)

            # Refine to exact positions on raw signal within ±2 radius
            search_radius = 2
            raw_len = len(diff_np_raw)
            peaks_refined = []
            valleys_refined = []
            for p in peaks_approx:
                lo = max(0, p - search_radius)
                hi = min(raw_len - 1, p + search_radius + 1)
                peaks_refined.append(lo + np.argmax(diff_np_raw[lo:hi]))
            for v in valleys_approx:
                lo = max(0, v - search_radius)
                hi = min(raw_len - 1, v + search_radius + 1)
                valleys_refined.append(lo + np.argmin(diff_np_raw[lo:hi]))

            merged = np.unique(np.concatenate([np.array(peaks_refined) + 1, np.array(valleys_refined)]))
            return merged

        else:  # "hanning", "naive", "none" — raw diff, no filtering
            diff_np = diff_np_raw
            peaks, _ = find_peaks(diff_np)
            valleys, _ = find_peaks(-diff_np)
            merged = np.unique(np.concatenate([peaks + 1, valleys]))
            return merged

    # ------------------------------------------------------------------
    # Hanning-weighted smoothing
    # ------------------------------------------------------------------
    def _hanning_smooth(
        self,
        main_part: torch.Tensor,
        diff_b: torch.Tensor,
        merged_indices: torch.Tensor,
        main_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Apply learnable Hanning-window weighted averaging around each
        detected peak/valley frame."""
        half_win = self.hanning_window_size // 2
        N = len(merged_indices)

        # Build input features for the weight predictor
        if self.weight_predictor_mode == "diff5":
            indices_5 = (
                merged_indices.unsqueeze(-1)
                + torch.tensor([-2, -1, 0, 1, 2], device=device)
            ).clamp(0, len(diff_b) - 1)
            five_diff_vals = diff_b[indices_5]  # (N, 5)
            peak_features = torch.abs(
                five_diff_vals[:, 1:] - five_diff_vals[:, :-1]
            )  # (N, 4)
        elif self.weight_predictor_mode == "five_frames":
            indices_5 = (
                merged_indices.unsqueeze(-1)
                + torch.tensor([-2, -1, 0, 1, 2], device=device)
            ).clamp(0, main_len - 1)
            peak_features = main_part[indices_5].reshape(N, -1)  # (N, 5*F)
        else:  # "single_frame"
            peak_features = main_part[merged_indices]  # (N, F)

        all_weights = torch.softmax(
            self.weight_predictor(peak_features), dim=-1
        )  # (N, window_size)

        smoothed_frames = []
        for i, idx in enumerate(merged_indices):
            start = max(0, idx - half_win)
            end = min(main_len, idx + half_win + 1)
            w_start = start - (idx - half_win)
            w_end = self.hanning_window_size - ((idx + half_win + 1) - end)
            weights = all_weights[i, w_start:w_end]
            weights = weights / weights.sum()
            smoothed = (main_part[start:end] * weights.unsqueeze(-1)).sum(dim=0)
            smoothed_frames.append(smoothed)

        return torch.stack(smoothed_frames)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, input: Tensor, lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""Compress input sequences by selecting key frames at inter-frame
        change points.

        B: batch size;
        T: max input sequence length in batch;
        F: feature dimension.

        Args:
            input (Tensor): Input sequences, shape ``(B, T, F)``.
            lengths (Tensor): Valid frame counts (including right context),
                shape ``(B,)``.

        Returns:
            (Tensor, Tensor):
                - Compressed sequences, shape ``(B, T_compressed, F)``.
                - New valid lengths, shape ``(B,)``.
        """
        if input.dim() != 3:
            raise ValueError(f"Input must be 3D (B, T, F), got {input.dim()}D")
        if lengths.dim() != 1:
            raise ValueError(f"Lengths must be 1D (B,), got {lengths.dim()}D")
        if input.shape[0] != lengths.shape[0]:
            raise ValueError(
                f"Batch size mismatch: input={input.shape[0]}, lengths={lengths.shape[0]}"
            )

        B, T, F = input.shape
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
                merged_np = self._detect_peaks_valleys(diff_np_raw)

            merged_indices = torch.as_tensor(merged_np, device=device).long()
            merged_indices = merged_indices[merged_indices < main_len]

            if len(merged_indices) == 0:
                merged_indices = torch.tensor([0], device=device)

            # Apply smoothing or direct selection
            if self.compression_filter in ("hanning", "savgol_hanning"):
                compressed_main = self._hanning_smooth(
                    main_part, diff_b, merged_indices, main_len, device
                )
            else:
                compressed_main = main_part.index_select(0, merged_indices)

            compressed_feat = torch.cat([compressed_main, right_context_part], dim=0)
            compressed_features.append(compressed_feat)

        # Pad to uniform length
        compressed_lengths = torch.tensor(
            [feat.shape[0] for feat in compressed_features],
            device=device,
            dtype=torch.int32,
        )
        compressed_output = pad_sequence(
            compressed_features, batch_first=True, padding_value=0.0
        )

        return compressed_output, compressed_lengths
