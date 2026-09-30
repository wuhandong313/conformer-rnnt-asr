"""Shared helpers for Phase A0 diagnostic analysis (research-plan v2.0 §6).

These scripts run on the training server (dataset + checkpoint required).
Layer activation capture replicates ``ConformerEncoder.forward`` exactly,
so statistics are computed on the same tensors the model itself sees.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks

from common import checkpoint_dir, load_config, setup_logging
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule


def load_model(
    config_path: str,
    ckpt_path: str | None = None,
    cpu: bool = False,
) -> Tuple[LibriSpeechRNNTModule, Dict[str, Any], torch.device, str]:
    """Build model from config, load checkpoint, return (model, config, device, ckpt_path)."""
    config = load_config(config_path)
    setup_logging(config)
    model = LibriSpeechRNNTModule.from_config(config)

    if ckpt_path is None:
        ckpt_dir = checkpoint_dir(config)
        files = glob.glob(os.path.join(ckpt_dir, "val-*.ckpt"))
        if not files:
            raise SystemExit(f"No val-*.ckpt in {ckpt_dir}; pass --checkpoint")
        ckpt_path = max(files, key=os.path.getmtime)
        logging.info("Auto-selected checkpoint: %s", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    accel = config.get("trainer", {}).get("accelerator", "cuda")
    use_cuda = not cpu and accel in ("gpu", "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device).eval()
    return model, config, device, ckpt_path


def encoder_frame_period(config: Dict[str, Any]) -> float:
    """Seconds per encoder frame (after time reduction)."""
    data = config["data"]
    stride = config["model"]["time_reduction_stride"]
    return data["hop_length"] * stride / data["sample_rate"]


def right_context_frames(config: Dict[str, Any]) -> int:
    """Right-context frames at encoder resolution (after time reduction)."""
    model_cfg = config["model"]
    return model_cfg["right_context_length"] // model_cfg["time_reduction_stride"]


@torch.no_grad()
def capture_layer_activations(
    model: LibriSpeechRNNTModule,
    features: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Run the encoder frontend, capturing every Conformer sub-layer output.

    Mirrors ``ConformerEncoder.forward`` step by step.

    Returns:
        acts: layer name → (B, T, F) tensor. Keys: ``time_reduction``,
            ``block1.L1..LN``, ``block1_out``, ``compression_out``,
            ``block2.L1..LM``.
        len_block1: valid lengths entering compression (incl. right context).
        len_compressed: valid lengths after compression (incl. right context).
    """
    tr = model.model.transcriber
    acts: Dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(_module, _args, output):
            acts[name] = output.detach()

        return hook

    handles = []
    for i, layer in enumerate(tr.conformer1.conformer_layers):
        handles.append(layer.register_forward_hook(make_hook(f"block1.L{i+1}")))
    for i, layer in enumerate(tr.conformer2.conformer_layers):
        handles.append(layer.register_forward_hook(make_hook(f"block2.L{i+1}")))

    rc = tr.right_context_length // tr.time_reduction_stride
    x, x_len = tr.time_reduction(tr.input_linear(features), lengths)
    acts["time_reduction"] = x.detach()
    x, len_b1 = tr.conformer1(x, x_len + rc)
    acts["block1_out"] = x.detach()
    x_c, len_c = tr.compression(x, len_b1)
    acts["compression_out"] = x_c.detach()
    tr.conformer2(x_c, len_c)

    for h in handles:
        h.remove()
    return acts, len_b1.detach(), len_c.detach()


def diff_signal(frames: torch.Tensor) -> np.ndarray:
    """Per-frame L1 novelty signal, identical to DynamicCompression's diff.

    frames: (T, F) → (T-1,) numpy array, ``diff[i] = |frame[i+1] - frame[i]|``.
    """
    return torch.abs(frames[1:] - frames[:-1]).sum(dim=-1).cpu().numpy()


def detect_peak_frames(diff: np.ndarray) -> np.ndarray:
    """Naive peak detector identical to compression.py: peak at diff[k] → frame k+1."""
    peaks, _ = find_peaks(diff)
    return peaks + 1


def boundary_f1(
    pred_frames: np.ndarray,
    ref_frames: np.ndarray,
    tolerance: int,
) -> Tuple[float, float, float]:
    """Greedy one-to-one matching within ±tolerance frames → (F1, precision, recall)."""
    from bisect import bisect_left, bisect_right

    pred = sorted(int(p) for p in pred_frames)
    ref = sorted(int(r) for r in ref_frames)
    used = [False] * len(ref)
    tp = 0
    for p in pred:
        best, best_d = -1, tolerance + 1
        for j in range(bisect_left(ref, p - tolerance), bisect_right(ref, p + tolerance)):
            if not used[j] and abs(ref[j] - p) < best_d:
                best, best_d = j, abs(ref[j] - p)
        if best >= 0:
            used[best] = True
            tp += 1
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall
