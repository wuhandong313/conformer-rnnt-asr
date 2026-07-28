"""Shared utilities for Conformer-RNNT ASR.

This module provides common components shared across the project:
- Batch named tuple for data organization
- GlobalStatsNormalization for feature standardization
- Spectrogram transform and piecewise-linear-log
- Batching by token count
- Post-processing of beam search hypotheses
"""

import json
import math
from collections import namedtuple
from typing import List, Tuple

import sentencepiece as spm
import torch
import torchaudio
from torchaudio.models import Hypothesis


MODEL_TYPE_LIBRISPEECH = "librispeech"
MODEL_TYPE_TEDLIUM3 = "tedlium3"
MODEL_TYPE_MUSTC = "mustc"


DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)

spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    n_mels=80,
    hop_length=80,
)

Batch = namedtuple("Batch", ["features", "feature_lengths", "targets", "target_lengths"])


def piecewise_linear_log(x: torch.Tensor) -> torch.Tensor:
    """Apply piecewise linear-log transform for feature compression."""
    x = x * GAIN
    x = torch.where(x > math.e, torch.log(x), x / math.e)
    return x


def batch_by_token_count(
    idx_target_lengths: List[Tuple[int, int]], token_limit: int
) -> List[List[int]]:
    """Group samples into batches based on cumulative token count.

    Args:
        idx_target_lengths: List of (sample_index, target_length) tuples,
            sorted by target length descending.
        token_limit: Maximum total tokens per batch.

    Returns:
        List of batches, each a list of sample indices.
    """
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > token_limit:
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


def post_process_hypos(
    hypos: List[Hypothesis], sp_model: spm.SentencePieceProcessor
) -> List[Tuple[str, float, List[int], List[int]]]:
    """Post-process beam search hypotheses: remove special tokens, decode text.

    Args:
        hypos: List of Hypothesis objects from RNNT beam search.
        sp_model: SentencePiece model for decoding.

    Returns:
        List of (decoded_text, score, token_ids) tuples.
    """
    tokens_idx = 0
    score_idx = 3
    post_process_remove_list = [
        sp_model.unk_id(),
        sp_model.eos_id(),
        sp_model.pad_id(),
    ]
    filtered_hypo_tokens = [
        [
            token_index
            for token_index in h[tokens_idx][1:]
            if token_index not in post_process_remove_list
        ]
        for h in hypos
    ]
    hypos_str = [sp_model.decode(s) for s in filtered_hypo_tokens]
    hypos_ids = [h[tokens_idx][1:] for h in hypos]
    hypos_score = [[math.exp(h[score_idx])] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ids))
    return nbest_batch


class FunctionalModule(torch.nn.Module):
    """Wrap a function as a torch.nn.Module for use in nn.Sequential."""

    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class GlobalStatsNormalization(torch.nn.Module):
    """Global mean-variance normalization using pre-computed statistics.

    Args:
        global_stats_path: Path to JSON file with 'mean' and 'invstddev' keys.
    """

    def __init__(self, global_stats_path: str):
        super().__init__()

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.mean = torch.tensor(blob["mean"])
        self.invstddev = torch.tensor(blob["invstddev"])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self.mean) * self.invstddev


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer.
        warmup_updates: Number of steps to linearly increase LR over.
    """

    def __init__(self, optimizer, warmup_updates, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        return [
            (min(1.0, self._step_count / self.warmup_updates)) * base_lr
            for base_lr in self.base_lrs
        ]
