"""Shared utilities for Conformer-RNNT ASR.

This module provides common components shared across the project:
- Batch named tuple for data organization
- GlobalStatsNormalization for feature standardization
- Piecewise-linear-log feature compression
- Batching by token count
- Post-processing of beam search hypotheses
- Config loading / checkpoint tagging (shared by train.py and eval.py)
"""

import json
import logging
import math
import os
from collections import namedtuple
from typing import Any, Dict, List, Tuple

import sentencepiece as spm
import torch
import yaml
from torchaudio.models import Hypothesis


DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)

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


def strip_special_tokens(ids: List[int], sp_model: spm.SentencePieceProcessor) -> List[int]:
    """Remove unk/eos/pad tokens from a decoded id list."""
    remove = {sp_model.unk_id(), sp_model.eos_id(), sp_model.pad_id()}
    return [t for t in ids if t not in remove]


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
    filtered_hypo_tokens = [
        strip_special_tokens(h[tokens_idx][1:], sp_model) for h in hypos
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


# ═══════════════════════════════════════════════════════════════════════════
# Config / checkpoint helpers (shared by train.py and eval.py)
# ═══════════════════════════════════════════════════════════════════════════


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file '{path}' is empty.")
    return cfg


def setup_logging(config: Dict[str, Any]) -> None:
    debug = config.get("debug", False)
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def checkpoint_dir(config: Dict[str, Any]) -> str:
    exp_dir = config.get("paths", {}).get("exp_dir", "./exp")
    return os.path.join(exp_dir, "checkpoints")


def model_ckpt_tag(config: Dict[str, Any]) -> str:
    """从训练实际使用的 config 生成 checkpoint 文件名标记。

    真实训练参数标记 —— 压缩模式 + Medusa 头数 + 冻结状态 + 输入模式。
    与 from_config 使用同一个 config, 保证文件名反映模型真实训练模式。

    例子:
        naive + 1 head + 联合训练  -> "naive-h1-free-sl1"
        naive + 2 head + 冻结基底  -> "naive-h2-frz-sl1"
        无 Medusa (base 模型)      -> "naive-base"
    """
    model_cfg = config.get("model", {})
    comp = model_cfg.get("compression", {})
    medusa = model_cfg.get("medusa", {})

    filt = comp.get("filter", "hanning")
    heads = int(medusa.get("num_heads", 0))
    freeze = "frz" if medusa.get("freeze_base", False) else "free"
    input_mode = medusa.get("input_mode", "stream_local")
    mode_tag = {
        "stream_local": "sl",
        "predictor_only": "pred",
        "current_joint": "joint",
    }.get(input_mode, str(input_mode).replace("_", "-"))
    lookahead = medusa.get("lookahead_frames")
    lookahead_tag = f"{mode_tag}{lookahead}" if lookahead is not None else mode_tag
    if medusa.get("draft_after_emit", False):
        lookahead_tag += "-post"
    if medusa.get("chain_states", False):
        lookahead_tag += "-chain"
    # iter-8 标记: 过去帧锚定; iter-9: sim/dl 通道已移除 (历史 ckpt 名里的
    # "-sim"/"-dl" 后缀不再生成)
    past = medusa.get("past_frames", 0)
    if past:
        lookahead_tag += f"-p{past}"
    # iter-9 标记: 配置驱动的自由后缀 (如 "disp" → "-disp"), 区分参数相同
    # 但价值模型不同的 checkpoint 文件名 — 避免同名 ckpt 触发 monitor 缓存串档
    suffix = medusa.get("tag_suffix")
    if suffix:
        lookahead_tag += f"-{suffix}"

    # 有 Medusa heads 才区分冻结状态, 无 heads 时省略 (避免无意义后缀)
    if heads > 0:
        return f"{filt}-h{heads}-{freeze}-{lookahead_tag}"
    return f"{filt}-base"
