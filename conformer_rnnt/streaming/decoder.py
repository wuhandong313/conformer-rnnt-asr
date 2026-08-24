"""Event-driven greedy RNN-T decoder: push encoder frames as they arrive,
produce tokens identical to the offline greedy decoder on the same frames.
"""
from typing import List

import torch
from torch import Tensor

from conformer_rnnt.spec_decode import (
    _advance_predictor, _init_decode, _join_argmax, _make_tok,
)


class StreamingGreedyDecoder:
    def __init__(self, *, model, blank_idx: int, max_steps: int = 100000) -> None:
        self.model = model
        self.blank_idx = blank_idx
        self.max_steps = max_steps
        self.dev = next(model.parameters()).device
        self.one = torch.tensor([1], dtype=torch.int32)
        self._reset()

    def _reset(self) -> None:
        self.out: List[int] = []
        self.t = 0
        self.steps = 0
        self._enc_frames: List[Tensor] = []
        self.g_last, self.state, self.one = _init_decode(
            self.model, self.blank_idx, self.dev)

    def push(self, frames: Tensor) -> List[int]:
        """Append (1, k, D) encoder frames and advance decode as far as possible."""
        self._enc_frames.append(frames.detach())
        self._advance()
        return list(self.out)

    def _enc(self) -> Tensor:
        if not self._enc_frames:
            return torch.zeros(1, 0, self.model.joiner.linear.in_features,
                               device=self.dev)
        return torch.cat(self._enc_frames, dim=1)

    def _advance(self) -> None:
        enc = self._enc()
        while self.t < enc.size(1) and self.steps < self.max_steps:
            pred = _join_argmax(self.model, enc, self.t, self.g_last, self.one)
            if pred == self.blank_idx:
                self.t += 1
            else:
                self.out.append(pred)
                self.g_last, self.state = _advance_predictor(
                    self.model, _make_tok(pred, self.dev), self.state, self.one)
            self.steps += 1

    def partial(self) -> List[int]:
        return list(self.out)