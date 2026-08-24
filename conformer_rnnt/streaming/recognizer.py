"""Assemble extractor, chunked encoder, decoder and preview."""
from typing import List, Optional

import torch
from torch import Tensor

from conformer_rnnt.streaming.decoder import StreamingGreedyDecoder
from conformer_rnnt.streaming.encoder import ChunkedEncoder
from conformer_rnnt.streaming.preview import MedusaPreview
from conformer_rnnt.streaming.plan import ChunkPlan


class StreamingRecognizer:
    def __init__(self, *, model, mel: Tensor, chunk_frames: int,
                 left_context_frames: int, right_context_frames: int,
                 stride: int, blank_idx: int, preview: bool = True,
                 preview_heads: Optional[int] = None) -> None:
        self.model = model
        self.plan = ChunkPlan(chunk_frames=chunk_frames,
                              left_context_frames=left_context_frames,
                              right_context_frames=right_context_frames,
                              stride=stride, total_frames=mel.size(0))
        self.encoder = ChunkedEncoder(model=model, chunk_frames=chunk_frames,
                                      left_context_frames=left_context_frames,
                                      right_context_frames=right_context_frames,
                                      stride=stride)
        self.decoder = StreamingGreedyDecoder(model=model, blank_idx=blank_idx)
        self.previewer = MedusaPreview(model=model, max_heads=preview_heads) if preview else None
        self.preview_log = []   # (u, drafts) per displayed lookahead; 统计用, 不回写
        self.rc_reduced = self.encoder.rc // self.encoder.stride
        self.mel = mel.detach()

    def window_for_chunk(self, ci: int) -> Tensor:
        g0, need, left_pad, right_pad = self.plan.window_for_chunk(ci)
        w = torch.zeros(1, need, self.mel.size(1), dtype=self.mel.dtype,
                        device=self.mel.device)
        real = need - left_pad - right_pad
        w[0, left_pad: left_pad + real] = self.mel[g0: g0 + real]
        return w

    def run_offline(self) -> List[int]:
        """Encode all chunks into the decoder; preview (if enabled) is captured
        per chunk but never fed back. Returns final token ids."""
        committed = None
        for ci in range(self.plan.num_chunks()):
            w = self.window_for_chunk(ci)
            content_len = self.plan.content_len()
            out, ids = self.encoder.encode_window(w, content_len)
            start = self.plan.local_content_start(ci)
            c_reduced = self.plan.chunk_content_frames(ci) // self.plan.stride
            end = start + c_reduced
            if ci == self.plan.num_chunks() - 1:
                end += self.rc_reduced  # 末 chunk 补足离线 rc 尾部帧
            committed = self.encoder.committed_frames(out, ids, start, end)
            self.decoder.push(committed)
            self._maybe_preview()
        return self.decoder.partial()

    def _maybe_preview(self) -> None:
        # Read-only: capture drafts for display, never touch decoder state.
        if self.previewer is None:
            return
        drafts = self.previewer.lookahead(self.decoder, self.decoder._enc())
        if drafts:
            self.preview_log.append((len(self.decoder.out), drafts))