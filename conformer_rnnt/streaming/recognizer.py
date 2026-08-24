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
        self.preview_log = []   # (u, drafts) per emitted token; 统计用, 不回写
        self._preview_lookahead = None
        if self.previewer is not None:
            self.decoder.on_emit = self._preview_on_emit
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
        """Encode all chunks into the decoder; preview is sampled at every token
        emission via ``decoder.on_emit``, never fed back. Returns final ids."""
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
            # 窗口右文(lookahead)帧: source >= end, 已编码未提交。预览在 emit 时
            # 需要它们给出前沿之后的未来帧, 不引入额外音频等待。
            self._preview_lookahead = self.encoder.slice_since(out, ids, end)
            self.decoder.push(committed)
        return self.decoder.partial()

    def _preview_on_emit(self) -> None:
        # Called by decoder after each token emission (predictor advanced to u+1).
        # Read-only: capture drafts, never feed back into decoder state.
        look = self._preview_lookahead
        enc = self.decoder._enc()
        if look is not None and look.size(1) > 0:
            enc = torch.cat([enc, look], dim=1)
        drafts = self.previewer.lookahead(self.decoder, enc)
        if drafts:
            self.preview_log.append((len(self.decoder.out), drafts))