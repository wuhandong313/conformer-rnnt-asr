"""Chunk geometry for streaming recognition.

Every chunk ``[s, s+C)`` is encoded inside a fixed window of
``L + C + R`` mel frames ``[max(0, s-L), min(T, s+C+R))``, zero-padded to the
full window length. ``R`` must equal the model's ``right_context_length`` so the
encoder's internal ``rc_frames = rc // stride`` bookkeeping stays consistent
with the offline path. L/C/R must be stride multiples.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkPlan:
    chunk_frames: int
    left_context_frames: int
    right_context_frames: int
    stride: int
    total_frames: int = 0  # content mel frames (0 = unknown, streaming)

    def __post_init__(self):
        for name in ("chunk_frames", "left_context_frames", "right_context_frames"):
            v = getattr(self, name)
            if v < 0 or v % self.stride != 0:
                raise ValueError(
                    f"{name}={v} must be a non-negative multiple of stride={self.stride}"
                )

    def num_chunks(self) -> int:
        t = self.total_frames
        return 0 if t <= 0 else (t + self.chunk_frames - 1) // self.chunk_frames

    def chunk_content_frames(self, chunk_idx: int) -> int:
        t = self.total_frames
        s = chunk_idx * self.chunk_frames
        if t <= 0 or s >= t:
            return 0
        return min(self.chunk_frames, t - s)

    def window_for_chunk(self, chunk_idx: int) -> tuple:
        """(g0, need, left_pad, right_pad); see module docstring."""
        s = chunk_idx * self.chunk_frames
        c = self.chunk_content_frames(chunk_idx)
        need = self.left_context_frames + self.chunk_frames + self.right_context_frames
        g0 = max(0, s - self.left_context_frames)
        left_pad = max(0, self.left_context_frames - s)
        if self.total_frames > 0:
            g1 = min(self.total_frames, s + self.chunk_frames + self.right_context_frames)
            right_pad = (s + self.chunk_frames + self.right_context_frames) - g1
        else:  # streaming: caller supplies full window when enough frames exist
            g1 = s + c + self.right_context_frames
            right_pad = (self.chunk_frames - c) + self.right_context_frames
        return (g0, need, left_pad, right_pad)

    def content_len(self) -> int:
        """Frames counted as 'content' for the encoder (left + full chunk)."""
        return self.left_context_frames + self.chunk_frames

    def local_content_start(self, chunk_idx: int) -> int:
        """Chunk content's window-local reduced start index."""
        g0, _, left_pad, _ = self.window_for_chunk(chunk_idx)
        s = chunk_idx * self.chunk_frames
        return (left_pad + (s - g0)) // self.stride