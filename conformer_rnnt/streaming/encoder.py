"""Chunked encoder: run the existing ConformerEncoder stage-by-stage on a
window, returning each output frame's source reduced index.

Running stages manually (vs calling ``transcriber`` wholesale) lets us know the
source index of every output frame -- necessary once compression drops frames
(task 4). For ``filter == "none"`` compression is identity: output frame *i* is
reduced frame *i*.
"""
from typing import List, Tuple

import torch
from torch import Tensor


class ChunkedEncoder:
    def __init__(self, *, model, chunk_frames: int, left_context_frames: int,
                 right_context_frames: int, stride: int) -> None:
        self.model = model
        self.enc = model.transcriber
        self.comp = self.enc.compression
        self.rc = int(self.enc.right_context_length)
        self.stride = int(stride)
        self.chunk_frames = int(chunk_frames)
        self.left_context_frames = int(left_context_frames)
        self.right_context_frames = int(right_context_frames)
        if self.right_context_frames != self.rc:
            raise ValueError(
                f"right_context_frames={self.right_context_frames} must equal "
                f"model right_context_length={self.rc} (encoder rc bookkeeping)"
            )

    def encode_window(self, window: Tensor, content_len: int
                      ) -> Tuple[Tensor, List[int]]:
        """Encode a (1, L+C+R, F) window; return (enc_out(1, T', D), source_ids)."""
        self.enc.eval()
        x = self.enc.input_linear(window)                       # (1, W, tr_dim)
        x, xlen = self.enc.time_reduction(
            x, torch.tensor([content_len], dtype=torch.int32))
        rc_frames = self.rc // self.stride                       # = 1
        x, _ = self.enc.conformer1(x, xlen + rc_frames)
        total_reduced = x.size(1)
        x, xlen, ids = replicate_compression(self.comp, x, total_reduced)
        x, _ = self.enc.conformer2(x, xlen)
        out = self.enc.output_norm(self.enc.output_linear(x))
        return out, list(ids)

    def committed_frames(self, enc_out: Tensor, source_ids: List[int],
                         chunk_local_start: int, chunk_local_end: int) -> Tensor:
        """Select output frames whose source reduced index is in
        ``[chunk_local_start, chunk_local_end)``. Returns (1, k, D)."""
        if not source_ids:
            return enc_out.new_zeros(1, 0, enc_out.size(-1))
        idx = [i for i, s in enumerate(source_ids)
               if chunk_local_start <= s < chunk_local_end]
        return enc_out[:, idx, :] if idx else enc_out.new_zeros(1, 0, enc_out.size(-1))


def replicate_compression(comp, x: Tensor, total_reduced: int
                          ) -> Tuple[Tensor, Tensor, List[int]]:
    """Single-sequence replication of DynamicCompression.forward, delegating to
    the module's own private methods so peak selection stays bit-identical to
    the trained path. Returns (out, lengths, source_ids)."""
    dev = x.device
    if comp.compression_filter == "none":
        return x, torch.tensor([x.size(1)], dtype=torch.int32), list(range(total_reduced))
    main_len = total_reduced - comp.right_context_length   # mirrors module bug semantics
    if total_reduced <= 1 or main_len <= 0:
        return x, torch.tensor([x.size(1)], dtype=torch.int32), list(range(total_reduced))
    main = x[:, :main_len]
    rc = x[:, main_len:]
    diff = torch.abs(x[:, 1:main_len, :] - x[:, : main_len - 1, :]).sum(dim=-1)
    with torch.no_grad():
        merged = comp._detect_peaks_valleys(diff.squeeze(0).cpu().numpy())
    merged = torch.as_tensor(merged, dtype=torch.long, device=dev)
    merged = merged[merged < main_len]
    if len(merged) == 0:
        merged = torch.tensor([0], dtype=torch.long, device=dev)
    if comp.compression_filter in ("hanning", "savgol_hanning"):
        compressed_main = comp._hanning_smooth(main, diff, merged, main_len, dev)
    else:
        compressed_main = main.index_select(1, merged)
    out = torch.cat([compressed_main, rc], dim=1)
    ids = list(merged.tolist()) + list(range(main_len, total_reduced))
    return out, torch.tensor([out.size(1)], dtype=torch.int32), ids