"""Incremental feature extraction matching the offline pipeline frame-for-frame.

Offline: MelSpectrogram(sr, n_fft, n_mels, hop) → piecewise_linear_log → global
stats → (rc pad at window level, not here). With ``center=True`` a mel frame *f*
centers at ``f*hop`` over ``[f*hop - n_fft//2, f*hop + n_fft//2)``, so it is fully
determined once ``len(pcm) >= f*hop + n_fft//2``. ``push`` emits these frames as
audio arrives; offline additionally emits trailing frames whose right edge is
reflected from the buffer end, so ``flush`` reproduces that count exactly.
"""
import torch
import torchaudio

from common import GlobalStatsNormalization, piecewise_linear_log


class StreamingFeatureExtractor:
    def __init__(self, *, sample_rate: int, n_fft: int, n_mels: int,
                 hop_length: int, global_stats_path: str) -> None:
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels,
            hop_length=hop_length,
        )
        self.hop = hop_length
        self.n_fft = n_fft
        self.norm = GlobalStatsNormalization(global_stats_path)
        self._pcm = torch.tensor([], dtype=torch.float32)
        self._frames: list = []
        self._flushed = False

    def push(self, pcm: torch.Tensor) -> None:
        if self._flushed:
            raise RuntimeError("push after flush")
        if pcm.dim() == 2:
            pcm = pcm.squeeze(0)
        self._pcm = torch.cat([self._pcm, pcm.float()])
        self._emit_ready()

    def _emit_ready(self) -> None:
        # frame f needs samples up to f*hop + n_fft//2
        f = len(self._frames)
        mel = self.mel(self._pcm)                       # (n_mels, n_frames)
        logmel = self.norm(piecewise_linear_log(mel).transpose(0, 1))  # (n_frames, n_mels)
        while f < logmel.size(0) and self._pcm.size(0) >= f * self.hop + self.n_fft // 2:
            self._frames.append(logmel[f])
            f += 1

    def flush(self) -> None:
        """Emit the trailing reflect-padded frames, matching offline count."""
        if self._flushed:
            return
        self._flushed = True
        mel = self.mel(self._pcm)                       # (n_mels, n_frames)
        logmel = self.norm(piecewise_linear_log(mel).transpose(0, 1))
        for f in range(len(self._frames), logmel.size(0)):
            self._frames.append(logmel[f])

    def compute_all(self, pcm: torch.Tensor) -> torch.Tensor:
        self._pcm = torch.tensor([], dtype=torch.float32)
        self._frames = []
        self._flushed = False
        self.push(pcm)
        self.flush()
        return self.frames()

    def frames(self) -> torch.Tensor:
        if not self._frames:
            return torch.zeros(0, int(self.norm.mean.numel()))
        return torch.stack(self._frames)