"""音素 CE / ArtV 对齐：按 DataLoader 的裁剪时间切 MFA 标注，再拉到探针帧长 T。

探针输出 ``[B, n_src, T, K]`` 的长度 T 是对齐标准；目标来自
``{utt}.artv.npy`` + 区间时间戳（``phone_times.npy`` 或 TextGrid）。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhonemeCELoss(nn.Module):
    """帧级音素 CE。全是 ignore_index 时返回 0，避免 nan。"""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, n_src, T, K]
            targets: [B, n_src, T] long
        """
        if not targets.ne(self.ignore_index).any():
            return logits.new_zeros(())
        b, n_src, t, k = logits.shape
        return F.cross_entropy(
            logits.reshape(b * n_src * t, k),
            targets.reshape(b * n_src * t),
            ignore_index=self.ignore_index,
        )


def apply_perm(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """按 PIT 排列重排源维。``x``: [B, n_src, ...]；``perm``: [B, n_src]。"""
    extra = x.ndim - 2
    idx = perm.view(*perm.shape, *([1] * extra)).expand_as(x)
    return torch.gather(x, 1, idx)


def apply_perm_to_logits(logits: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return apply_perm(logits, perm)


def combine_signal_phoneme_loss(
    signal_loss: torch.Tensor,
    phoneme_loss: torch.Tensor,
    lambda_phoneme: float,
) -> torch.Tensor:
    """Total = λ * Loss_phoneme + (1-λ) * Loss_Signal。"""
    lam = float(lambda_phoneme)
    return lam * phoneme_loss + (1.0 - lam) * signal_loss


def combine_signal_ce_artv_loss(
    signal_loss: torch.Tensor,
    ce_loss: torch.Tensor,
    artv_loss: torch.Tensor,
    lambda_phoneme: float,
    artv_weight: float = 1.0,
    ce_weight: float = 1.0,
) -> torch.Tensor:
    """Total = (1-λ)·Signal + λ·(ce_weight·CE + artv_weight·ArtV_MSE)。"""
    phoneme = float(ce_weight) * ce_loss + float(artv_weight) * artv_loss
    return combine_signal_phoneme_loss(signal_loss, phoneme, lambda_phoneme)


def gaussianize_artv_numpy(
    artv: np.ndarray,
    phone_ids: np.ndarray,
    ignore_index: int = -100,
    sigma_scale: float = 0.25,
    sil_ids: tuple[int, ...] = (0, 1),
) -> np.ndarray:
    """分段常数 ArtV → 高斯窗加权叠加。

    每段连续 ``phone_id`` 在区间中心放一维高斯（``sigma = L * sigma_scale``），
    **不按权重归一**，因此中心接近原型、边缘衰减；邻段外溢可叠加。
    ``sil_ids`` / ``ignore_index`` 不参与高斯，保持 0。
    """
    t_len, dim = artv.shape
    acc = np.zeros((t_len, dim), dtype=np.float64)
    ids = np.asarray(phone_ids).reshape(-1)
    i = 0
    while i < t_len:
        pid = int(ids[i])
        if pid == ignore_index or pid in sil_ids:
            i += 1
            continue
        start = i
        while i < t_len and int(ids[i]) == pid:
            i += 1
        end = i
        length = end - start
        center = (start + end - 1) / 2.0
        sigma = max(float(length) * float(sigma_scale), 0.5)
        lo = max(0, int(np.floor(center - 3.0 * sigma)))
        hi = min(t_len, int(np.ceil(center + 3.0 * sigma)) + 1)
        xs = np.arange(lo, hi, dtype=np.float64)
        weights = np.exp(-0.5 * ((xs - center) / sigma) ** 2)
        vec = artv[start].astype(np.float64)
        acc[lo:hi] += weights[:, None] * vec[None, :]
    return acc.astype(np.float32)


def gaussianize_artv_batch(
    artv: torch.Tensor,
    phone_ids: torch.Tensor,
    ignore_index: int = -100,
    sigma_scale: float = 0.25,
) -> torch.Tensor:
    """``artv`` ``[B, n_src, T, D]``，``phone_ids`` ``[B, n_src, T]``。"""
    b, n_src, t_len, dim = artv.shape
    out = torch.zeros_like(artv)
    artv_np = artv.detach().cpu().numpy()
    ids_np = phone_ids.detach().cpu().numpy()
    for bi in range(b):
        for si in range(n_src):
            out[bi, si] = torch.from_numpy(
                gaussianize_artv_numpy(
                    artv_np[bi, si],
                    ids_np[bi, si],
                    ignore_index=ignore_index,
                    sigma_scale=sigma_scale,
                )
            )
    return out.to(device=artv.device, dtype=artv.dtype)


def align_artv_to_len(artv: torch.Tensor, t_out: int) -> torch.Tensor:
    """线性插值时间维 ``[B, n_src, T, D] → [B, n_src, t_out, D]``。"""
    if artv.size(2) == t_out:
        return artv
    b, n_src, _t, dim = artv.shape
    z = artv.permute(0, 1, 3, 2).reshape(b * n_src, dim, _t)
    z = F.interpolate(z, size=t_out, mode="linear", align_corners=False)
    return z.reshape(b, n_src, dim, t_out).permute(0, 1, 3, 2)


def align_ids_to_len(phone_ids: torch.Tensor, t_out: int) -> torch.Tensor:
    """最近邻插值 ID ``[B, n_src, T] → [B, n_src, t_out]``。"""
    if phone_ids.size(2) == t_out:
        return phone_ids
    b, n_src, t_len = phone_ids.shape
    z = phone_ids.reshape(b * n_src, 1, t_len).float()
    z = F.interpolate(z, size=t_out, mode="nearest")
    return z.reshape(b, n_src, t_out).long()


def artv_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    phone_ids: torch.Tensor,
    ignore_index: int = -100,
    sil_ids: tuple[int, ...] = (0, 1),
) -> torch.Tensor:
    """仅在有效音素帧上做 MSE。"""
    valid = phone_ids.ne(ignore_index)
    for sid in sil_ids:
        valid = valid & phone_ids.ne(sid)
    if not valid.any():
        return pred.new_zeros(())
    return F.mse_loss(pred[valid], target[valid])


def should_dump_utt(utt_key: str, dump_fraction: float, salt: str = "") -> bool:
    if dump_fraction <= 0:
        return False
    if dump_fraction >= 1:
        return True
    import hashlib

    digest = hashlib.md5((salt + "|" + str(utt_key)).encode("utf-8")).hexdigest()
    h = int(digest, 16) % 10_000
    return h < int(dump_fraction * 10_000)


def append_artv_csv(
    path: Path | str,
    *,
    split: str,
    utt_keys: Sequence[str],
    src: int,
    frame_ids: np.ndarray,
    target: np.ndarray,
    gauss: np.ndarray,
    pred: np.ndarray,
    frame_loss: np.ndarray,
    dump_fraction: float = 0.1,
    epoch: int | None = None,
) -> None:
    """抽中的 utterance 按帧追加 CSV。向量展平为 ``target_0..`` 列。

    ``epoch`` 为训练轮次（val dump 用 ``current_epoch``；最终测试用 -1）。
    若目标文件已是旧表头（无 epoch 列），则按旧格式追加，避免把已有文件写乱。
    """
    import csv

    path = Path(path)
    dim = int(target.shape[-1])
    epoch_val = -1 if epoch is None else int(epoch)
    tail = [
        "utt",
        "src",
        "frame",
        "loss",
        "phone_id",
        *[f"target_{i}" for i in range(dim)],
        *[f"gauss_{i}" for i in range(dim)],
        *[f"pred_{i}" for i in range(dim)],
    ]
    header_with_epoch = ["split", "epoch", *tail]

    path.parent.mkdir(parents=True, exist_ok=True)
    write_epoch = True
    new_file = not path.is_file() or path.stat().st_size == 0
    if not new_file:
        with path.open("r", encoding="utf-8", newline="") as rf:
            first = rf.readline()
        cols = [c.strip() for c in first.split(",")]
        write_epoch = "epoch" in cols

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header_with_epoch)
        n = min(len(frame_ids), len(target), len(gauss), len(pred), len(frame_loss))
        utt = utt_keys[src] if src < len(utt_keys) else ",".join(utt_keys)
        if not should_dump_utt(str(utt), dump_fraction):
            return
        for t in range(n):
            body = [
                utt,
                src,
                t,
                float(frame_loss[t]),
                int(frame_ids[t]),
                *[float(x) for x in target[t]],
                *[float(x) for x in gauss[t]],
                *[float(x) for x in pred[t]],
            ]
            if write_epoch:
                writer.writerow([split, epoch_val, *body])
            else:
                writer.writerow([split, *body])


# ---------------------------------------------------------------------------
# 裁剪时间 → 切 ArtV / phone_ids → 对齐到探针 T
# ---------------------------------------------------------------------------

_TG_HEAD = re.compile(
    rf'name\s*=\s*"phones"\s*'
    r"xmin\s*=\s*([0-9.]+)\s*"
    r"xmax\s*=\s*([0-9.]+)\s*"
    r"intervals:\s*size\s*=\s*(\d+)\s*",
    re.I,
)
_TG_ITEM = re.compile(
    r"intervals\s*\[\d+\]\s*:\s*"
    r"xmin\s*=\s*([0-9.]+)\s*"
    r"xmax\s*=\s*([0-9.]+)\s*"
    r'text\s*=\s*"(.*?)"',
    re.S,
)


def resolve_phoneme_root(config: dict) -> Path:
    """``phoneme_loss.phoneme_root``，相对 ``data_root``；默认 ``libri_phoneme``。"""
    ph = config.get("phoneme_loss") or {}
    data_cfg = (config.get("datamodule") or {}).get("data_config") or {}
    rel = ph.get("phoneme_root") or "libri_phoneme"
    if os.path.isabs(rel):
        return Path(rel)
    root = data_cfg.get("data_root") or ""
    return Path(root) / rel if root else Path(rel)


def split_utt_key(utt_key: str, n_src: int) -> list[str]:
    parts = [p for p in str(utt_key).split(",") if p]
    if len(parts) >= n_src:
        return parts[:n_src]
    return parts + [""] * (n_src - len(parts))


def parse_textgrid_phone_times(text: str) -> np.ndarray:
    """TextGrid phones 层 → ``[N, 2]``（秒）。"""
    m = _TG_HEAD.search(text)
    if not m:
        return np.zeros((0, 2), dtype=np.float32)
    n = int(m.group(3))
    rest = text[m.end() :]
    rows = []
    for i, im in enumerate(_TG_ITEM.finditer(rest)):
        if i >= n:
            break
        rows.append((float(im.group(1)), float(im.group(2))))
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def align_interval_values_to_frames(
    values: np.ndarray,
    times: np.ndarray,
    t0: float,
    t1: float,
    n_frames: int,
    fill,
) -> np.ndarray:
    """把区间序列画到探针帧上：帧中心落在 ``[xmin, xmax)`` 的取该区间值。

    ``values``: [N, ...]；``times``: [N, 2] 秒；返回 ``[n_frames, ...]``。
    长度标准是 ``n_frames``（探针 T），不是 MFA 区间个数。
    """
    tail = values.shape[1:]
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else values.dtype
    out = np.full((n_frames, *tail), fill, dtype=dtype)
    if n_frames <= 0 or t1 <= t0 or times.size == 0 or len(values) == 0:
        return out
    n = min(len(values), len(times))
    values = values[:n]
    times = times[:n]
    dur = float(t1 - t0)
    centers = t0 + (np.arange(n_frames, dtype=np.float64) + 0.5) / n_frames * dur
    xmin = times[:, 0].astype(np.float64)
    xmax = times[:, 1].astype(np.float64)
    hit = (centers[:, None] >= xmin[None, :]) & (centers[:, None] < xmax[None, :])
    has = hit.any(axis=1)
    # 重叠时取最后一个区间
    last = hit.shape[1] - 1 - np.fliplr(hit).argmax(axis=1)
    for i in np.where(has)[0]:
        out[i] = values[int(last[i])]
    return out


class PhonemeAnnotationBank:
    """扫描 ``libri_phoneme/**/{utt}.artv.npy``，按需读 ArtV / ID / 时间戳。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._index: dict[str, Path] | None = None
        self._cache: dict[str, dict | None] = {}

    def _ensure_index(self) -> dict[str, Path]:
        if self._index is None:
            self._index = {}
            if self.root.is_dir():
                for p in self.root.rglob("*.artv.npy"):
                    utt = p.name.removesuffix(".artv.npy")
                    self._index[utt] = p
        return self._index

    def has_annotations(self) -> bool:
        """目录里是否有任何 ``*.artv.npy``（没做音素预处理则为 False）。"""
        return bool(self._ensure_index())

    def has_utt(self, utt_id: str) -> bool:
        return bool(utt_id) and utt_id in self._ensure_index()

    def _times_for(self, artv_path: Path, n: int) -> np.ndarray | None:
        times_path = artv_path.with_name(artv_path.name.removesuffix(".artv.npy") + ".phone_times.npy")
        if times_path.is_file():
            times = np.load(str(times_path))
            if len(times) == n:
                return np.asarray(times, dtype=np.float32)
        tg = artv_path.with_name(artv_path.name.removesuffix(".artv.npy") + ".TextGrid")
        if tg.is_file():
            times = parse_textgrid_phone_times(
                tg.read_text(encoding="utf-8", errors="replace")
            )
            if len(times) == n:
                return times
        return None

    def load_utt(self, utt_id: str) -> dict | None:
        if not utt_id:
            return None
        if utt_id in self._cache:
            return self._cache[utt_id]
        path = self._ensure_index().get(utt_id)
        if path is None or not path.is_file():
            self._cache[utt_id] = None
            return None
        artv = np.load(str(path))
        ids_path = path.with_name(path.name.removesuffix(".artv.npy") + ".phone_ids.npy")
        if ids_path.is_file():
            phone_ids = np.load(str(ids_path))
        else:
            phone_ids = np.zeros((len(artv),), dtype=np.int64)
        if len(phone_ids) != len(artv):
            n = min(len(phone_ids), len(artv))
            phone_ids = phone_ids[:n]
            artv = artv[:n]
        times = self._times_for(path, len(artv))
        if times is None:
            self._cache[utt_id] = None
            return None
        rec = {"artv": np.asarray(artv, dtype=np.float32), "ids": np.asarray(phone_ids, dtype=np.int64), "times": times}
        self._cache[utt_id] = rec
        return rec


def slice_artv_to_probe_frames(
    anno: dict,
    t0: float,
    t1: float,
    n_frames: int,
    ignore_index: int = -100,
) -> tuple[np.ndarray, np.ndarray]:
    """按 [t0, t1) 秒切 ArtV，并对齐到探针 T。返回 ``(ids [T], artv [T, D])``。"""
    ids = align_interval_values_to_frames(
        anno["ids"], anno["times"], t0, t1, n_frames, fill=ignore_index
    )
    zeros = 0.0
    artv = align_interval_values_to_frames(
        anno["artv"], anno["times"], t0, t1, n_frames, fill=zeros
    )
    # ignore 帧的 ArtV 置 0，避免静音区间外的垃圾
    artv[ids == ignore_index] = 0.0
    return ids.astype(np.int64, copy=False), artv.astype(np.float32, copy=False)


def build_frame_targets_from_clips(
    bank: PhonemeAnnotationBank,
    utt_keys: Sequence[str],
    clip_start,
    clip_stop,
    sample_rate: int,
    n_frames: int,
    n_src: int,
    ignore_index: int = -100,
    device=None,
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """一批裁剪窗 → 帧级 phone_ids / ArtV。

    Args:
        utt_keys: 长度 B，每项 ``s1utt,s2utt``
        clip_start / clip_stop: [B] 采样点（左闭右开）
        n_frames: 探针时间维 T（对齐标准）
    Returns:
        phone_ids: [B, n_src, T] long
        artv:      [B, n_src, T, D] float（缺标注则为 0，ids 为 ignore）
    """
    starts = clip_start.detach().cpu().tolist() if torch.is_tensor(clip_start) else list(clip_start)
    stops = clip_stop.detach().cpu().tolist() if torch.is_tensor(clip_stop) else list(clip_stop)
    b = len(utt_keys)
    id_rows = []
    artv_rows = []
    artv_dim = 24
    for i in range(b):
        t0 = float(starts[i]) / float(sample_rate)
        t1 = float(stops[i]) / float(sample_rate)
        utts = split_utt_key(utt_keys[i], n_src)
        src_ids = []
        src_artv = []
        for utt in utts:
            anno = bank.load_utt(utt)
            if anno is None:
                src_ids.append(np.full((n_frames,), ignore_index, dtype=np.int64))
                src_artv.append(np.zeros((n_frames, artv_dim), dtype=np.float32))
                continue
            artv_dim = int(anno["artv"].shape[-1])
            ids, artv = slice_artv_to_probe_frames(
                anno, t0, t1, n_frames, ignore_index=ignore_index
            )
            src_ids.append(ids)
            src_artv.append(artv)
        while len(src_ids) < n_src:
            src_ids.append(np.full((n_frames,), ignore_index, dtype=np.int64))
            src_artv.append(np.zeros((n_frames, artv_dim), dtype=np.float32))
        id_rows.append(np.stack(src_ids[:n_src], axis=0))
        artv_rows.append(np.stack(src_artv[:n_src], axis=0))
    ids_t = torch.from_numpy(np.stack(id_rows, axis=0))
    artv_t = torch.from_numpy(np.stack(artv_rows, axis=0))
    if device is not None:
        ids_t = ids_t.to(device=device)
        artv_t = artv_t.to(device=device, dtype=dtype)
    return ids_t, artv_t
