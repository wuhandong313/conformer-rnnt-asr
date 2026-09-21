"""DynamicCompression 的 LLM 移植：Qwen3.5 第 N 层输出后的 token 压缩。

算法蓝本 compression.py (ASR 帧间压缩)，模式与其命名逐一对齐：

  naive / savgol / fir    : 帧差 sum|Δh| (可平滑) → find_peaks 峰∪谷 选关键 token
  hanning / savgol_hanning / fir_hanning : 同上检测 + Hanning 窗加权混合
      (compression.py 的可学习 weight predictor 在 LLM 上无训练来源，
       用标准 Hanning 系数替代，结构不变)
  diffq(_hanning)         : 原法保序推广 —— 帧差幅值分位阈值 (自适应
      prominence)，keep_frac 直控压缩率 (峰谷式保留率被极值密度锁死 ~41%)
  novelty(_hanning)       : 局部窗口最近邻距离 —— 非 compression.py 方法，
      仅作 bench 对照消融

插入点 = 第 ``after_layer`` 层输出之后 (默认 3 = 第一个 full_attention 层)。
压缩必须同步三件事 (transformers 5.9.0 modeling_qwen3_5.py 已探明)：

  1. hidden_states (B,L,1024) → 压缩输出 (B,S,1024)
  2. 该层 KV 缓存 (B,2,L,256) → index_select(-2, keep)
     —— Cache.get_seq_length() 恰好读本层，裁掉即自动重锚 decode 位置
  3. 后续所有层的 position_embeddings (cos,sin) → 按 keep 切片

已接受的近似：layers 0..after 的 linear-attn 状态摘要了全量 token，
after 之后的层只见压缩流 (不可逆，方法特性)；kept 位置连续重编号。
控制实验证明两者不损害 needle-QA (needle 保住即答对)。
仅支持 B=1 (全 1 mask → SDPA is_causal 路径，无掩码手术)。
L <= max(min_len, keep_last+2) 时整条链 no-op，解码步 (L==1) 天然跳过。
"""

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks, savgol_filter, firwin, filtfilt
from torch import Tensor

DIFF_MODES = ("naive", "hanning", "fir", "savgol", "savgol_hanning",
              "fir_hanning", "diffq", "diffq_hanning",
              "novelty", "novelty_hanning")
BLEND_MODES = ("hanning", "fir_hanning", "savgol_hanning",
               "diffq_hanning", "novelty_hanning")
QUANT_MODES = ("diffq", "diffq_hanning")


class TokenSaliencyCompressor:
    """帧差显著性 token 压缩器 (无参数)。

    Args:
        mode: naive|hanning|fir|savgol|savgol_hanning|fir_hanning
            (compression.py 对齐族), diffq|diffq_hanning (分位推广),
            novelty|novelty_hanning (对照消融)。
        distance/prominence: find_peaks 参数 (diff 族压缩率旋钮)。
        hanning_window_size: 混合窗宽 (compression.py 同名参数)。
        keep_frac/novelty_window: novelty 模式参数。
        keep_last: 无条件保留的尾部 token 数 (对应 right_context 直通)。
        min_len: 短于此长度不压缩 (透传)。
        enabled: False 时完全 no-op (bench 基线)。
    """

    def __init__(
        self,
        mode: str = "savgol_hanning",
        distance: int = 2,
        prominence: float = 0.0,
        hanning_window_size: int = 5,
        keep_frac: float = 0.5,
        novelty_window: int = 24,
        keep_last: int = 16,
        min_len: int = 64,
        enabled: bool = True,
    ) -> None:
        if mode not in DIFF_MODES + ("novelty",):
            raise ValueError(f"mode 必须是 {DIFF_MODES + ('novelty',)}，得到 {mode}")
        self.mode = mode
        self.distance = distance
        self.prominence = prominence
        self.hanning_window_size = hanning_window_size
        self.keep_frac = keep_frac
        self.novelty_window = novelty_window
        self.keep_last = keep_last
        self.min_len = min_len
        self.enabled = enabled
        self.keep: Optional[Tensor] = None
        self.stats = {"kept": 0, "total": 0, "events": 0}
        self._hann = torch.tensor(
            np.hanning(hanning_window_size), dtype=torch.float32
        )

    @property
    def skip_below(self) -> int:
        return max(self.min_len, self.keep_last + 2)

    def record(self, kept: int, total: int) -> None:
        self.stats["kept"] = kept
        self.stats["total"] = total
        self.stats["events"] += 1

    # ------------------------------------------------------------------
    # 峰/谷检测 (compression.py _detect_peaks_valleys 的 LLM 移植)
    # ------------------------------------------------------------------
    def _detect(self, dn: np.ndarray) -> np.ndarray:
        sig = dn
        if self.mode in ("savgol", "savgol_hanning") and len(dn) > 3:
            wlen = min(9, len(dn))
            if wlen % 2 == 0:
                wlen -= 1
            if wlen > 3:
                sig = savgol_filter(dn, window_length=wlen, polyorder=4)
        prom = self.prominence if self.prominence > 0 else None

        if self.mode in ("fir", "fir_hanning"):
            # compression.py fir 分支: FIR 低通粗检 → 原始信号 ±2 内精修。
            # ASR 里 firwin(num_taps, 50/100) = 截止 1/2 Nyquist，token 域
            # 无帧率概念，直接取归一化截止 0.5。
            num_taps = min(21, len(dn) // 8)
            if num_taps >= 5:
                if num_taps % 2 == 0:
                    num_taps -= 1
                taps = firwin(num_taps, 0.5, pass_zero="lowpass")
                sig = filtfilt(taps, 1.0, dn)
            peaks_a, valleys_a = (
                find_peaks(sig, distance=self.distance, prominence=prom)[0],
                find_peaks(-sig, distance=self.distance, prominence=prom)[0],
            )
            r = 2
            raw = dn
            peaks = [
                lo + int(np.argmax(raw[lo:hi]))
                for p in peaks_a
                for lo, hi in [(max(0, p - r), min(len(raw) - 1, p + r + 1))]
            ]
            valleys = [
                lo + int(np.argmin(raw[lo:hi]))
                for v in valleys_a
                for lo, hi in [(max(0, v - r), min(len(raw) - 1, v + r + 1))]
            ]
            return np.array(peaks), np.array(valleys)

        peaks, _ = find_peaks(sig, distance=self.distance, prominence=prom)
        valleys, _ = find_peaks(-sig, distance=self.distance, prominence=prom)
        return peaks, valleys

    # ------------------------------------------------------------------
    def select(self, hidden: Tensor) -> Tensor:
        """(1,L,D) → 保留索引 (S,) long。峰∪谷∪{0}∪尾K。"""
        L = hidden.size(1)
        if not self.enabled or L <= self.skip_below:
            return torch.arange(L, device=hidden.device)
        x = hidden[0].detach().float()

        if self.mode in ("novelty", "novelty_hanning"):
            score = self._novelty(x)
            thr = torch.quantile(score, 1.0 - self.keep_frac)
            mask = score >= thr
            mask[0] = True
            mask[L - self.keep_last:] = True
            return mask.nonzero(as_tuple=True)[0].to(torch.long)

        diff = (x[1:] - x[:-1]).abs().sum(dim=-1)
        dn = diff.cpu().numpy()

        if self.mode in QUANT_MODES:
            # 帧差信号的保序推广: 每帧显著度 s_i = max(diff_{i-1}, diff_i)
            # (边界 diff 幅值), 按 keep_frac 分位阈值选择 —— 等价于自适应
            # prominence 的峰检测, 使原方法族能扫出温和压缩工作点。
            sig = dn
            if self.mode == "diffq_hanning" and len(dn) > 3:
                wlen = min(9, len(dn))
                if wlen % 2 == 0:
                    wlen -= 1
                if wlen > 3:
                    sig = savgol_filter(dn, window_length=wlen, polyorder=4)
            s = np.maximum(
                np.concatenate([[0.0], sig]), np.concatenate([sig, [0.0]])
            )
            thr = np.quantile(s, 1.0 - self.keep_frac)
            mask = s >= thr
            mask[0] = True
            mask[L - self.keep_last:] = True
            keep = mask.nonzero()[0]
            return torch.tensor(keep.tolist(), device=hidden.device, dtype=torch.long)

        peaks, valleys = self._detect(dn)
        # diff[i] = |h[i+1]-h[i]|: 峰在 diff[k] → 帧 k+1; 谷 → 帧 k
        idx = set((peaks + 1).tolist()) | set(valleys.tolist())
        idx.add(0)
        idx.update(range(L - self.keep_last, L))
        return torch.tensor(sorted(idx), device=hidden.device, dtype=torch.long)

    def _novelty(self, x: Tensor) -> Tensor:
        """(L,D) → (L,) 局部窗口最近邻距离 (分块控内存)。"""
        w = self.novelty_window
        L, D = x.shape
        xp = torch.cat([x[:w].flip(0), x, x[-w:].flip(0)], dim=0)
        out = torch.empty(L, device=x.device, dtype=x.dtype)
        for s in range(0, L, 256):
            e = min(s + 256, L)
            win = xp[s : e + 2 * w].unfold(0, 2 * w + 1, 1)[: e - s]  # (n,D,2w+1)
            d = (win - x[s:e].unsqueeze(-1)).pow(2).sum(1).sqrt()  # (n,2w+1)
            d[:, w] = float("inf")
            out[s:e] = d.min(dim=1).values
        return out

    # ------------------------------------------------------------------
    def compress(self, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """(1,L,D) → (压缩输出 (1,S,D), keep 索引 (S,))。

        hanning 族: kept 帧 = 固定 Hanning 窗内加权平均 (被丢 token 的
        信息折叠进相邻 kept 帧, 对应 compression.py _hanning_smooth;
        混合窗限制在主区 [0, L-keep_last) 内, 尾 K 与 compression.py 的
        right_context 一样原样直通)。
        """
        L = hidden.size(1)
        keep = self.select(hidden)
        if keep.numel() == L:
            return hidden, keep
        if self.mode not in BLEND_MODES:
            return hidden.index_select(1, keep), keep

        x = hidden[0]
        dtype = x.dtype
        main_end = L - self.keep_last
        half = self.hanning_window_size // 2
        hann = self._hann.to(device=x.device, dtype=torch.float32)
        rows = []
        for i in keep.tolist():
            i = int(i)
            if i >= main_end:  # right_context 直通
                rows.append(x[i].float())
                continue
            s = max(0, i - half)
            e = min(main_end, i + half + 1)
            ws = s - (i - half)
            we = self.hanning_window_size - ((i + half + 1) - e)
            w = hann[ws:we]
            w = w / w.sum()
            rows.append((x[s:e].float() * w[:, None]).sum(0))
        out = torch.stack(rows).to(dtype).unsqueeze(0)
        return out, keep


# ═══════════════════════════════════════════════════════════════════════════
# 钩子安装
# ═══════════════════════════════════════════════════════════════════════════


def install_compression(model, compressor: TokenSaliencyCompressor, after_layer: int = 3):
    """把压缩器挂进 Qwen3.5 主干的层钩子。返回安装的 handle 列表。

    要求 after_layer 是 full_attention 层 (Qwen3.5: 3/7/11/15/19/23)：
    该层缓存为逐 token 行，可精确 index_select；get_seq_length 亦读它。
    """
    text_model = model.model.language_model
    cfg = text_model.config
    if cfg.layer_types[after_layer] != "full_attention":
        raise ValueError(
            f"after_layer={after_layer} 不是 full_attention "
            f"(layer_types={cfg.layer_types})；linear 层缓存不可裁。"
        )
    layers = text_model.layers
    handles = []

    def pre_cut(module, args, kwargs):
        # 每次 forward (prefill 或 decode 步) 先清状态
        compressor.keep = None

    def post_cut(module, args, kwargs, output):
        hidden = output
        if hidden.size(0) != 1:
            raise RuntimeError(f"token 压缩仅支持 B=1，得到 batch={hidden.size(0)}")
        L = hidden.size(1)
        out, keep = compressor.compress(hidden)
        if keep.numel() == L:
            return output  # 透传，零副作用
        cache = kwargs.get("past_key_values")
        if cache is not None:
            entry = cache.layers[after_layer]
            if getattr(entry, "is_initialized", False) and entry.keys.size(-2) == L:
                entry.keys = entry.keys.index_select(-2, keep)
                entry.values = entry.values.index_select(-2, keep)
        compressor.keep = keep
        compressor.record(keep.numel(), L)
        return out

    def pre_after(module, args, kwargs):
        # 循环传下来的 hidden 已在 post_cut 压缩过，这里只修 pe/mask
        keep = compressor.keep
        if keep is None:
            return None  # 未压缩，零开销
        pe = kwargs.get("position_embeddings")
        if pe is not None and pe[0].size(1) != keep.numel():
            new_kwargs = dict(kwargs)
            new_kwargs["position_embeddings"] = (pe[0][:, keep], pe[1][:, keep])
            mask = new_kwargs.get("attention_mask")
            if mask is not None and mask.dim() == 4 and mask.size(-2) != keep.numel():
                new_kwargs["attention_mask"] = mask.index_select(-2, keep)[..., keep]
            return args, new_kwargs
        return None

    handles.append(
        layers[after_layer].register_forward_pre_hook(pre_cut, with_kwargs=True)
    )
    handles.append(
        layers[after_layer].register_forward_hook(post_cut, with_kwargs=True)
    )
    for i in range(after_layer + 1, len(layers)):
        handles.append(layers[i].register_forward_pre_hook(pre_after, with_kwargs=True))
    return handles
