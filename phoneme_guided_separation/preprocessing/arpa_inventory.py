"""去重音 ARPABET 音素表、ID、以及 ARPABET → IPA → ArtV 原型向量。

MFA ``english_us_arpa`` 产出带重音的 ARPABET（如 IH1）。训练 CE 用去重音后的
离散 ID；ArtV 再经 IPA 查 PanPhon。本模块是这两张表的源。

静音 ``sil`` / 噪声 ``spn`` / 未知 ``unk`` 不查 PanPhon，ArtV 用全 0。
``AH0`` 与 ``AH1`` 去重音后都是 ``AH``（同一 ID）；IPA 暂统一为 ``ʌ``。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

# 0=sil 便于当「无音素」类；ignore_index 仍用 -100（未对齐帧），不要占用 0。
SIL_PHONES = ("sil", "spn")
UNK_PHONE = "unk"

# CMU / MFA english_us_arpa 去重音后的标准音素（39）
ARPA_PHONES: tuple[str, ...] = (
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
)

PHONES: tuple[str, ...] = SIL_PHONES + ARPA_PHONES + (UNK_PHONE,)
PHONE2ID: dict[str, int] = {p: i for i, p in enumerate(PHONES)}
ID2PHONE: dict[int, str] = {i: p for p, i in PHONE2ID.items()}
NUM_PHONES = len(PHONES)  # 42；模型 num_phonemes 可 ≥ 此值

# 去重音 ARPABET → IPA。双元音 / 塞擦音写成 IPA 串，查 PanPhon 后对音段向量取平均。
ARPA_TO_IPA: dict[str, str] = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

# 个别符号 PanPhon 不认时的备选（仍表示同一音段）
_IPA_FALLBACKS: dict[str, tuple[str, ...]] = {
    "ɝ": ("ɚ", "əɹ", "ɜ"),
    "ɹ": ("r",),
    "ɡ": ("g",),
    "ɑ": ("ɑː", "a"),
    "tʃ": ("ʧ",),
    "dʒ": ("ʤ",),
}

_STRESS_RE = re.compile(r"[012]$")
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PHONE2ID_JSON = _THIS_DIR / "arpa_phone2id.json"
DEFAULT_IPA_JSON = _THIS_DIR / "arpa_to_ipa.json"
DEFAULT_ARTV_NPY = _THIS_DIR / "arpa_artv.npy"


def destress_phone(label: str) -> str:
    """MFA phones 标签 → 去重音 ARPABET；<eps>/空 → sil。"""
    s = (label or "").strip()
    if s in ("", "<eps>"):
        return "sil"
    s = _STRESS_RE.sub("", s)
    low = s.lower()
    if low in ("sil", "silence"):
        return "sil"
    if low in ("spn", "unk"):
        return "spn" if low == "spn" else UNK_PHONE
    if low == "sp":
        return "sil"
    return s.upper()


def phone_to_id(label: str) -> int:
    dest = destress_phone(label)
    return PHONE2ID.get(dest, PHONE2ID[UNK_PHONE])


def id_to_phone(idx: int) -> str:
    return ID2PHONE.get(int(idx), UNK_PHONE)


def destress_to_ipa(dest: str) -> str | None:
    if dest in SIL_PHONES or dest == UNK_PHONE:
        return None
    return ARPA_TO_IPA.get(dest)


def dump_json_maps(
    phone2id_path: Path = DEFAULT_PHONE2ID_JSON,
    ipa_path: Path = DEFAULT_IPA_JSON,
) -> None:
    phone2id_path.write_text(
        json.dumps(PHONE2ID, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ipa_path.write_text(
        json.dumps(ARPA_TO_IPA, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _ipa_candidates(ipa: str) -> list[str]:
    out = [ipa]
    out.extend(_IPA_FALLBACKS.get(ipa, ()))
    return out


def _feature_table():
    """构造 PanPhon FeatureTable。

    中文 Windows 默认 GBK，而 panphon 的 CSV 是 UTF-8；
    ``Traversable.open()`` / ``pd.read_csv`` 不指定 encoding 就会炸。
    """
    import panphon
    import pandas as pd
    from pathlib import Path

    orig_path_open = Path.open
    orig_read_csv = pd.read_csv
    orig_open = __import__("builtins").open

    def path_open_utf8(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in str(mode) and encoding is None:
            encoding = "utf-8"
        return orig_path_open(self, mode, buffering, encoding, errors, newline)

    def builtin_open_utf8(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode:
            kwargs.setdefault("encoding", "utf-8")
        return orig_open(file, mode, *args, **kwargs)

    def read_csv_utf8(*args, **kwargs):
        kwargs.setdefault("encoding", "utf-8")
        return orig_read_csv(*args, **kwargs)

    import builtins

    Path.open = path_open_utf8  # type: ignore[method-assign]
    builtins.open = builtin_open_utf8  # type: ignore[assignment]
    pd.read_csv = read_csv_utf8
    try:
        return panphon.FeatureTable()
    finally:
        Path.open = orig_path_open  # type: ignore[method-assign]
        builtins.open = orig_open  # type: ignore[assignment]
        pd.read_csv = orig_read_csv



def ipa_to_artv(ipa: str, ft=None) -> np.ndarray | None:
    """IPA 串 → 发音特征向量。多音段（双元音/塞擦音）对 numeric 向量取平均。"""
    if ft is None:
        ft = _feature_table()
    for cand in _ipa_candidates(ipa):
        segs = ft.word_fts(cand)
        if not segs:
            continue
        vecs = [np.asarray(s.numeric(), dtype=np.float32) for s in segs]
        return np.mean(np.stack(vecs, axis=0), axis=0)
    return None


def artv_dim(ft=None) -> int:
    if ft is None:
        ft = _feature_table()
    probe = ipa_to_artv("s", ft)
    if probe is None:
        raise RuntimeError("PanPhon 无法编码探针音素 's'，请检查 panphon 安装")
    return int(probe.shape[0])


def build_artv_table(ft=None) -> tuple[np.ndarray, list[str]]:
    """返回 (table [K, D], missing)。missing 为未能查出 ArtV 的 ARPABET。"""
    if ft is None:
        ft = _feature_table()
    d = artv_dim(ft)
    table = np.zeros((NUM_PHONES, d), dtype=np.float32)
    missing: list[str] = []
    for phone, idx in PHONE2ID.items():
        ipa = destress_to_ipa(phone)
        if ipa is None:
            continue
        vec = ipa_to_artv(ipa, ft)
        if vec is None:
            missing.append(phone)
            continue
        table[idx] = vec
    return table, missing


def save_artv_table(path: Path = DEFAULT_ARTV_NPY, table: np.ndarray | None = None) -> np.ndarray:
    if table is None:
        table, missing = build_artv_table()
        if missing:
            print(f"[warn] 下列音素未查到 PanPhon 向量，保持全 0: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), table)
    return table
