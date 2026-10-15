"""MFA 强制对齐 → 去重音 ARPABET ID → PanPhon ArtV。

Python 依赖：``conda activate convtasnet``（soundfile / numpy / panphon / tqdm）。
MFA 命令行可在当前环境或兄弟环境 ``aligner``；Windows / Ubuntu 均可，脚本会自动查找。

默认数据根目录来自 ``configs/basic_env.yaml`` 的 ``system.system_type``
（经 ``system_paths.get_data_root()``），再拼 ``libri`` / ``libri_phoneme`` 等相对路径。

单条冒烟::

    python phoneme_guided_separation/preprocessing/mfa_preprocess.py

批量处理 LibriSpeech（目录结构镜像到 libri_phoneme）::

    python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch
    python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch --num-jobs 6   # 按机器可用核调 MFA 并行度
    python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch --filter-librimix
    python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch --filter-librimix-small
    python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch --splits dev-clean --limit 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# python phoneme_guided_separation/preprocessing/mfa_preprocess.py 时脚本目录在 path 最前，可 import arpa_inventory
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import arpa_inventory  # noqa: E402
from system_paths import get_data_root  # noqa: E402

# Windows 控制台常见 GBK：统一用 UTF-8，避免中文/音素符号打印炸编码
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 默认路径在 parse_args → _fill_data_path_defaults 里按 data_root 拼接
# （data_root 来自 configs/basic_env.yaml 的 system.system_type）
DICT_NAME = "english_us_arpa"  # 词典
ACOUSTIC_NAME = "english_us_arpa"  # 声学模型


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MFA 对齐：单条冒烟或批量 LibriSpeech")
    p.add_argument(
        "--batch",
        action="store_true",
        help="批量处理 --src → --dst（镜像相对路径）",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="数据根目录（默认: basic_env.yaml → system_paths.get_data_root()）",
    )
    p.add_argument("--src", type=Path, default=None, help="输入根目录（Libri 解压树）")
    p.add_argument("--dst", type=Path, default=None, help="输出根目录")
    p.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="只处理这些 split，如 train-clean-100 dev-clean；默认全部",
    )
    p.add_argument(
        "--filter-librimix",
        action="store_true",
        help="只处理 librimix 的 s1/s2 里出现过的 LibriSpeech 句子，其余跳过",
    )
    p.add_argument(
        "--filter-librimix-small",
        action="store_true",
        help="只处理 librimix-small 的 s1/s2 里出现过的句子（优先于 --filter-librimix）",
    )
    p.add_argument(
        "--librimix-root",
        type=Path,
        default=None,
        help="--filter-librimix 用的 LibriMix 根目录",
    )
    p.add_argument(
        "--librimix-small-root",
        type=Path,
        default=None,
        help="--filter-librimix-small 用的小集根目录",
    )
    p.add_argument("--limit", type=int, default=None, help="最多处理多少条（调试用）")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有 phone_ids.npy；默认跳过",
    )
    p.add_argument(
        "--num-jobs",
        type=int,
        default=0,
        help="传给 mfa align 的 --num_jobs（CPU 并行）。"
        "0=自动：max(1, cpu_count-2)，例如 10 核约用 8；"
        "可用核约 6 时可显式 --num-jobs 6",
    )
    p.add_argument(
        "--save-json",
        action="store_true",
        help="批量时也写 phones.json（体积大，默认不写）",
    )
    p.add_argument("--wav", type=Path, default=None, help="单条模式：输入 wav")
    p.add_argument(
        "--transcript",
        type=str,
        default=None,
        help="转写文本；默认从 LibriSpeech *.trans.txt 按文件名解析",
    )
    p.add_argument(
        "--librispeech-root",
        type=Path,
        default=None,
        help="含 train-clean-100 的 LibriSpeech 根目录",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="对齐输出目录（默认 phoneme_guided_separation/preprocessing/_mfa_out/<utt_id>）",
    )
    p.add_argument(
        "--dictionary",
        default=DICT_NAME,
        help="MFA 词典名或路径",
    )
    p.add_argument(
        "--acoustic",
        default=ACOUSTIC_NAME,
        help="MFA 声学模型名或路径",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="不调用 mfa model download（本地已有模型时默认也会跳过）",
    )
    p.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="写入 corpus 前重采样到该采样率（MFA 英文模型常用 16k）",
    )
    p.add_argument(
        "--skip-align",
        action="store_true",
        help="不跑 MFA，只用 --out-dir（或默认目录）里已有 TextGrid 做后续 ID/ArtV",
    )
    p.add_argument(
        "--skip-artv",
        action="store_true",
        help="只输出去重音 ARPABET ID，不查 PanPhon",
    )
    args = p.parse_args()
    _fill_data_path_defaults(args)
    return args


def _fill_data_path_defaults(args: argparse.Namespace) -> None:
    """未显式传入的路径，按 basic_env 的 data_root 拼接。"""
    root = Path(args.data_root) if args.data_root else Path(get_data_root())
    args.data_root = root
    if args.src is None:
        args.src = root / "libri"
    if args.dst is None:
        args.dst = root / "libri_phoneme"
    if args.librimix_root is None:
        args.librimix_root = root / "librimix"
    if args.librimix_small_root is None:
        args.librimix_small_root = root / "librimix-small"
    if args.librispeech_root is None:
        args.librispeech_root = root / "libri" / "unzip" / "LibriSpeech"
    if args.wav is None:
        args.wav = (
            root
            / "librimix-small"
            / "wav8k"
            / "min"
            / "train-100"
            / "s1"
            / "19-198-0001_2.216_3259-158083-0121_-2.216.wav"
        )
    if args.num_jobs is None or int(args.num_jobs) <= 0:
        # 给系统/其它任务留 2 核；瓶颈在 MFA 的多进程，不是 Python 线程
        ncpu = os.cpu_count() or 4
        args.num_jobs = max(1, ncpu - 2)
    else:
        args.num_jobs = max(1, int(args.num_jobs))


def find_mfa() -> str:
    """Locate MFA CLI: PATH → current env → sibling ``aligner`` env (Win/Linux)."""
    mfa = shutil.which("mfa")
    if mfa:
        return mfa

    prefix = Path(sys.prefix)
    # conda/venv: Windows 用 Scripts/mfa.exe，Ubuntu/Linux 用 bin/mfa
    names = ("mfa.exe", "mfa")
    bin_dirs = ("Scripts", "bin")
    candidates: list[Path] = []
    # 1) 当前环境（可能就装在本 env）
    for d in bin_dirs:
        for name in names:
            candidates.append(prefix / d / name)
    # 2) 兄弟环境 aligner（常见：Python 在 convtasnet，MFA 在 aligner）
    for d in bin_dirs:
        for name in names:
            candidates.append(prefix.parent / "aligner" / d / name)
    # 3) 本机常见绝对路径兜底
    candidates.extend(
        [
            Path(r"C:\Users\feng\anaconda3\envs\aligner\Scripts\mfa.exe"),
            Path.home() / "anaconda3" / "envs" / "aligner" / "bin" / "mfa",
            Path.home() / "miniconda3" / "envs" / "aligner" / "bin" / "mfa",
            Path("/opt/conda/envs/aligner/bin/mfa"),
            Path("/root/miniconda3/envs/aligner/bin/mfa"),
            Path("/root/anaconda3/envs/aligner/bin/mfa"),
        ]
    )
    for c in candidates:
        if c.is_file():
            return str(c)
    raise SystemExit(
        "未找到 mfa。可任选其一：\n"
        "  1) 装进当前环境：conda install -c conda-forge montreal-forced-aligner\n"
        "  2) 独立环境：conda create -n aligner -c conda-forge montreal-forced-aligner -y\n"
        "     （脚本会在当前 env 的 bin/Scripts 或兄弟 env aligner 里查找）\n"
        "  3) 确保 `mfa` 已在 PATH 中：which mfa / where mfa"
    )


def mfa_subprocess_env() -> dict[str, str]:
    """调用 MFA 时补上其所在 conda env 的动态库目录（Windows libsndfile / Linux lib）。"""
    env = os.environ.copy()
    mfa_exe = Path(find_mfa())
    root = mfa_exe.parent.parent
    extras = [
        root / "Library" / "bin",  # Windows conda
        root / "Scripts",
        root / "Library" / "usr" / "bin",
        root / "bin",  # Linux conda
        root / "lib",
        root / "lib64",
    ]
    prefix = os.pathsep.join(str(p) for p in extras if p.is_dir())
    if prefix:
        env["PATH"] = prefix + os.pathsep + env.get("PATH", "")
        # Linux 常需 LD_LIBRARY_PATH 才能找到 conda 里的 .so
        lib_dirs = [str(p) for p in (root / "lib", root / "lib64") if p.is_dir()]
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = (
                os.pathsep.join(lib_dirs)
                + os.pathsep
                + env.get("LD_LIBRARY_PATH", "")
            )
    return env


def run(cmd: list[str], verbose: bool = True, **kwargs) -> subprocess.CompletedProcess:
    if verbose:
        print(">>", " ".join(str(c) for c in cmd), flush=True)
    kwargs.setdefault("env", mfa_subprocess_env())
    return subprocess.run(cmd, check=True, **kwargs)


def mfa_pretrained_root() -> Path:
    """MFA 预训练模型目录：``$MFA_ROOT_DIR/pretrained_models``，默认 ``~/Documents/MFA``。"""
    root = Path(os.environ.get("MFA_ROOT_DIR", "~/Documents/MFA")).expanduser()
    return root / "pretrained_models"


def find_local_mfa_model(kind: str, name: str) -> Path | None:
    """词典/声学模型已在本地则返回路径。``name`` 也可以是文件路径。"""
    p = Path(name)
    if p.is_file() or p.is_dir():
        return p
    folder = mfa_pretrained_root() / kind
    if kind == "dictionary":
        suffixes = (".dict", ".txt", ".zip", "")
    else:
        suffixes = (".zip", "")
    for suf in suffixes:
        cand = folder / (f"{name}{suf}" if suf else name)
        if cand.is_file() or (not suf and cand.is_dir()):
            return cand
    return None


def ensure_models(mfa: str, dictionary: str, acoustic: str, skip: bool) -> None:
    """本地已有则不 download。``mfa model download`` 每次都会打 GitHub，未认证每小时 60 次就限流。"""
    needed = (("dictionary", dictionary), ("acoustic", acoustic))
    missing: list[tuple[str, str]] = []
    for kind, name in needed:
        found = find_local_mfa_model(kind, name)
        if found:
            print(f"[info] 本地 MFA {kind}: {found}", flush=True)
        else:
            missing.append((kind, name))
    if skip:
        if missing:
            names = ", ".join(f"{k}/{n}" for k, n in missing)
            raise SystemExit(
                f"--skip-download 但本地没有: {names}\n"
                f"  目录: {mfa_pretrained_root()}\n"
                "  或等 GitHub 限流过后再跑（不要加 --skip-download）"
            )
        return
    if not missing:
        print("[info] 本地已有词典和声学模型，跳过 mfa model download", flush=True)
        return
    token = os.environ.get("MFA_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    extra: list[str] = ["--github_token", token] if token else []
    for kind, name in missing:
        cmd = [mfa, "model", "download", kind, name, *extra]
        try:
            run(cmd)
        except subprocess.CalledProcessError as e:
            found = find_local_mfa_model(kind, name)
            if found:
                print(f"[warn] download 失败，改用本地 {found}", flush=True)
                continue
            raise SystemExit(
                f"mfa model download {kind} {name} 失败（常是 GitHub API 限流）。\n"
                "  模型若已下载过：加 --skip-download 重跑。\n"
                "  未下载：等限流重置，或设置环境变量 GITHUB_TOKEN 后重跑。"
            ) from e


def parse_utt_id_from_wav(wav: Path) -> str:
    """LibriMix ``{utt}_{snr}_...`` 或 LibriSpeech ``spk-chap-idx`` → utt id。"""
    stem = wav.stem
    m = re.match(r"^(\d+-\d+-\d+)_", stem)
    if m:
        return m.group(1)
    m = re.match(r"^(\d+-\d+-\d+)$", stem)
    if m:
        return m.group(1)
    raise SystemExit(f"无法从文件名解析 LibriSpeech utt id: {wav.name}")


def lookup_librispeech_transcript(utt_id: str, libri_root: Path) -> str:
    # 19-198-0001 → train-clean-100/19/198/19-198.trans.txt
    parts = utt_id.split("-")
    if len(parts) != 3:
        raise SystemExit(f"非法 utt id: {utt_id}")
    spk, chap, _ = parts
    trans = None
    for split in ("train-clean-100", "train-clean-360", "dev-clean", "test-clean"):
        cand = libri_root / split / spk / chap / f"{spk}-{chap}.trans.txt"
        if cand.is_file():
            trans = cand
            break
    if trans is None:
        raise SystemExit(
            f"未找到转写文件（utt={utt_id}）。请指定 --transcript 或检查 "
            f"--librispeech-root={libri_root}"
        )
    prefix = utt_id + " "
    for line in trans.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            text = line[len(prefix) :].strip()
            if not text:
                raise SystemExit(f"转写行为空: {trans} :: {utt_id}")
            return text
    raise SystemExit(f"{trans} 中没有 {utt_id}")


def lookup_chapter_transcript(utt_id: str, chapter_dir: Path) -> str | None:
    parts = utt_id.split("-")
    if len(parts) != 3:
        return None
    trans = chapter_dir / f"{parts[0]}-{parts[1]}.trans.txt"
    if not trans.is_file():
        return None
    prefix = utt_id + " "
    for line in trans.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            text = line[len(prefix) :].strip()
            return text or None
    return None


def prepare_corpus(
    wav: Path,
    transcript: str,
    corpus_dir: Path,
    target_sr: int,
    utt_id: str | None = None,
    verbose: bool = True,
) -> Path:
    """写成 MFA 语料：同名 .wav + .lab。"""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    short = utt_id or parse_utt_id_from_wav(wav)
    out_wav = corpus_dir / f"{short}.wav"
    out_lab = corpus_dir / f"{short}.lab"

    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        shutil.copy2(wav, out_wav)
        out_lab.write_text(transcript + "\n", encoding="utf-8")
        if verbose:
            print(f"[warn] 未安装 soundfile，未重采样，直接复制 {wav}")
        return corpus_dir

    audio, sr = sf.read(str(wav), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        n_out = int(round(len(audio) * target_sr / sr))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)
        if verbose:
            print(f"[info] 重采样 {sr} → {target_sr} Hz（{len(audio)} samples）")
    else:
        audio = audio.astype(np.float32)
        if verbose:
            print(f"[info] 采样率已是 {sr} Hz")

    sf.write(str(out_wav), audio, target_sr)
    out_lab.write_text(transcript + "\n", encoding="utf-8")
    if verbose:
        print(f"[info] corpus: {out_wav.name} + {out_lab.name}")
        print(f"[info] transcript: {transcript}")
    return corpus_dir


def align_one(
    mfa: str,
    wav: Path,
    lab: Path,
    dictionary: str,
    acoustic: str,
    out_dir: Path,
    verbose: bool = True,
) -> Path:
    """优先 mfa align_one；失败则退回临时 corpus + mfa align。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd_one = [
        mfa,
        "align_one",
        str(wav),
        str(lab),
        dictionary,
        acoustic,
        str(out_dir),
    ]
    try:
        run(cmd_one, verbose=verbose, stdout=None if verbose else subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        if verbose:
            print("[warn] align_one 失败，改用 mfa align（单句 corpus）…", flush=True)
        corpus = wav.parent
        run(
            [
                mfa,
                "align",
                str(corpus),
                dictionary,
                acoustic,
                str(out_dir),
                "--clean",
            ],
            verbose=verbose,
            stdout=None if verbose else subprocess.DEVNULL,
        )
    grids = list(out_dir.rglob("*.TextGrid"))
    if not grids:
        raise FileNotFoundError(f"未找到 TextGrid: {out_dir}")
    return grids[0]


def parse_textgrid_intervals(text: str, tier_name: str) -> list[tuple[float, float, str]]:
    """极简 TextGrid 解析：按 tier 名取 intervals（不依赖 praatio）。"""
    # 找 name = "phones" / "words" 的 IntervalTier
    pattern = (
        rf'name\s*=\s*"{re.escape(tier_name)}"\s*'
        r"xmin\s*=\s*([0-9.]+)\s*"
        r"xmax\s*=\s*([0-9.]+)\s*"
        r"intervals:\s*size\s*=\s*(\d+)\s*"
    )
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return []
    n = int(m.group(3))
    rest = text[m.end() :]
    intervals: list[tuple[float, float, str]] = []
    item_re = re.compile(
        r"intervals\s*\[\d+\]\s*:\s*"
        r"xmin\s*=\s*([0-9.]+)\s*"
        r"xmax\s*=\s*([0-9.]+)\s*"
        r'text\s*=\s*"(.*?)"',
        re.DOTALL,
    )
    for i, im in enumerate(item_re.finditer(rest)):
        if i >= n:
            break
        xmin, xmax, lab = float(im.group(1)), float(im.group(2)), im.group(3)
        intervals.append((xmin, xmax, lab))
    return intervals


def print_alignment(tg_path: Path) -> None:
    raw = tg_path.read_text(encoding="utf-8", errors="replace")
    print("\n========== MFA 输出 ==========")
    print(f"TextGrid: {tg_path}")
    for tier in ("words", "phones"):
        intervals = parse_textgrid_intervals(raw, tier)
        print(f"\n--- tier: {tier} ({len(intervals)} intervals) ---")
        for xmin, xmax, lab in intervals:
            if lab.strip() == "":
                continue
            print(f"{xmin:8.3f}  {xmax:8.3f}  {lab}")
    print("\n========== TextGrid 全文（前 80 行）==========")
    lines = raw.splitlines()
    for line in lines[:80]:
        print(line)
    if len(lines) > 80:
        print(f"... ({len(lines) - 80} more lines)")


def encode_phones(
    intervals: list[tuple[float, float, str]],
    artv_table: np.ndarray | None,
) -> list[dict]:
    """去重音 → ID →（可选）查 ArtV 原型。"""
    rows: list[dict] = []
    for xmin, xmax, raw in intervals:
        dest = arpa_inventory.destress_phone(raw)
        pid = arpa_inventory.phone_to_id(raw)
        ipa = arpa_inventory.destress_to_ipa(dest)
        rec: dict = {
            "xmin": xmin,
            "xmax": xmax,
            "raw": raw,
            "destressed": dest,
            "id": pid,
            "ipa": ipa,
        }
        if artv_table is not None:
            rec["artv"] = artv_table[pid].tolist()
        rows.append(rec)
    return rows


def print_phone_ids(rows: list[dict], artv_table: np.ndarray | None) -> None:
    print("\n========== 去重音 ARPABET ID ==========")
    print(
        f"词典: {arpa_inventory.DEFAULT_PHONE2ID_JSON.name}  "
        f"K={arpa_inventory.NUM_PHONES}  (sil=0, spn=1, unk={arpa_inventory.PHONE2ID['unk']})"
    )
    if artv_table is not None:
        print(f"ArtV 原型表: {tuple(artv_table.shape)}  (sil/spn/unk 为全 0)")
    print(f"{'t0':>8}  {'t1':>8}  {'raw':<8}  {'dest':<6}  {'id':>4}  {'ipa':<6}  artv[:8]")
    for r in rows:
        ipa = r["ipa"] or "-"
        if "artv" in r:
            head = np.asarray(r["artv"], dtype=np.float32)[:8]
            artv_s = "[" + " ".join(f"{x:.0f}" for x in head) + "]"
        else:
            artv_s = "-"
        print(
            f"{r['xmin']:8.3f}  {r['xmax']:8.3f}  {r['raw']:<8}  "
            f"{r['destressed']:<6}  {r['id']:4d}  {ipa:<6}  {artv_s}"
        )


def save_phone_outputs(
    out_dir: Path,
    rows: list[dict],
    artv_table: np.ndarray | None,
    stem: str | None = None,
    save_json: bool = True,
    verbose: bool = True,
) -> None:
    """stem 为 None 时写 phones.json；否则写 {stem}.phone_ids.npy 等同名文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_json:
        json_path = out_dir / (f"{stem}.phones.json" if stem else "phones.json")
        payload = {
            "phone2id": arpa_inventory.PHONE2ID,
            "arpa_to_ipa": arpa_inventory.ARPA_TO_IPA,
            "intervals": rows,
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if verbose:
            print(f"[info] 写出 {json_path}")
    ids_name = f"{stem}.phone_ids.npy" if stem else "phone_ids.npy"
    ids = np.asarray([r["id"] for r in rows], dtype=np.int64)
    np.save(str(out_dir / ids_name), ids)
    if verbose:
        print(f"[info] 写出 {out_dir / ids_name}  shape={ids.shape}")
    times_name = f"{stem}.phone_times.npy" if stem else "phone_times.npy"
    times = np.asarray([[r["xmin"], r["xmax"]] for r in rows], dtype=np.float32)
    np.save(str(out_dir / times_name), times)
    if verbose:
        print(f"[info] 写出 {out_dir / times_name}  shape={times.shape}")
    if artv_table is not None and rows and "artv" in rows[0]:
        artv_name = f"{stem}.artv.npy" if stem else "artv.npy"
        artv = np.asarray([r["artv"] for r in rows], dtype=np.float32)
        np.save(str(out_dir / artv_name), artv)
        if verbose:
            print(f"[info] 写出 {out_dir / artv_name}  shape={artv.shape}")


def load_artv_table() -> np.ndarray | None:
    cached = arpa_inventory.DEFAULT_ARTV_NPY
    if cached.is_file():
        table = np.load(str(cached))
        print(
            f"[info] 使用已有 ArtV 原型表 {cached}  shape={tuple(table.shape)}",
            flush=True,
        )
        return table
    try:
        table, missing = arpa_inventory.build_artv_table()
    except ImportError:
        print(
            "[warn] 未安装 panphon，只输出 ARPABET ID。在 convtasnet 中:\n"
            "         pip install panphon",
            flush=True,
        )
        return None
    except UnicodeDecodeError as e:
        print(
            f"[warn] PanPhon 读表编码失败（{e}）。"
            "请设 UTF-8 或先生成 phoneme_guided_separation/preprocessing/arpa_artv.npy。本次只输出 ARPABET ID。",
            flush=True,
        )
        return None
    if missing:
        print(f"[warn] PanPhon 未覆盖（ArtV 全 0）: {missing}", flush=True)
    arpa_inventory.dump_json_maps()
    arpa_inventory.save_artv_table(table=table)
    print(
        f"[info] 原型表已写 {arpa_inventory.DEFAULT_ARTV_NPY}  "
        f"shape={tuple(table.shape)}",
        flush=True,
    )
    return table


def find_existing_textgrid(out_dir: Path) -> Path:
    grids = list(out_dir.rglob("*.TextGrid"))
    if not grids:
        raise SystemExit(f"--skip-align 但未找到 TextGrid: {out_dir}")
    return grids[0]


AUDIO_EXTS = {".flac", ".wav"}
_UTT_IN_NAME = re.compile(r"\d+-\d+-\d+")


def collect_librimix_utt_ids(librimix_root: Path) -> tuple[set[str], int]:
    """从 librimix 的 s1/s2 wav 文件名收集用到的 LibriSpeech utt id。

    文件名形如 ``19-198-0001_2.216_3259-158083-0121_-2.216.wav``：
    s1 目录取第一段 id，s2 目录取第二段 id。
    """
    if not librimix_root.is_dir():
        raise SystemExit(f"--librimix-root 不是目录: {librimix_root}")
    ids: set[str] = set()
    n_files = 0
    for p in librimix_root.rglob("*.wav"):
        chan = p.parent.name.lower()
        if chan not in ("s1", "s2"):
            continue
        n_files += 1
        found = _UTT_IN_NAME.findall(p.stem)
        if chan == "s1" and found:
            ids.add(found[0])
        elif chan == "s2" and len(found) >= 2:
            ids.add(found[1])
        elif found:
            ids.add(found[0])
    return ids, n_files


def iter_libri_audio(src: Path, splits: list[str] | None) -> list[Path]:
    files: list[Path] = []
    for p in src.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        if splits:
            parts = set(p.relative_to(src).parts)
            if not parts.intersection(splits):
                continue
        files.append(p)
    files.sort()
    return files


def dst_paths(src: Path, dst: Path, audio: Path) -> tuple[Path, Path, Path, Path]:
    """镜像相对路径：xxx.flac → xxx.phone_ids.npy / xxx.artv.npy / xxx.TextGrid。"""
    rel = audio.relative_to(src)
    stem_path = dst / rel.with_suffix("")
    parent = stem_path.parent
    stem = stem_path.name
    return (
        parent / f"{stem}.phone_ids.npy",
        parent / f"{stem}.artv.npy",
        parent / f"{stem}.TextGrid",
        parent,
    )


def export_alignment(
    tg: Path,
    dst_parent: Path,
    stem: str,
    artv_table: np.ndarray | None,
    save_json: bool,
) -> None:
    raw = tg.read_text(encoding="utf-8", errors="replace")
    phone_iv = parse_textgrid_intervals(raw, "phones")
    if not phone_iv:
        raise RuntimeError(f"无 phones 层: {tg}")
    rows = encode_phones(phone_iv, artv_table)
    dest_tg = dst_parent / f"{stem}.TextGrid"
    dst_parent.mkdir(parents=True, exist_ok=True)
    if tg.resolve() != dest_tg.resolve():
        shutil.copy2(tg, dest_tg)
    save_phone_outputs(
        dst_parent,
        rows,
        artv_table,
        stem=stem,
        save_json=save_json,
        verbose=False,
    )


def align_corpus(
    mfa: str,
    corpus: Path,
    dictionary: str,
    acoustic: str,
    out_dir: Path,
    num_jobs: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_flags: list[list[str]] = [["--single_speaker"], []]
    last_err = ""
    for extra in extra_flags:
        cmd = [
            mfa,
            "align",
            str(corpus),
            dictionary,
            acoustic,
            str(out_dir),
            "--clean",
            "--num_jobs",
            str(num_jobs),
            *extra,
        ]
        proc = subprocess.run(
            cmd,
            env=mfa_subprocess_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            return
        last_err = (proc.stderr or proc.stdout or "")[-4000:]
        low = last_err.lower()
        if extra and (
            "unrecognized" in low or "no such option" in low or "invalid" in low and "single_speaker" in low
        ):
            continue
        raise RuntimeError(f"mfa align 失败\n{last_err}")
    raise RuntimeError(f"mfa align 失败\n{last_err}")


def _eta_postfix(pbar, n_todo: int, ok: int, failed: int, skipped: int) -> str:
    rate = pbar.format_dict.get("rate") or 0.0
    n_left = max(n_todo - pbar.n, 0)
    if rate and rate > 0:
        eta_dt = datetime.now() + timedelta(seconds=n_left / rate)
        eta_str = eta_dt.strftime("%Y-%m-%d %H:%M:%S")
        remain_str = str(timedelta(seconds=int(n_left / rate)))
    else:
        eta_str = "…"
        remain_str = "…"
    return f"ETA {eta_str}  remain {remain_str}  ok={ok} fail={failed} skip={skipped}"


def run_batch(args: argparse.Namespace) -> None:
    try:
        from tqdm import tqdm
    except ImportError as e:
        raise SystemExit("批量模式需要 tqdm：conda activate convtasnet && pip install tqdm") from e

    src = args.src.resolve()
    dst = args.dst.resolve()
    if not src.is_dir():
        raise SystemExit(f"--src 不是目录: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    audios = iter_libri_audio(src, args.splits)
    filtered_out = 0
    filter_root: Path | None = None
    if args.filter_librimix_small:
        filter_root = args.librimix_small_root.resolve()
    elif args.filter_librimix:
        filter_root = args.librimix_root.resolve()
    if filter_root is not None:
        allowed, n_mix = collect_librimix_utt_ids(filter_root)
        print(
            f"[batch] filter {filter_root}  "
            f"s1/s2 wav={n_mix}  unique utt={len(allowed)}"
        )
        kept: list[Path] = []
        for audio in audios:
            try:
                uid = parse_utt_id_from_wav(audio)
            except SystemExit:
                filtered_out += 1
                continue
            if uid not in allowed:
                filtered_out += 1
                continue
            kept.append(audio)
        audios = kept
    items: list[tuple[Path, str, Path, str]] = []
    skipped_existing = 0
    no_trans = 0
    for audio in audios:
        try:
            utt_id = parse_utt_id_from_wav(audio)
        except SystemExit:
            no_trans += 1
            continue
        trans = lookup_chapter_transcript(utt_id, audio.parent)
        if not trans:
            no_trans += 1
            continue
        ids_path, _artv_path, _tg_path, parent = dst_paths(src, dst, audio)
        if ids_path.is_file() and not args.overwrite:
            skipped_existing += 1
            continue
        items.append((audio, trans, parent, utt_id))
        if args.limit is not None and len(items) >= args.limit:
            break

    print(f"[batch] src={src}")
    print(f"[batch] dst={dst}")
    print(
        f"[batch] num_jobs={args.num_jobs}  "
        f"cpu_count={os.cpu_count()}  "
        f"(MFA 内部多进程；章节串行，不必再套 Python 多线程)"
    )
    print(
        f"[batch] audio_keep={len(audios)}  filtered_out={filtered_out}  "
        f"todo={len(items)}  skip_existing={skipped_existing}  no_transcript={no_trans}"
    )
    if not items:
        print("[batch] 没有待处理条目。")
        return

    mfa = find_mfa()
    print(f"[info] mfa = {mfa}")
    ensure_models(mfa, args.dictionary, args.acoustic, args.skip_download)
    artv_table = None if args.skip_artv else load_artv_table()
    arpa_inventory.dump_json_maps()
    shutil.copy2(arpa_inventory.DEFAULT_PHONE2ID_JSON, dst / "arpa_phone2id.json")
    shutil.copy2(arpa_inventory.DEFAULT_IPA_JSON, dst / "arpa_to_ipa.json")
    if arpa_inventory.DEFAULT_ARTV_NPY.is_file():
        shutil.copy2(arpa_inventory.DEFAULT_ARTV_NPY, dst / "arpa_artv.npy")

    by_chapter: dict[Path, list[tuple[Path, str, Path, str]]] = defaultdict(list)
    for it in items:
        by_chapter[it[0].parent].append(it)

    fail_log = dst / "_failed.txt"
    ok = failed = 0
    t0 = datetime.now()
    print(f"[batch] 开始 {t0.strftime('%Y-%m-%d %H:%M:%S')}  章节数={len(by_chapter)}")
    pbar = tqdm(total=len(items), desc="MFA Libri", unit="utt", dynamic_ncols=True, mininterval=0.5)

    for chap_dir, chap_items in by_chapter.items():
        pbar.set_description(f"MFA {chap_dir.name}")
        try:
            with tempfile.TemporaryDirectory(prefix="mfa_chap_") as tmp:
                tmp_p = Path(tmp)
                corpus = tmp_p / "corpus"
                mfa_out = tmp_p / "out"
                for audio, trans, _parent, utt_id in chap_items:
                    prepare_corpus(
                        audio, trans, corpus, args.target_sr, utt_id=utt_id, verbose=False
                    )
                align_corpus(
                    mfa,
                    corpus,
                    args.dictionary,
                    args.acoustic,
                    mfa_out,
                    args.num_jobs,
                )
                for audio, _trans, parent, utt_id in chap_items:
                    grids = list(mfa_out.rglob(f"{utt_id}.TextGrid"))
                    if not grids:
                        raise FileNotFoundError(f"{utt_id}.TextGrid")
                    export_alignment(
                        grids[0], parent, utt_id, artv_table, args.save_json
                    )
            ok += len(chap_items)
            pbar.update(len(chap_items))
            pbar.set_postfix_str(
                _eta_postfix(pbar, len(items), ok, failed, skipped_existing),
                refresh=False,
            )
        except Exception as e:
            tqdm.write(f"[warn] 章节失败 {chap_dir}：{type(e).__name__}: {e}；改单条")
            for audio, trans, parent, utt_id in chap_items:
                done = (parent / f"{utt_id}.phone_ids.npy").is_file()
                if done and not args.overwrite:
                    ok += 1
                    pbar.update(1)
                    continue
                try:
                    with tempfile.TemporaryDirectory(prefix="mfa_one_") as tmp:
                        tmp_p = Path(tmp)
                        corpus = tmp_p / "corpus"
                        mfa_out = tmp_p / "out"
                        prepare_corpus(
                            audio, trans, corpus, args.target_sr, utt_id=utt_id, verbose=False
                        )
                        tg = align_one(
                            mfa,
                            corpus / f"{utt_id}.wav",
                            corpus / f"{utt_id}.lab",
                            args.dictionary,
                            args.acoustic,
                            mfa_out,
                            verbose=False,
                        )
                        export_alignment(tg, parent, utt_id, artv_table, args.save_json)
                    ok += 1
                except Exception as e2:
                    failed += 1
                    msg = f"{audio.relative_to(src)}  {type(e2).__name__}: {e2}"
                    tqdm.write(f"FAIL {msg}")
                    with fail_log.open("a", encoding="utf-8") as f:
                        f.write(msg + "\n")
                pbar.update(1)
                pbar.set_postfix_str(
                    _eta_postfix(pbar, len(items), ok, failed, skipped_existing),
                    refresh=False,
                )

    pbar.close()
    t1 = datetime.now()
    print(
        f"[done] {t1.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"elapsed={t1 - t0}  ok={ok} fail={failed} skip={skipped_existing}"
    )
    print(f"[done] 输出: {dst}")
    if failed:
        print(f"[done] 失败列表: {fail_log}")


def main() -> None:
    args = parse_args()
    if args.batch:
        run_batch(args)
        return

    wav: Path = args.wav
    script_dir = Path(__file__).resolve().parent

    if args.skip_align:
        if args.out_dir is not None:
            out_dir = args.out_dir.resolve()
        elif wav.is_file():
            out_dir = (script_dir / "_mfa_out" / parse_utt_id_from_wav(wav)).resolve()
        else:
            out_dir = (script_dir / "_mfa_out").resolve()
        tg = find_existing_textgrid(out_dir)
        print(f"[info] skip-align，使用已有 TextGrid: {tg}")
        out_dir = tg.parent
    else:
        if not wav.is_file():
            raise SystemExit(f"音频不存在: {wav}")
        utt_id = parse_utt_id_from_wav(wav)
        out_dir = args.out_dir or (script_dir / "_mfa_out" / utt_id)
        out_dir = out_dir.resolve()
        transcript = args.transcript or lookup_librispeech_transcript(
            utt_id, args.librispeech_root
        )
        mfa = find_mfa()
        print(f"[info] mfa = {mfa}")
        ensure_models(mfa, args.dictionary, args.acoustic, args.skip_download)

        with tempfile.TemporaryDirectory(prefix="mfa_smoke_") as tmp:
            corpus = Path(tmp) / "corpus"
            prepare_corpus(wav, transcript, corpus, args.target_sr)
            prepared_wav = corpus / f"{utt_id}.wav"
            prepared_lab = corpus / f"{utt_id}.lab"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            tg = align_one(
                mfa,
                prepared_wav,
                prepared_lab,
                args.dictionary,
                args.acoustic,
                out_dir,
            )
            dest = out_dir / tg.name
            if tg.resolve() != dest.resolve():
                shutil.copy2(tg, dest)
                tg = dest

    print_alignment(tg)

    raw = tg.read_text(encoding="utf-8", errors="replace")
    phone_iv = parse_textgrid_intervals(raw, "phones")
    if not phone_iv:
        raise SystemExit("TextGrid 里没有 phones 层")

    artv_table = None if args.skip_artv else load_artv_table()
    rows = encode_phones(phone_iv, artv_table)
    print_phone_ids(rows, artv_table)
    save_phone_outputs(out_dir, rows, artv_table)
    print(f"\n[done] 完整输出目录: {out_dir}")


if __name__ == "__main__":
    main()
