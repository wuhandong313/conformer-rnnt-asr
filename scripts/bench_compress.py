"""llm_compress 压缩率-质量曲线 bench (Qwen3.5-0.8B, needle/gist/short 三类探针)。

每个配置 × 每个样本: 单 forward 取 keep 统计 → generate 出答案 → 判命中。
配置族:
配置命名与 compression.py 模式对齐 (原法族统一 distance=2):
  none              旁路基线 (= compression.py "none", 未压缩)
  random_50         随机子集对照 (同保留率下界)
  naive             帧差原始 find_peaks 峰谷 (纯选择)
  hanning           同上 + 固定 Hanning 窗加权混合
  fir               FIR 低通粗检 ±2 精修 (纯选择)
  fir_hanning       fir + Hanning 混合
  savgol            Savitzky-Golay 平滑峰谷 (纯选择)
  savgol_hanning    savgol + Hanning 混合
  diffq_f           帧差幅值分位阈值 (保序推广, keep_frac 直控压缩率)
  novelty_f         局部窗口最近邻距离 (非原法, 对照消融)

输出: 终端表 + CSV (artifacts/evaluations/bench_compress_results.csv)。
用法: /root/.../python scripts/bench_compress.py [--limit N] [--max-new 32]
本地: python scripts/bench_compress.py --limit 2 先冒烟。
"""

import argparse
import csv
import random
import sys
import time
import types
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_compress import TokenSaliencyCompressor, install_compression  # noqa: E402

MODEL_DIR = "qwen3.5-0.8b"

FILLERS = [
    "山城的清晨总被雾气包裹，渡船五点就离岸了。",
    "档案室的木柜第三层放着一叠未编号的勘探图。",
    "雨季修路工程按惯例在四月前完成路基验收。",
    "老会计习惯用铅笔复账，红色印泥放在抽屉右侧。",
    "码头工人换班时间是中午十二点整，误差不超过五分钟。",
    "地质队的电台每周一与省局例行通报一次。",
    "仓库盘点采用双人复核制，盘单留存三年。",
    "县城通往矿区的班车每天两趟，末班十六点发出。",
    "值班表贴在走廊尽头，字迹被雨水洇开过。",
    "器材室的钥匙由夜班保管员统一收管。",
]

NEEDLE_PROBES = [
    dict(
        fact="2031年3月17日，林巧薇在赣州港签下了三号探矿合同。",
        q="档案记录中，林巧薇签下三号探矿合同的日期是哪一天？只回答日期。",
        expect=["2031", "17"],
    ),
    dict(
        fact="样品编号K-774的稀土矿脉含氧化铈百分之十九点六。",
        q="记录里样品K-774的氧化铈含量是多少？只回答含量。",
        expect=["十九点六", "774"],
    ),
    dict(
        fact="押运员老章习惯把通行证缝在蓝色工装内衬左胸位置。",
        q="老章把通行证缝在什么地方？回答关键位置词。",
        expect=["左胸"],
    ),
    dict(
        fact="探矿队于癸卯年冬至前完成了第七机井的固井作业。",
        q="第七机井固井作业在什么时间之前完成？回答时间词。",
        expect=["冬至"],
    ),
]

GIST_PROBES = [
    dict(
        gist_fact="矿区的规矩是逢双周五全库断电检修。" * 3,
        q="这段记录反复强调的矿区规矩是什么？简短回答。",
        expect=["双周五", "断电"],
        mode="any",
    ),
    dict(
        gist_fact="所有勘探图纸必须经两位工程师会签才能归档。" * 3,
        q="记录的归档要求核心是什么？简短回答。",
        expect=["会签", "两"],
        mode="any",
    ),
    dict(
        gist_fact="雨季期间班车停运，物资靠骡队转运。" * 4,
        q="雨季的运输安排是什么？简短回答。",
        expect=["骡队", "停运"],
        mode="any",
    ),
    dict(
        gist_fact="新工人上岗第一课永远是安全检查。" * 4,
        q="这段文字的主旨是什么？简短回答。",
        expect=["安全检查", "上岗"],
        mode="any",
    ),
]

SHORT_PROBES = [
    dict(q="1+1等于几？只答数字。", expect=["2"]),
    dict(q="用三个字形容今天的天气（自由发挥）。", expect=[]),
    dict(q="请说一种矿物的名字，只答名字。", expect=[]),
]

MIN_LEN = 64


def _range_tail(L: int, k: int = 16):
    return range(L - k, L)


def build_needle_ctx(fact: str, seed: int) -> str:
    rnd = random.Random(seed)
    parts = []
    for i in range(14):
        parts.append("".join(rnd.sample(FILLERS, 3)))
    parts[5] = fact + parts[5]
    return "".join(parts)


def build_gist_ctx(gist: str, seed: int) -> str:
    rnd = random.Random(seed + 100)
    parts = []
    for i in range(8):
        parts.append("".join(rnd.sample(FILLERS[:6] + FILLERS[6:], 2)))
    parts[3] = gist + parts[3]
    parts[6] = gist + parts[6]
    return "".join(parts)


def hit_check(text: str, expect, mode: str) -> bool:
    if not expect:
        return True  # 开放题只记录不断言
    return all(e in text for e in expect) if mode == "all" else any(
        e in text for e in expect
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每类探针只跑前 N 个")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = args.device
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.float32
    ).to(dev).eval()

    # 命名与 compression.py 模式对齐; 原法族统一 distance=2
    configs = [
        ("none", None),  # compression.py 旁路模式 = 未压缩基线
        ("random_50", dict(kind="random", keep_frac=0.5)),
        ("naive", dict(mode="naive", distance=2)),
        ("hanning", dict(mode="hanning", distance=2)),
        ("fir", dict(mode="fir", distance=2)),
        ("fir_hanning", dict(mode="fir_hanning", distance=2)),
        ("savgol", dict(mode="savgol", distance=2)),
        ("savgol_hanning", dict(mode="savgol_hanning", distance=2)),
        ("diffq_50", dict(mode="diffq", keep_frac=0.5)),
        ("diffq_85", dict(mode="diffq", keep_frac=0.85)),
        ("novelty_50", dict(mode="novelty", keep_frac=0.5)),
    ]

    probes = []
    for i, p in enumerate(NEEDLE_PROBES):
        ctx = build_needle_ctx(p["fact"], i)
        probes.append(
            dict(kind="needle", prompt=f"以下是档案节选。{ctx}\n问题：{p['q']}",
                 expect=p["expect"], mode="all")
        )
    for i, p in enumerate(GIST_PROBES):
        ctx = build_gist_ctx(p["gist_fact"], i)
        probes.append(
            dict(kind="gist", prompt=f"以下记录重复较多。{ctx}\n问题：{p['q']}",
                 expect=p["expect"], mode="any")
        )
    for p in SHORT_PROBES:
        probes.append(dict(kind="short", prompt=p["q"], expect=p["expect"], mode="all"))
    if args.limit:
        groups = {}
        for pr in probes:
            groups.setdefault(pr["kind"], [])
            if len(groups[pr["kind"]]) < args.limit:
                groups[pr["kind"]].append(pr)
        probes = [pr for g in groups.values() for pr in g]

    rows = []
    for name, spec in configs:
        # 确定性配置: 每 config 装一次钩子; random 对照: 每探针装一次 (keep 集合按样本生成)
        comp_static = None
        handles_static = []
        if spec is not None and spec.get("kind") != "random":
            comp_static = TokenSaliencyCompressor(min_len=MIN_LEN, **spec)
            handles_static = install_compression(model, comp_static, after_layer=3)

        for pr in probes:
            # 与 infer.py 同路径: chat template 包裹, 避免裸 prompt 复读
            enc = tok.apply_chat_template(
                [{"role": "user", "content": pr["prompt"]}],
                tokenize=True, return_dict=True, add_generation_prompt=True,
                enable_thinking=False,
            )
            fi = enc["input_ids"]
            if hasattr(fi, "tolist"):
                fi = fi.tolist()
            full = [int(x) for x in (fi[0] if fi and isinstance(fi[0], list) else fi)]
            L = len(full)
            ids = torch.tensor([full], device=dev)

            comp = comp_static
            handles = handles_static
            if spec is not None and spec.get("kind") == "random":
                comp = TokenSaliencyCompressor(min_len=MIN_LEN)
                k = max(18, int(L * spec["keep_frac"]))

                def patched_select(hidden, _L=L, _k=k):
                    Lh = hidden.size(1)
                    if Lh != _L or Lh <= MIN_LEN:  # decode 步与短样本透传
                        return torch.arange(Lh, device=hidden.device)
                    sel = {0, *_range_tail(_L)} | set(
                        random.sample(range(_L - 16), max(0, _k - 17))
                    )
                    return torch.tensor(sorted(sel), device=hidden.device)

                comp.select = types.MethodType(
                    lambda self, h: patched_select(h), comp
                )
                handles = install_compression(model, comp, after_layer=3)

            random.seed(hash(pr["prompt"]) % (2**32))
            t0 = time.time()
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
                keep_n = (
                    comp.keep.numel()
                    if comp is not None and comp.keep is not None
                    else L
                )
                out = model.generate(
                    ids, max_new_tokens=args.max_new, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            wall = time.time() - t0
            if spec is not None and spec.get("kind") == "random":
                for h in handles:
                    h.remove()

            text = tok.decode(out[0][L:], skip_special_tokens=True).strip()
            hit = hit_check(text, pr["expect"], pr["mode"])
            rows.append(
                dict(config=name, probe=pr["kind"], L=L, kept=keep_n,
                     ratio=round(keep_n / L, 3), hit=int(hit), wall=round(wall, 1),
                     answer=text.replace("\n", " ")[:60])
            )
            print(f"  {name:16s} {pr['kind']:6s} kept={keep_n}/{L}({keep_n/L:.0%}) hit={int(hit)} → {text[:24]}", flush=True)

        for h in handles_static:
            h.remove()
        print(f"[config {name} done]", flush=True)

    out_csv = "artifacts/evaluations/bench_compress_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("\n===== 汇总 (hit% by probe kind) =====")
    print(f"{'config':18s} {'needle':>6s} {'gist':>6s} {'short':>6s} {'avg ratio(needle+gist)':>22s}")
    for name, _ in configs:
        sub = {k: [r for r in rows if r["config"] == name and r["probe"] == k]
               for k in ("needle", "gist", "short")}
        line = f"{name:18s}"
        for k in ("needle", "gist", "short"):
            hits = sum(r["hit"] for r in sub[k])
            line += f" {hits}/{len(sub[k]):<4d}"
        ratios = [r["ratio"] for r in rows if r["config"] == name and r["probe"] in ("needle", "gist")]
        line += f" {'':6s}{sum(ratios)/max(1,len(ratios)):.2f}"
        print(line)
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
