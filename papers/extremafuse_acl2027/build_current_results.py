"""Rebuild the manuscript figure and tables from the selected full evaluations."""
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
SELECTION = ROOT / "artifacts/research/phone69_paper_results_table_20261117/data.json"
OUTPUT = PAPER / "generated"
sys.path.insert(0, str(ROOT))
from scripts.evaluation.analyze_phone69_speech_rate_groups import aggregate
NAMES = {
    "S1": "Stride 1", "S2": "Stride 2", "E09": "Stride 4",
    "E05": "Mel", "P_L1": "L1", "E02": "L2", "P_L3": "L3",
    "E06": "L5", "E07": "L7", "P_L9": "L9",
    "P_MEL_L3": "Mel+L3", "E10": "Mel+L5",
}
PAIRS = [("E06", "S1"), ("E06", "S2"), ("E07", "S2"),
         ("E10", "E09"), ("E10", "E05"), ("P_MEL_L3", "E10")]
COST = "encoder_compression_gmac_30s"
ERROR = "per_percent"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate_groups(models):
    previous = ROOT / "artifacts/research/phone69_supplementary_rate_groups_20261112/results.json"
    frozen = json.loads(previous.read_text())["source_sha256"]
    bins = [[], [], []]
    for decile in range(1, 11):
        path = ROOT / f"speech_rate_analysis/deciles/decile_{decile:02}.txt"
        if sha(path) != frozen[str(path.relative_to(ROOT))]:
            raise ValueError("The original speaking-rate group changed")
        bins[0 if decile <= 3 else 1 if decile <= 7 else 2].extend(
            line.split()[0] for line in path.read_text().splitlines() if line.strip())
    flat = sum(bins, [])
    if len(set(flat)) != len(flat):
        raise ValueError("Speaking-rate groups overlap")
    rendered, groups = [], {}
    for x in models:
        sources = x["sources"]
        csv_path, = [p for p in sources if p.endswith("/utterances.csv")]
        with (ROOT / csv_path).open(newline="", encoding="utf-8") as f:
            predictions = {r["uid"]: r for r in csv.DictReader(f)}
        counts = predictions
        if "encoder_weights" not in next(iter(predictions.values())):
            count_path, = [p for p in sources if "/full/" in p and p.endswith(".json")]
            if sha(ROOT / count_path) != sources[count_path]:
                raise ValueError("Mismatch in archived decoder count")
            payload = json.loads((ROOT / count_path).read_text())
            if payload["checkpoint_sha256"] != x["checkpoint_sha256"]:
                raise ValueError("Count and prediction checkpoints differ")
            counts = {r["uid"]: r for r in payload["rows"]}
        if set(flat) != predictions.keys() or set(flat) != counts.keys():
            raise ValueError("Missing predictions or counts for a rate group")
        parts = [aggregate(predictions, counts, uids) for uids in bins]
        rate = sum(g["total_major_macs"] for g in parts) / (1e9 * sum(g["audio_seconds"] for g in parts))
        if sum(g["phone_errors"] for g in parts) != x["phone_errors"] or not math.isclose(
                rate, x["actual_beam20_gmac_per_audio_s"]["total"], rel_tol=1e-10):
            raise ValueError("Speaking-rate groups do not reproduce corpus scores")
        uid = x["id"]
        groups[uid] = dict(zip(("slow", "middle", "fast"), parts))
        if uid in ("E05", "P_MEL_L3"):
            rendered.append(r"\midrule")
        fields = [NAMES[uid]]
        for g in parts:
            fields.extend([f"{100*g['corpus_PER']:.3f}", f"{g['total_gmac_per_audio_second']:.3f}"])
        rendered.append(" & ".join(fields) + r" \\")
    write_rows("speaking_rate.tex", rendered)
    return groups


def trials(rows):
    """Align utterances and resample speakers jointly, keeping clusters intact."""
    counts, reference = {}, None
    for model in rows:
        uid = model["id"]
        csv_path, = [p for p in model["sources"] if p.endswith("/utterances.csv")]
        path = ROOT / csv_path
        if sha(path) != model["sources"][csv_path]:
            raise ValueError(f"{uid}: utterance archive hash changed")
        if model["checkpoint_selection"] != "dev-clean validation loss":
            raise ValueError(f"{uid}: checkpoint was not selected on validation loss")
        targets, errors = {}, {}
        with path.open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                key = r["uid"]
                if key in targets:
                    raise ValueError(f"{uid}: repeated utterance {key}")
                targets[key] = (int(r["reference_phones"]), r["ref"])
                speaker = int(key.split("-")[0])
                errors[speaker] = errors.get(speaker, 0) + int(r["phone_errors"])
        if reference is None:
            reference = targets
        if targets != reference or len(targets) != model["utterances"]:
            raise ValueError(f"{uid}: reference texts or utterance set differ")
        if sum(errors.values()) != model["phone_errors"]:
            raise ValueError(f"{uid}: inconsistent total errors")
        if sum(r[0] for r in targets.values()) != model["reference_phones"]:
            raise ValueError(f"{uid}: inconsistent total reference length")
        counts[uid] = errors

    speakers = sorted(counts[rows[0]["id"]])
    phones = dict.fromkeys(speakers, 0)
    for key, (length, _) in reference.items():
        phones[int(key.split("-")[0])] += length
    n = len(speakers)
    draws = np.random.default_rng(42).integers(n, size=(10_000, n))
    den = np.array([phones[s] for s in speakers])
    den_draw = den[draws].sum(axis=1)
    result = []
    for a, b in PAIRS:
        delta = np.array([counts[a][s] - counts[b][s] for s in speakers])
        per_draw = 100 * delta[draws].sum(axis=1) / den_draw
        lo, hi = np.percentile(per_draw, [2.5, 97.5])
        result.append(dict(a=a, b=b, delta_pp=100 * delta.sum().item() / den.sum().item(),
                           lo=float(lo), hi=float(hi)))
    return result, n


def frontier(rows):
    best = float("inf")
    result = []
    for r in sorted(rows, key=lambda m: (m[COST], m[ERROR])):
        if r[ERROR] < best:
            result.append(r)
            best = r[ERROR]
    return result


def make_figure(rows, edge):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.labelsize": 10.5, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.40, 2.86))
    fig.subplots_adjust(left=.105, right=.985, top=.865, bottom=.195)
    colors = ["#303541", "#17665e", "#a84a16"]
    symbols = ["s", "o", "D"]
    legends = ["Fixed stride", "One fusion site", "Two fusion sites"]
    offsets = {"S1": (0, 8), "S2": (9, 10), "E09": (0, 7), "E05": (2, 8),
               "P_L1": (0, -16), "E02": (6, 7), "P_L3": (0, 6),
               "E06": (0, -15), "E07": (-12, -14), "P_L9": (-5, 7),
               "P_MEL_L3": (5, 8), "E10": (0, -16)}
    xmax = 37
    ax.step([r[COST] for r in edge] + [xmax], [r[ERROR] for r in edge] + [edge[-1][ERROR]],
            where="post", linestyle=(0, (3, 2.6)), linewidth=1.15, color="#85958f",
            zorder=1)
    for family in range(3):
        rr = [r for r in rows if len(r["stage_rates"]) == family]
        ax.scatter([r[COST] for r in rr], [r[ERROR] for r in rr],
                   s=40, marker=symbols[family], facecolor=colors[family],
                   edgecolor="white", linewidth=.65, label=legends[family], zorder=4)
        for r in rr:
            dx, dy = offsets[r["id"]]
            ax.annotate(NAMES[r["id"]], (r[COST], r[ERROR]), (dx, dy),
                        textcoords="offset points", ha="center" if r["id"] != "S2" else "left",
                        fontsize=9, color=colors[family],
                        arrowprops=({"arrowstyle": "-", "color": colors[family], "lw": .6,
                                     "shrinkA": 1, "shrinkB": 3} if r["id"] in {"S2", "E07"} else None))
    ax.set(xlim=(1, xmax), ylim=(4.47, 6.65),
           xticks=[5, 10, 15, 20, 25, 30, 35], yticks=[4.75, 5.25, 5.75, 6.25],
           xlabel="Encoder + compression (GMAC per 30 s of input)", ylabel="Test-clean PER (%)")
    ax.spines[["right", "top"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#81878b")
    ax.tick_params(length=3, color="#81878b")
    ax.grid(axis="y", color="#e1e4e3", lw=.65)
    ax.legend(loc="lower center", bbox_to_anchor=(.50, 1.025), ncol=3, frameon=False,
              borderaxespad=0, columnspacing=1.5, handlelength=1.1, handletextpad=.35)
    fig.savefig(PAPER / "figures/quality_budget_30s.pdf",
                metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def write_rows(name, rows):
    macro = {"operating_points.tex": "OperatingRows", "actual_decode.tex": "DecodeRows",
             "paired_per.tex": "PairedRows", "speaking_rate.tex": "RateRows"}[name]
    (OUTPUT / name).write_text("% Generated by build_current_results.py\n" +
                               r"\newcommand{\%s}{%%" % macro + "\n" +
                               "\n".join(rows) + "%\n}\n", encoding="utf-8")


def main():
    data = json.loads(SELECTION.read_text(encoding="utf-8"))
    if sha(ROOT / data["source"]) != data["source_sha256"]:
        raise ValueError("Selection does not match the full evaluation archive")
    rows = data["models"]
    if ([m["id"] for m in rows] != data["selected_ids"] or
            set(data["selected_ids"]) != set(NAMES) or data["duration_seconds"] != 30):
        raise ValueError("Review the manuscript labels and contrasts for the new selection")
    pairs, n = trials(rows)
    edge = frontier(rows)
    OUTPUT.mkdir(exist_ok=True)
    summary, decode = [], []
    for x in rows:
        i = x["id"]
        key = NAMES[i]
        if i in ("E05", "P_MEL_L3"):
            summary.append(r"\midrule")
            decode.append(r"\midrule")
        per = f"{x[ERROR]:.4f}"
        cost = f"{x[COST]:.3f}"
        if x[ERROR] == min(m[ERROR] for m in rows):
            per = r"\textbf{" + per + "}"
        if x[COST] == min(m[COST] for m in rows):
            cost = r"\textbf{" + cost + "}"
        summary.append(" & ".join([key, str(x["backbone"]),
                                   f"{x['trace_30s']['output_frames']:,}", per, cost,
                                   f"{x['encoder_compression_gmac_per_audio_s_30s']:.3f}"]) + r" \\")
        c = x["actual_beam20_gmac_per_audio_s"]
        step = int(re.search(r"step=(\d+)", x["checkpoint"])[1])
        decode.append(" & ".join([key, f"{step:,}"] +
                                 [f"{c[j]:.4f}" for j in ("encoder_compression", "predictor",
                                                          "joiner", "total")]) + r" \\")
    write_rows("operating_points.tex", summary)
    write_rows("actual_decode.tex", decode)
    write_rows("paired_per.tex", [f"{NAMES[p['a']]} / {NAMES[p['b']]} & "
                                  f"${p['delta_pp']:+.3f}$ & "
                                  f"$[{p['lo']:+.3f}, {p['hi']:+.3f}]$" + r" \\" for p in pairs])
    record = dict(selection=str(SELECTION.relative_to(ROOT)), sha256=sha(SELECTION),
                  paired_unit="speaker", speakers=n, draws=10_000, seed=42,
                  interval="percentile; exploratory, not adjusted for multiple comparisons",
                  observed_frontier=[r["id"] for r in edge], contrasts=pairs,
                  natural_rate_groups=rate_groups(rows))
    (OUTPUT / "analysis.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    make_figure(rows, edge)


if __name__ == "__main__":
    main()
