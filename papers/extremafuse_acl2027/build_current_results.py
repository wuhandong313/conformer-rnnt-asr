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
from matplotlib.lines import Line2D
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


def figure_comparisons(rows, pairs):
    """At each fixed baseline's observed PER, choose the least expensive eligible fusion."""
    models = {r["id"]: r for r in rows}
    intervals = {(p["a"], p["b"]): p for p in pairs}
    fusion = [r for r in rows if r["stage_rates"]]
    if len({r["reference_phones"] for r in rows}) != 1:
        raise ValueError("Matching on exact error counts requires the same reference phones")
    selected = []
    for base in (r for r in rows if not r["stage_rates"]):
        eligible = [r for r in fusion if r["phone_errors"] <= base["phone_errors"]]
        if not eligible:
            raise ValueError(f"No fusion system meets {base['id']}'s observed PER")
        chosen = min(eligible, key=lambda r: (r[COST], r["phone_errors"]))
        selected.append(("baseline ceiling", chosen, base))
    # This is a placement contrast, not a point chosen to meet a quality ceiling.
    selected.append(("earlier second site", models["P_MEL_L3"], models["E10"]))
    return [
        {**intervals[(a["id"], b["id"])], "selection": rule,
         "gmac_30s_a": a[COST], "gmac_30s_b": b[COST],
         "gmac_saved_percent": 100 * (1 - a[COST] / b[COST])}
        for rule, a, b in selected
    ]


def make_figure(rows, edge, comparisons):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 9.5, "pdf.fonttype": 42})
    ink, muted, grid = "#222c32", "#52616a", "#e0e5e6"
    colors = ["#37434b", "#006a72", "#994812"]
    shapes = ["s", "o", "D"]
    models = {r["id"]: r for r in rows}
    fig = plt.figure(figsize=(6.36, 4.53), facecolor="white")

    def heading(x, y, letter, title):
        fig.text(x, y, letter, va="center", weight="bold", fontsize=11, color=ink)
        fig.text(x + .041, y, title, va="center", weight="bold", fontsize=10.6, color=ink)

    def axis_style(ax):
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5, colors=muted, length=2.5, pad=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#9ba6aa")
            ax.spines[side].set_linewidth(.7)

    heading(.012, .958, "a", "All 12 operating points")
    ax = fig.add_axes([.108, .552, .864, .331])
    axis_style(ax)
    ax.grid(axis="y", color=grid, linewidth=.55)
    ax.step([r[COST] for r in edge] + [max(r[COST] for r in rows)],
            [r[ERROR] for r in edge] + [edge[-1][ERROR]], where="post",
            linestyle=(0, (3, 2.5)), color="#8d9b9d", lw=1.05, zorder=1)
    offsets = {"S1": (0, 8), "S2": (-7, 14), "E09": (0, 7), "E05": (1, 8),
               "P_L1": (0, -13), "E02": (2, 7), "P_L3": (0, 7),
               "E06": (0, -13), "E07": (3, -13), "P_L9": (0, 8),
               "P_MEL_L3": (2, 7), "E10": (0, -14)}
    for n in range(3):
        group = [r for r in rows if len(r["stage_rates"]) == n]
        ax.scatter([r[COST] for r in group], [r[ERROR] for r in group],
                   s=55 if n == 0 else 27, marker=shapes[n],
                   facecolor="white" if n == 0 else colors[n],
                   edgecolor=colors[n] if n == 0 else "white",
                   linewidth=1.25 if n == 0 else .5, zorder=3+n)
        for r in group:
            key = r["id"]
            leader = ({"arrowstyle": "-", "color": colors[n], "lw": .65,
                       "shrinkA": 1, "shrinkB": 3} if key in {"S2", "E07"} else None)
            ax.annotate(NAMES[key], (r[COST], r[ERROR]), offsets[key],
                        textcoords="offset points", ha="right" if key == "S2" else "center",
                        va="baseline", fontsize=8.4, color=colors[n], zorder=6,
                        arrowprops=leader)
    ax.set(xlim=(0, 37), ylim=(4.48, 6.70), xticks=np.arange(0, 36, 5),
           yticks=[4.8, 5.2, 5.6, 6.0, 6.4], ylabel="Test-clean PER (%)",
           xlabel="Encoder + fusion (GMAC per 30 s of audio)")
    ax.text(.991, .93, "↙  Lower left is better", transform=ax.transAxes, ha="right",
            va="top", color=muted, fontsize=8.3)
    handles = [Line2D([], [], marker=s, color=c, markerfacecolor="white" if i == 0 else c,
                      markersize=5, linestyle="none", label=name)
               for i, (s, c, name) in enumerate(zip(shapes, colors,
                                      ["Fixed stride", "One fusion site", "Two fusion sites"]))]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(.985, .958),
               ncol=3, frameon=False, fontsize=8.5, columnspacing=1,
               handlelength=1.0, handletextpad=.25, borderaxespad=0)

    heading(.012, .384, "b", "Compute savings and paired changes in phone error")
    bottom, height = .086, .195
    # Share row positions across the savings bars, the forest plot and the numeric labels.
    ys = [i+.38 for i in range(len(comparisons)-1, 0, -1)] + [.10]
    ylimits = (-.44, ys[0]+.55)
    bars = fig.add_axes([.298, bottom, .271, height])
    ci = fig.add_axes([.653, bottom, .225, height])
    for panel in (bars, ci):
        axis_style(panel)
        panel.spines["left"].set_visible(False)
        panel.set(ylim=ylimits, yticks=[])
    bars.set(xlim=(0, 100), xticks=[0, 50, 100])
    bars.axvline(0, color="#9ba6aa", lw=.7)
    ci.set(xlim=(-.40, .50), xticks=[-.4, 0, .4], xticklabels=["−0.4", "0", "+0.4"])
    ci.axvline(0, color=ink, linestyle=(0, (2.8, 2)), linewidth=.9, zorder=1)
    for x, label in ((.013, "Model vs. reference"), (.298, "GMAC saved (%)"),
                     (.653, "ΔPER (pp), 95% CI")):
        fig.text(x, .305, label, fontsize=8.8, color=ink, weight="bold", va="center")
    def row_y(y):
        return bottom + height * (y - ylimits[0]) / (ylimits[1] - ylimits[0])
    sep_y = row_y((ys[-2]+ys[-1])/2)
    fig.add_artist(Line2D([.013, .981], [sep_y, sep_y], transform=fig.transFigure,
                          color=grid, lw=.6, zorder=7))
    for row, y in zip(comparisons, ys):
        a, b = row["a"], row["b"]
        n = len(models[a]["stage_rates"])
        pct, delta = row["gmac_saved_percent"], row["delta_pp"]
        fig.text(.013, row_y(y), f"{NAMES[a]} vs. {NAMES[b]}", va="center", color=ink, fontsize=9)
        bars.barh(y, pct, height=.44, color=colors[n], zorder=3)
        bars.annotate(f"{pct:.1f}%", (pct, y), xytext=(3, 0), textcoords="offset points",
                      color=colors[n], fontsize=8.5, weight="semibold", va="center",
                      annotation_clip=False)
        ci.errorbar(delta, y, xerr=[[delta-row["lo"]], [row["hi"]-delta]], fmt=shapes[n],
                    color=colors[n], elinewidth=1.25, capsize=2.5, capthick=1.1,
                    markersize=4.9, markeredgewidth=1.1, zorder=3,
                    markerfacecolor="white" if row["lo"] <= 0 <= row["hi"] else colors[n])
        fig.text(.98, row_y(y), f"{delta:+.3f}".replace("-", "−"), ha="right",
                 va="center", fontsize=8.7, color=ink)
    fig.text(.013, .018, "Open point in b: interval includes zero.", fontsize=8,
             color=muted, va="center")
    fig.text(.653, .018, "← Lower error", fontsize=8, color=muted, va="center")
    fig.text(.976, .018, "Higher error →", fontsize=8, color=muted, va="center", ha="right")
    output = PAPER / "figures"
    fig.savefig(output / "quality_budget_30s.pdf",
                metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None})
    fig.savefig(output / "quality_budget_30s.png", dpi=250, facecolor="white")
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
    comparisons = figure_comparisons(rows, pairs)
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
                  figure_comparisons=comparisons,
                  natural_rate_groups=rate_groups(rows))
    (OUTPUT / "analysis.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    make_figure(rows, edge, comparisons)


if __name__ == "__main__":
    main()
