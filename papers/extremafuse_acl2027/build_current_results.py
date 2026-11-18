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
from matplotlib.ticker import NullLocator
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
    """Nondominated observations, in increasing cost and decreasing PER."""
    best = float("inf")
    result = []
    for r in sorted(rows, key=lambda m: (m[COST], m[ERROR])):
        if r[ERROR] < best:
            result.append(r)
            best = r[ERROR]
    return result


def cost_at_error_ceilings(rows, ceilings):
    """Find each family's minimum compute at the specified observed PER ceilings."""
    families = {"fixed_stride": [r for r in rows if not r["stage_rates"]],
                "fusion": [r for r in rows if r["stage_rates"]]}
    edges = {k: list(reversed(frontier(group))) for k, group in families.items()}
    # Use the original counts; rounded table entries are only for display.
    examples = []
    for ceiling in ceilings:
        best = {k: min((r for r in group
                        if 100 * r["phone_errors"] <= ceiling * r["reference_phones"]),
                       key=lambda r: (r[COST], r[ERROR]))
                for k, group in families.items()}
        base, fusion = best["fixed_stride"], best["fusion"]
        examples.append({"per_ceiling_percent": ceiling, "fixed_id": base["id"],
                         "fusion_id": fusion["id"], "fixed_gmac_30s": base[COST],
                         "fusion_gmac_30s": fusion[COST], "fixed_over_fusion": base[COST] / fusion[COST]})
    return edges, examples


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


def make_figure(rows, edges, comparisons):
    """Plot the complete trade-off and show both magnitudes for one named pair."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.unicode_minus": True, "pdf.fonttype": 42})
    ink, muted, grid = "#1f3038", "#485a63", "#dce3e5"
    fixed, single, double = "#51636f", "#007673", "#aa4c1e"
    colors, shapes = (fixed, single, double), ("s", "o", "D")
    models = {r["id"]: r for r in rows}
    contrast = next(p for p in comparisons if p["a"] == "E10" and p["b"] == "E09")
    ref, fusion = models[contrast["b"]], models[contrast["a"]]
    ratio = ref[COST] / fusion[COST]
    fig = plt.figure(figsize=(6.36, 3.20), facecolor="white")

    def heading(x, letter, title):
        fig.text(x, .951, f"({letter})  {title}", family="STIXGeneral", fontsize=11.8,
                 weight="bold", color=ink, va="center")

    heading(.014, "a", "Compute at a target PER")
    heading(.586, "b", f"Stride 4 costs {ratio:.2f}× as much")
    ax = fig.add_axes([.089, .231, .443, .604])
    ax.set_yscale("log", base=2)
    ax.set(xlim=(4, 7), ylim=(3.25, 44), xticks=[4, 5, 6, 7], yticks=[4, 8, 16, 32],
           yticklabels=["4", "8", "16", "32"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Test-clean PER (%)", labelpad=5, color=ink, fontsize=9.4)
    ax.set_ylabel("GMAC per 30 s", labelpad=3, color=ink, fontsize=9.4)
    ax.grid(axis="y", color=grid, linewidth=.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.7, colors=muted, length=2.7, pad=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#9ca9ae")
        ax.spines[side].set_linewidth(.65)

    stop = max(r[ERROR] for r in rows)
    for family, color, style in (("fixed_stride", fixed, (0, (3.5, 2.8))),
                                 ("fusion", single, "-")):
        edge = edges[family]
        ax.step([r[ERROR] for r in edge] + [stop], [r[COST] for r in edge] + [edge[-1][COST]],
                where="post", linestyle=style, color=color, lw=1.05, alpha=.83, zorder=1.5)
    # Only the labels move. The two near-coincident L7 and stride-2 markers are drawn at
    # their original measurements, with the smaller circle visible inside the open square.
    offsets = {"S1": (5, 6), "S2": (9, -6), "E09": (6, 6), "E05": (6, 5),
               "P_L1": (-6, 0), "E02": (6, 0), "P_L3": (6, 3), "E06": (-6, -3),
               "E07": (-8, 0), "P_L9": (-6, 5), "P_MEL_L3": (5, 6), "E10": (-7, 0)}
    handles = []
    for n, (color, shape, label) in enumerate(zip(colors, shapes,
                                                  ("Fixed stride", "One site", "Two sites"))):
        group = [r for r in rows if len(r["stage_rates"]) == n]
        ax.scatter([r[ERROR] for r in group], [r[COST] for r in group], s=49 if n == 0 else 30,
                   marker=shape, facecolor="white" if n == 0 else color,
                   edgecolor=color if n == 0 else "white", linewidth=1.25 if n == 0 else .6,
                   zorder=3+n)
        for r in group:
            dx, dy = offsets[r["id"]]
            leader = ({"arrowstyle": "-", "color": color, "lw": .6, "shrinkA": 1, "shrinkB": 3}
                      if r["id"] in ("S2", "E07") else None)
            ax.annotate(NAMES[r["id"]], (r[ERROR], r[COST]), (dx, dy), textcoords="offset points",
                        va="center", ha="left" if dx > 0 else "right",
                        color=color, fontsize=8.1, zorder=6, arrowprops=leader)
        handles.append(Line2D([], [], linestyle="none", marker=shape, markeredgecolor=color,
                              markerfacecolor="white" if n == 0 else color, markersize=5, label=label))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.29, .046), ncol=3, frameon=False,
               handlelength=.9, handletextpad=.32, columnspacing=1.05, fontsize=8.2)

    def paired_bars(bottom, title, field, maximum, ticks, precision):
        # Both bars in a metric have a common zero. The values have their own column so the
        # nearly equal error rates can be compared at print size.
        left, width, height = .718, .196, .200
        panel = fig.add_axes([left, bottom, width, height])
        panel.set(xlim=(0, maximum), ylim=(-.57, 1.55), yticks=[])
        panel.set_xticks(ticks, [f"{t:g}" for t in ticks])
        panel.spines[["left", "right", "top"]].set_visible(False)
        panel.spines["bottom"].set_color("#aab5b9")
        panel.spines["bottom"].set_linewidth(.65)
        panel.tick_params(axis="x", colors=muted, labelsize=8, length=2.3, pad=2)
        panel.axvline(0, lw=.7, color="#50616a", zorder=1)
        fig.text(.588, bottom+height+.047, title, fontsize=9.5, weight="semibold", color=ink)
        for y, model, color in ((1, ref, fixed), (0, fusion, double)):
            value = model[field]
            panel.barh(y, value, height=.39, color=color, zorder=2)
            fy = bottom + height * (y+.57) / 2.12
            fig.text(.588, fy, NAMES[model["id"]], va="center", color=ink, fontsize=8.8)
            fig.text(.989, fy, f"{value:.{precision}f}", ha="right", va="center",
                     fontsize=9.1, weight="semibold", color=color)

    paired_bars(.558, "Encoder + fusion (GMAC / 30 s)", COST, 30, [0, 15, 30], 2)
    paired_bars(.183, "Test-clean PER (%)", ERROR, 7, [0, 3.5, 7], 3)
    output = PAPER / "figures"
    fig.savefig(output / "quality_budget_30s.pdf",
                metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None})
    fig.savefig(output / "quality_budget_30s.png", dpi=360, facecolor="white")
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
    edges, examples = cost_at_error_ceilings(rows, (5.0, 5.7, 6.1))
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
                  minimum_cost_at_per_ceiling={k: [r["id"] for r in e] for k, e in edges.items()},
                  example_per_ceilings=examples,
                  figure_comparisons=comparisons,
                  natural_rate_groups=rate_groups(rows))
    (OUTPUT / "analysis.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    make_figure(rows, edges, comparisons)


if __name__ == "__main__":
    main()
