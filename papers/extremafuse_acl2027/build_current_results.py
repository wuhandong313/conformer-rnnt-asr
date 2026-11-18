"""Rebuild the manuscript figures and tables from the selected full evaluations."""
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
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
import numpy as np


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
SELECTION = ROOT / "artifacts/research/phone69_paper_results_table_20261117/data.json"
OUTPUT = PAPER / "generated"
sys.path.insert(0, str(ROOT))
from papers.extremafuse_acl2027.method_figure import make_method_figure
from scripts.evaluation.analyze_phone69_speech_rate_groups import aggregate
NAMES = {
    "S1": "Stride 1", "S2": "Stride 2", "E09": "Stride 4",
    "E05": "Mel", "P_L1": "L1", "E02": "L2", "P_L3": "L3",
    "E06": "L5", "E07": "L7", "P_L9": "L9",
    "P_MEL_L3": "Mel+L3", "E10": "Mel+L5",
}
PAIRS = [("E06", "S1"), ("E06", "S2"), ("E07", "S2"), ("E07", "E09"),
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


def measured_matches(rows, comparisons):
    """Find observed nearest neighbours without interpolating between checkpoints."""
    fixed = {r["id"]: r for r in rows if not r["stage_rates"]}
    fusion = [r for r in rows if r["stage_rates"]]
    reference = fixed["E09"]
    best = min(fixed.values(), key=lambda r: (r[COST], r["phone_errors"]))
    if best["phone_errors"] != min(r["phone_errors"] for r in fixed.values()):
        raise ValueError("No single fixed stride minimizes both cost and error; review the figure.")

    def nearest(base, field):
        return min(fusion, key=lambda r: (abs(r[field] - base[field]), r["id"]))["id"]

    cheapest = next(p["a"] for p in comparisons
                    if p["selection"] == "baseline ceiling" and p["b"] == best["id"])
    return {"reference": reference["id"], "nearest_cost": nearest(reference, COST),
            "nearest_error": nearest(reference, "phone_errors"), "best_fixed": best["id"],
            "best_fixed_nearest_cost": nearest(best, COST), "best_fixed_per_limit": cheapest}


def make_figure(rows, edges, match, pairs):
    """Draw the measured pairs with a symbol key and comparison text below the plot."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    ink, muted, grid = "#22323a", "#465861", "#dfe5e6"
    colors = ("#4b5e6d", "#006c6b", "#a24619")
    markers = {"S1": "o", "S2": "d", "E09": "s", "E05": ">", "P_L1": "^", "E02": "P",
               "P_L3": "X", "E06": "o", "E07": "*", "P_L9": "s", "P_MEL_L3": "v", "E10": "D"}
    models = {r["id"]: r for r in rows}
    ref = models[match["reference"]]
    near_cost, near_error = models[match["nearest_cost"]], models[match["nearest_error"]]
    gap = next(p["delta_pp"] for p in pairs if (p["a"], p["b"]) == (near_cost["id"], ref["id"]))
    saving = 100 * (1 - near_error[COST] / ref[COST])

    fig = plt.figure(figsize=(6.36, 4.46), facecolor="white")
    ax = fig.add_axes([.105, .516, .867, .453])
    ax.set_yscale("log", base=2)
    ax.set(xlim=(4.70, 6.50), ylim=(3.5, 40), xticks=np.arange(4.75, 6.51, .25),
           yticks=[4, 8, 16, 32],
           yticklabels=["4", "8", "16", "32"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Test-clean phone error rate (%)", labelpad=3, color=ink, fontsize=9.2)
    ax.set_ylabel("Encoder + fusion\n(GMAC per 30 s)", color=ink, fontsize=9.2)
    ax.grid(axis="y", color=grid, linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.4, colors=muted, length=2.6, pad=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#8f9fa7")
        ax.spines[side].set_linewidth(.7)

    stop = max(r[ERROR] for r in rows)
    for family, style in (("fixed_stride", (0, (3, 2.4))), ("fusion", "-")):
        edge = edges[family]
        ax.step([r[ERROR] for r in edge] + [stop], [r[COST] for r in edge] + [edge[-1][COST]],
                where="post", linestyle=style, color="#9aaeb0", lw=.95, zorder=1.5)

    # The connector reaches both measured points. The arrowhead sits inside the segment
    # so it stays distinct from the L7 symbol and the neighbouring stride-2 symbol.
    for target, color in ((near_cost, colors[1]), (near_error, colors[2])):
        ax.plot([ref[ERROR], target[ERROR]], [ref[COST], target[COST]], color=color,
                lw=1.65, solid_capstyle="round", zorder=2)
        positions = [(ref[ERROR] + t*(target[ERROR]-ref[ERROR]),
                      ref[COST] * (target[COST]/ref[COST])**t) for t in (.48, .71)]
        ax.annotate("", xy=positions[1], xytext=positions[0],
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8,
                                "mutation_scale": 14, "shrinkA": 0, "shrinkB": 0}, zorder=2.5)

    handles = {}
    featured = {ref["id"], near_cost["id"], near_error["id"]}
    # Keep the almost cost-matched stride-2 and L7 symbols distinct at their
    # measured coordinates, including their outlines at the printed size.
    close_pair_sizes = {"S2": 5.6, "E07": 6.4}
    # The external detail repeats the measurements inside this box with a
    # wider horizontal scale. The two panels use the same log base.
    left, right, bottom, top = 4.74, 5.03, 16, 39
    outline = "#7c8d94"
    ax.add_patch(Rectangle((left, bottom), right-left, top-bottom, fill=False,
                           edgecolor=outline, linewidth=.7, linestyle=(0, (1.8, 2.4))))
    detail = fig.add_axes([.105, .257, .274, .130])
    detail.set_yscale("log", base=2)
    detail.set(xlim=(left, right), ylim=(bottom, top), xticks=[4.8, 4.9, 5.0],
               xticklabels=["4.8", "4.9", "5.0"], yticks=[16, 32], yticklabels=["16", "32"])
    detail.yaxis.set_minor_locator(NullLocator())
    detail.tick_params(labelsize=7.4, colors=muted, length=2.2, pad=2)
    detail.grid(axis="y", color=grid, linewidth=.55)
    detail.set_axisbelow(True)
    for spine in detail.spines.values():
        spine.set(color=outline, linewidth=.7, linestyle=(0, (1.8, 2.4)))
    for r in rows:
        key, n = r["id"], len(r["stage_rates"])
        color = colors[n]
        edge_width = 1.05 if key == "S2" else 1.3
        style = dict(marker=markers[key], markerfacecolor="white" if n == 0 else color,
                     markeredgecolor=color if n == 0 else "white",
                     markeredgewidth=edge_width if n == 0 else .65, linestyle="none")
        size = close_pair_sizes.get(key, 8 if key in featured else 6.4)
        point, = ax.plot(r[ERROR], r[COST], markersize=size, zorder=4+n, **style)
        if key == "S2":
            point.set_path_effects([path_effects.withStroke(linewidth=3.4, foreground="white")])
        if left <= r[ERROR] <= right and bottom <= r[COST] <= top:
            detail.plot(r[ERROR], r[COST], markersize=8 if key == "E07" else 6.3, **style)
        handles[key] = Line2D([], [], markersize=8 if markers[key] == "*" else 6.1, **style)

    # A single symbol/style definition drives both the data and its external key.
    fig.text(.242, .194, "Low-PER region, enlarged", ha="center", va="center",
             fontsize=8.3, color=ink)
    groups = [
        ("Fixed stride", .453, .621, .389, ["S1", "S2", "E09"], ["1", "2", "4"]),
        ("One site", .453, .621, .335, ["E05", "P_L1", "E02", "P_L3"], None),
        ("", .453, .621, .288, ["E06", "E07", "P_L9"], None),
        ("Two sites", .453, .621, .231, ["P_MEL_L3", "E10"], None),
    ]
    for family, text_x, key_x, y, ids, labels in groups:
        fig.text(text_x, y, family, va="center", fontsize=8.7, color=ink, weight="semibold")
        fig.legend([handles[k] for k in ids], labels or [NAMES[k] for k in ids],
                   loc="center left", bbox_to_anchor=(key_x, y), ncol=len(ids), frameon=False,
                   borderpad=0, borderaxespad=0, handlelength=.95, handletextpad=.33,
                   columnspacing=1.05, fontsize=8.6, labelcolor=ink)

    fig.add_artist(Line2D([.021, .977], [.162, .162], color=grid, lw=.75, transform=fig.transFigure))
    for x, color, heading, description, outcome in (
        (.021, colors[1], "←  Similar compute",
         f"{NAMES[ref['id']]} → {NAMES[near_cost['id']]}   ·   {ref[COST]:.2f} vs. {near_cost[COST]:.2f} GMAC",
         f"{ref[ERROR]:.3f}% → {near_cost[ERROR]:.3f}% PER  ({gap:+.3f} pp)"),
        (.515, colors[2], "↓  Similar PER",
         f"{NAMES[ref['id']]} → {NAMES[near_error['id']]}   ·   {ref[ERROR]:.3f}% vs. {near_error[ERROR]:.3f}%",
         f"{ref[COST]:.2f} → {near_error[COST]:.2f} GMAC  (−{saving:.1f}%)")):
        fig.text(x, .122, heading, va="center", weight="bold", fontsize=10.0, color=color)
        fig.text(x, .073, description, va="center", fontsize=8.4, color=muted)
        fig.text(x, .025, outcome.replace("-", "−"), va="center", weight="semibold",
                 fontsize=9.1, color=ink)

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
    match = measured_matches(rows, comparisons)
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
                  figure_comparisons=comparisons, measured_matches=match,
                  natural_rate_groups=rate_groups(rows))
    (OUTPUT / "analysis.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    make_figure(rows, edges, match, pairs)
    make_method_figure(next(x for x in rows if x["id"] == "E10"), PAPER / "figures")


if __name__ == "__main__":
    main()
