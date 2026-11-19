"""Rebuild the manuscript figures and tables from the selected full evaluations."""
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
MANUSCRIPT_EXCLUSIONS = {"S1", "P_L9"}
PAIRS = [("E05", "E09"), ("E06", "S2"), ("E07", "S2"), ("E07", "E09"),
         ("E10", "E09"), ("E10", "E05"), ("P_MEL_L3", "E10")]
COST = "encoder_compression_gmac_30s"
ERROR = "per_percent"
NARROW = "E01"
TABLE_STRIDES = {NARROW: 4, "E09": 4, "S2": 2}
REPORT_NAMES = {**NAMES, NARROW: "Stride 4 (128)", "E09": "Stride 4 (512)"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_rows(selection=None):
    """Load the November 17 selection and the archived narrow stride-4 control."""
    if selection is None:
        selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    source = ROOT / selection["source"]
    if sha(source) != selection["source_sha256"] or selection["duration_seconds"] != 30:
        raise ValueError("Review the evaluation archive and reference duration")
    archived = {r["id"]: r for r in json.loads(source.read_text(encoding="utf-8"))["models"]}
    ids = selection["selected_ids"]
    narrow = archived[NARROW]
    if (len(ids) != len(NAMES) or set(ids) != set(NAMES) or NARROW in ids or
            narrow["backbone"] != 128 or narrow["stage_rates"]):
        raise ValueError("Review the fixed-stride control after changing the selection")
    rows = [archived[key] for key in ids]
    if rows != selection["models"]:
        raise ValueError("The selected operating points differ from the full evaluation archive")
    rows.append(narrow)
    if len({(r["utterances"], r["reference_phones"], r["checkpoint_selection"]) for r in rows}) != 1:
        raise ValueError("The evaluations must use the same corpus and checkpoint selection")
    return rows


def equal_length_outputs(input_frames, fixed_steps, fusion_steps):
    """Compare the two front ends on utterances of exactly the same Mel length."""
    by_length = defaultdict(list)
    for uid, length in input_frames.items():
        if fixed_steps[uid] != (length + 3) // 4:
            raise ValueError(f"Stride-4 stacking disagrees with the recorded input length for {uid}")
        by_length[length].append(fusion_steps[uid])
    repeated = {length: steps for length, steps in by_length.items() if len(steps) >= 2}
    if not repeated:
        raise ValueError("No repeated input lengths for a comparison of the front ends")
    # Pick the most frequent input length without inspecting the fusion outputs.
    example = min(repeated, key=lambda length: (-len(repeated[length]), length))
    steps = repeated[example]
    return dict(input_reference="P_L1", input_field="input_mel_frames", fixed="E09", fusion="E05",
                repeated_input_lengths=len(repeated),
                utterances_at_repeated_lengths=sum(map(len, repeated.values())),
                groups_with_varying_fusion_steps=sum(min(g) < max(g) for g in repeated.values()),
                modal_input_frames=example, modal_utterances=len(steps),
                modal_fixed_steps=(example + 3) // 4, modal_fusion_steps=sorted(steps))


def corpus_analyses(models):
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
    rendered, groups, front_end_steps = [], {}, {}
    input_frames = {}
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
        if uid == "P_L1":
            input_frames = {u: int(row["input_mel_frames"]) for u, row in counts.items()}
        if uid in ("E09", "E05"):
            steps = {}
            for utterance, row in counts.items():
                frames = row["layer_frames"]
                if isinstance(frames, str):
                    frames = json.loads(frames)
                steps[utterance] = int(frames[0])
            front_end_steps[uid] = steps
        if uid in ("E05", "P_MEL_L3"):
            rendered.append(r"\midrule")
        fields = [REPORT_NAMES[uid]]
        for g in parts:
            fields.extend([f"{100*g['corpus_PER']:.3f}", f"{g['total_gmac_per_audio_second']:.3f}"])
        rendered.append(" & ".join(fields) + r" \\")
    write_rows("speaking_rate.tex", rendered)
    lengths = equal_length_outputs(input_frames, front_end_steps["E09"], front_end_steps["E05"])
    return groups, lengths


def trials(rows, pair_ids=None):
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
    for a, b in (PAIRS if pair_ids is None else pair_ids):
        delta = np.array([counts[a][s] - counts[b][s] for s in speakers])
        per_draw = 100 * delta[draws].sum(axis=1) / den_draw
        lo, hi = np.percentile(per_draw, [2.5, 97.5])
        result.append(dict(a=a, b=b, delta_pp=100 * delta.sum().item() / den.sum().item(),
                           lo=float(lo), hi=float(hi)))
    return result, n


def table_matches(rows):
    """For each reduced reference, match on one measured axis at a time."""
    models = {r["id"]: r for r in rows}
    fusion = [r for r in rows if r["stage_rates"]]
    matches = []
    for field in (COST, "phone_errors"):
        for key in TABLE_STRIDES:
            base = models[key]
            partner = min(fusion, key=lambda r: (abs(r[field] - base[field]), r["id"]))
            matches.append((field, base, partner))
    return matches


def write_matched_points(matches, pairs):
    intervals = {(p["a"], p["b"]): p for p in pairs}
    headings = {
        COST: r"Nearest 30-s GMAC \quad Compare PER differences in percentage points [95\% interval]",
        "phone_errors": r"Nearest observed PER \quad Compare GMAC changes relative to fixed stride",
    }
    lines = []
    records = []
    for field, base, partner in matches:
        if not records or field != records[-1]["match_on"]:
            if lines:
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{7}{@{}l}{\emph{" + headings[field] + r"}} \\[1pt]")
        per = [f"{r[ERROR]:.4f}" for r in (base, partner)]
        cost = [f"{r[COST]:.3f}" for r in (base, partner)]
        pair = intervals[partner["id"], base["id"]]
        change = 100 * (partner[COST] / base[COST] - 1)
        if field == COST:
            if pair["lo"] > 0 or pair["hi"] < 0:
                lower = 1 if pair["delta_pp"] < 0 else 0
                per[lower] = r"\textbf{" + per[lower] + "}"
            result = f"${pair['delta_pp']:+.3f}" + r"\;[" + f"{pair['lo']:+.3f},{pair['hi']:+.3f}]$"
        else:
            if partner[COST] != base[COST]:
                lower = 1 if partner[COST] < base[COST] else 0
                cost[lower] = r"\textbf{" + cost[lower] + "}"
            result = f"${change:+.1f}" + r"\%$"
        fixed = f"{TABLE_STRIDES[base['id']]} / {base['backbone']}"
        lines.append(" & ".join([fixed, NAMES[partner["id"]], *per, *cost, result]) + r" \\")
        records.append(dict(match_on=field, fixed=base["id"], fusion=partner["id"],
                            delta_per_pp=pair["delta_pp"], interval_pp=[pair["lo"], pair["hi"]],
                            change_gmac_percent=change))
    write_rows("matched_points.tex", lines)
    return records


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


def make_figure(rows, edges, fixed, same_cost, at_ceiling):
    """Show all measured configurations and the two observed cost envelopes."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    ink, muted, grid = "#213239", "#455a64", "#dbe2e5"
    colors = ("#455766", "#006f70", "#ab5121")
    markers = {"S2": "d", NARROW: "^", "E09": "s", "E05": ">", "P_L1": "^",
               "E02": "P", "P_L3": "X", "E06": "o", "E07": "*",
               "P_MEL_L3": "v", "E10": "D"}
    fig = plt.figure(figsize=(6.45, 4.48), facecolor="white")
    ax = fig.add_axes([.103, .504, .868, .469])
    detail = fig.add_axes([.103, .243, .292, .125])
    left, right, bottom, top = 4.735, 5.035, 17, 39
    ax.set(xlim=(4.70, 6.50), ylim=(3.45, 40), xticks=np.arange(4.75, 6.51, .25))
    detail.set(xlim=(left, right), ylim=(bottom, top), xticks=[4.8, 4.9, 5.0])
    line_styles = {"fixed_stride": (colors[0], (0, (4.2, 2.6))),
                   "fusion": (colors[1], "-")}
    stop = max(r[ERROR] for r in rows)
    for panel, ticks in ((ax, [4, 8, 16, 32]), (detail, [20, 30])):
        panel.set_yscale("log", base=2)
        panel.set_yticks(ticks, labels=[str(t) for t in ticks])
        panel.yaxis.set_minor_locator(NullLocator())
        panel.tick_params(labelsize=8.3 if panel is ax else 7.6, colors=muted,
                          length=2.5, pad=3)
        panel.grid(axis="y", color=grid, linewidth=.55)
        panel.set_axisbelow(True)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            panel.spines[side].set(color="#87999f", linewidth=.7)
        for family, edge in edges.items():
            color, style = line_styles[family]
            panel.step([r[ERROR] for r in edge] + [stop],
                       [r[COST] for r in edge] + [edge[-1][COST]],
                       where="post", color=color, linestyle=style, linewidth=1.1,
                       alpha=.75, zorder=2)
    ax.set_xlabel("Test-clean phone error rate (%)", labelpad=2, color=ink, fontsize=9.2)
    ax.set_ylabel("Encoder + fusion\n(GMAC per 30 s)", color=ink, fontsize=9.2)
    ax.add_patch(Rectangle((left, bottom), right-left, top-bottom, fill=False,
                           edgecolor="#8b9ca0", linewidth=.7, linestyle=(0, (1.6, 2.8))))

    handles = {}
    for row in rows:
        key, n = row["id"], len(row["stage_rates"])
        size = {"S2": 5.9, "E07": 7.3}.get(key, 6.7)
        style = dict(marker=markers[key], markerfacecolor=colors[n] if n else "white",
                     markeredgecolor="white" if n else colors[n], linestyle="none",
                     markeredgewidth=.58 if n else 1.2)
        ax.plot(row[ERROR], row[COST], markersize=size, zorder=4, **style)
        if left <= row[ERROR] <= right and bottom <= row[COST] <= top:
            detail.plot(row[ERROR], row[COST], markersize=size, zorder=4, **style)
        handles[key] = Line2D([], [], markersize=size, **style)

    fig.text(.249, .174, "Low-PER region, enlarged", ha="center", va="center",
             fontsize=8, color=muted)
    groups = [
        ("Stride / width", .395, ["S2", NARROW, "E09"], ["2/256", "4/128", "4/512"]),
        ("One site", .347, ["E05", "P_L1", "E02", "P_L3", "E06", "E07"], None),
        ("Two sites", .299, ["P_MEL_L3", "E10"], None),
    ]
    legend_style = dict(loc="center left", frameon=False, borderpad=0, borderaxespad=0,
                        handlelength=.9, handletextpad=.24, columnspacing=.56,
                        fontsize=7.9, labelcolor=ink)
    for label, y, ids, names in groups:
        fig.text(.445, y, label, va="center", fontsize=8.1, color=ink, weight="semibold")
        fig.legend([handles[k] for k in ids], names or [NAMES[k] for k in ids],
                   bbox_to_anchor=(.598, y), ncol=len(ids), **legend_style)
    fig.text(.445, .249, "Lowest cost", va="center", fontsize=8.1, color=ink, weight="semibold")
    lines = [Line2D([], [], color=color, linestyle=style, linewidth=1.1)
             for color, style in line_styles.values()]
    fig.legend(lines, ["Fixed stride", "Fusion"], ncol=2, bbox_to_anchor=(.598, .249),
               **dict(legend_style, handlelength=2, columnspacing=1.15))

    fig.add_artist(Line2D([.03, .97], [.142, .142], color=grid, linewidth=.8,
                          transform=fig.transFigure))
    saving = 100 * (1 - at_ceiling[COST] / fixed[COST])
    for x, heading, description, result in (
        (.03, f"Near {fixed[COST]:.1f} GMAC",
         f"{NAMES[fixed['id']]} and {NAMES[same_cost['id']]}",
         f"{fixed[ERROR]:.3f}%  vs.  {same_cost[ERROR]:.3f}% PER"),
        (.50, f"At the {NAMES[fixed['id']].lower()} PER ceiling",
         f"{NAMES[fixed['id']]} and {NAMES[at_ceiling['id']]}",
         f"{fixed[COST]:.2f}  vs.  {at_ceiling[COST]:.2f} GMAC   (−{saving:.1f}%)")):
        fig.text(x, .108, heading, color=ink, weight="semibold", fontsize=9, va="center")
        fig.text(x, .067, description, color=muted, fontsize=8.2, va="center")
        fig.text(x, .024, result, color=ink, fontsize=8.9, va="center")

    output = PAPER / "figures"
    fig.savefig(output / "quality_budget_30s.pdf",
                metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None})
    fig.savefig(output / "quality_budget_30s.png", dpi=360, facecolor="white")
    plt.close(fig)


def write_rows(name, rows):
    macro = {"operating_points.tex": "OperatingRows", "actual_decode.tex": "DecodeRows",
             "paired_per.tex": "PairedRows", "speaking_rate.tex": "RateRows",
             "matched_points.tex": "MatchedRows"}[name]
    (OUTPUT / name).write_text("% Generated by build_current_results.py\n" +
                               r"\newcommand{\%s}{%%" % macro + "\n" +
                               "\n".join(rows) + "%\n}\n", encoding="utf-8")


def main():
    data = json.loads(SELECTION.read_text(encoding="utf-8"))
    selected = [r for r in comparison_rows(data) if r["id"] not in MANUSCRIPT_EXCLUSIONS]
    all_rows = sorted(selected, key=lambda r: len(r["stage_rates"]))
    matches = table_matches(all_rows)
    matched_ids = [(partner["id"], base["id"]) for _, base, partner in matches]
    pair_ids = dict.fromkeys(PAIRS + matched_ids)
    pairs, n = trials(all_rows, pair_ids)
    edge = frontier(all_rows)
    edges, examples = cost_at_error_ceilings(all_rows, (5.0, 5.7, 6.1, 6.13))
    comparisons = figure_comparisons(all_rows, pairs)
    best_fixed = edges["fixed_stride"][0]
    same_cost = next(partner for field, base, partner in matches
                     if field == COST and base["id"] == best_fixed["id"])
    ceiling_id = next(p["a"] for p in comparisons
                      if p["selection"] == "baseline ceiling" and p["b"] == best_fixed["id"])
    at_ceiling = next(r for r in all_rows if r["id"] == ceiling_id)
    OUTPUT.mkdir(exist_ok=True)
    matched_table = write_matched_points(matches, pairs)
    summary, decode = [], []
    for x in all_rows:
        i = x["id"]
        key = REPORT_NAMES[i]
        if i in ("E05", "P_MEL_L3"):
            summary.append(r"\midrule")
            decode.append(r"\midrule")
        per = f"{x[ERROR]:.4f}"
        cost = f"{x[COST]:.3f}"
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
    write_rows("paired_per.tex", [f"{REPORT_NAMES[p['a']]} / {REPORT_NAMES[p['b']]} & "
                                  f"${p['delta_pp']:+.3f}$ & "
                                  f"$[{p['lo']:+.3f}, {p['hi']:+.3f}]$" + r" \\" for p in pairs])
    groups, lengths = corpus_analyses(all_rows)
    record = dict(selection=str(SELECTION.relative_to(ROOT)), sha256=sha(SELECTION),
                  paired_unit="speaker", speakers=n, draws=10_000, seed=42,
                  interval="percentile; exploratory, not adjusted for multiple comparisons",
                  observed_frontier=[r["id"] for r in edge], contrasts=pairs,
                  minimum_cost_at_per_ceiling={k: [r["id"] for r in e] for k, e in edges.items()},
                  example_per_ceilings=examples,
                  figure_comparisons=comparisons,
                  figure_points=[r["id"] for r in all_rows],
                  figure_highlights=dict(fixed=best_fixed["id"], nearest_cost=same_cost["id"],
                                         least_cost_at_observed_per=at_ceiling["id"]),
                  matched_table=matched_table,
                  natural_rate_groups=groups, front_end_at_equal_input_length=lengths)
    (OUTPUT / "analysis.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    make_figure(all_rows, edges, best_fixed, same_cost, at_ceiling)
    make_method_figure(next(x for x in all_rows if x["id"] == "E10"), PAPER / "figures")


if __name__ == "__main__":
    main()
