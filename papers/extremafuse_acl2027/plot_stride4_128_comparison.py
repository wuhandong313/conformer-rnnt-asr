"""Compare the paper's twelve systems with the archived narrow stride-4 model."""
import json

from build_current_results import COST, ERROR, NAMES, PAPER, ROOT, SELECTION, cost_at_error_ceilings, sha
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator


NARROW = "E01"
MARKERS = {
    "S1": "o", "S2": "d", "E09": "s", NARROW: "^", "E05": ">", "P_L1": "^",
    "E02": "P", "P_L3": "X", "E06": "o", "E07": "*", "P_L9": "s",
    "P_MEL_L3": "v", "E10": "D",
}
INK, MUTED, GRID = "#24363c", "#485d65", "#dce5e6"
FIXED, ONE_SITE, TWO_SITES, HIGHLIGHT = "#506575", "#00736e", "#a74319", "#773b9a"
FRONTIERS = {"fixed_stride": (FIXED, (0, (4, 2.7))), "fusion": (ONE_SITE, "-")}


def comparison_rows():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    source = ROOT / selection["source"]
    if sha(source) != selection["source_sha256"] or selection["duration_seconds"] != 30:
        raise ValueError("Review the evaluation archive and reference duration")
    archive = json.loads(source.read_text(encoding="utf-8"))
    archived = {r["id"]: r for r in archive["models"]}
    ids = selection["selected_ids"]
    narrow = archived[NARROW]
    if (len(ids) != len(NAMES) or set(ids) != set(NAMES) or NARROW in ids or
            narrow["backbone"] != 128 or narrow["stage_rates"]):
        raise ValueError("Review the fixed-stride comparison after changing the selection")
    rows = [archived[key] for key in ids]
    if rows != selection["models"]:
        raise ValueError("The selected operating points differ from the full evaluation archive")
    rows.append(narrow)
    if len({(r["utterances"], r["reference_phones"], r["checkpoint_selection"]) for r in rows}) != 1:
        raise ValueError("The evaluations must use the same corpus and checkpoint selection")
    return rows


def point_style(row):
    key = row["id"]
    sites = len(row["stage_rates"])
    color = HIGHLIGHT if key == NARROW else (FIXED, ONE_SITE, TWO_SITES)[sites]
    sizes = {NARROW: 9, "E07": 7.5, "S2": 5.6, "P_MEL_L3": 7.5}
    return dict(marker=MARKERS[key], markersize=sizes.get(key, 6.8), linestyle="none",
                markerfacecolor=color if sites else "white",
                markeredgecolor="white" if sites else color,
                markeredgewidth=.65 if sites else 1.55)


def panel(ax, rows, edges, xlim, ylim, xticks, yticks):
    ax.set_yscale("log", base=2)
    ax.set(xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks,
           yticklabels=[str(tick) for tick in yticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(labelsize=9, colors=MUTED, length=2.6, pad=3.5)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set(color="#9aa9ae", linewidth=.75)
    ax.grid(axis="y", color=GRID, linewidth=.65)
    ax.set_axisbelow(True)
    for family, edge in edges.items():
        color, line = FRONTIERS[family]
        # Each step holds the observed cost until the next measured point qualifies.
        ax.step([r[ERROR] for r in edge] + [max(r[ERROR] for r in rows)],
                [r[COST] for r in edge] + [edge[-1][COST]],
                where="post", color=color, linestyle=line, linewidth=1.1, alpha=.78)
    for row in rows:
        ax.plot(row[ERROR], row[COST], zorder=4, **point_style(row))


def main():
    rows = comparison_rows()
    models = {r["id"]: r for r in rows}
    edges, _ = cost_at_error_ceilings(rows, ())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.65, 5.02), facecolor="white")
    ax = fig.add_axes([.094, .524, .878, .445])
    panel(ax, rows, edges, (4.70, 6.50), (3.5, 40),
          (4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50), (4, 8, 16, 32))
    ax.set_xlabel("Test-clean phone error rate (%)", color=INK, fontsize=10)
    ax.set_ylabel("Encoder + fusion\n(GMAC per 30 s)", color=INK, fontsize=10)

    left, right, low, high = 5.60, 6.23, 3.91, 5.60
    ax.add_patch(Rectangle((left, low), right-left, high-low, fill=False,
                           edgecolor="#89949c", linewidth=.85, linestyle=(0, (1.5, 2.2))))
    zoom = fig.add_axes([.094, .160, .289, .185])
    panel(zoom, rows, edges, (left, right), (low, high), (5.7, 5.9, 6.1), (4, 5))
    zoom.tick_params(labelsize=8)
    fig.text(.094, .370, "Low-compute region, enlarged", color=INK, weight="semibold")

    groups = [
        ("Stride / width", .386, ["S1", "S2", "E09", NARROW], ["1 / 128", "2 / 256", "4 / 512", "4 / 128"]),
        ("One site", .323, ["E05", "P_L1", "E02", "P_L3"], None),
        ("", .271, ["E06", "E07", "P_L9"], None),
        ("Two sites", .208, ["P_MEL_L3", "E10"], None),
    ]
    legend_style = dict(loc="center left", frameon=False, borderpad=0, borderaxespad=0,
                        handlelength=1, handletextpad=.31, columnspacing=.93, fontsize=8.6,
                        labelcolor=INK)
    for heading, y, ids, labels in groups:
        fig.text(.428, y, heading, va="center", fontsize=8.6, weight="semibold", color=INK)
        handles = [Line2D([], [], **point_style(models[key])) for key in ids]
        fig.legend(handles, labels or [NAMES[key] for key in ids], ncol=len(ids),
                   bbox_to_anchor=(.557, y), **legend_style)
    fig.text(.428, .140, "Family frontier", va="center", fontsize=8.6, weight="semibold", color=INK)
    lines = [Line2D([], [], color=color, linestyle=style) for color, style in FRONTIERS.values()]
    fig.legend(lines, ["Fixed stride", "Fusion"], bbox_to_anchor=(.582, .140), ncol=2,
               **dict(legend_style, handlelength=2))

    narrow = models[NARROW]
    fig.add_artist(Line2D([.046, .973], [.085, .085], color=GRID, linewidth=.85,
                          transform=fig.transFigure))
    fig.text(.049, .037, "Stride 4 · 128 dimensions", va="center", color=HIGHLIGHT, weight="bold")
    fig.text(.32, .037, f"{narrow[ERROR]:.4f}% PER     ·     {narrow[COST]:.3f} GMAC per 30 s",
             va="center", color=INK)
    out = PAPER / "figures"
    fig.savefig(out / "stride4_128_comparison.pdf",
                metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None})
    fig.savefig(out / "stride4_128_comparison.png", dpi=270, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
