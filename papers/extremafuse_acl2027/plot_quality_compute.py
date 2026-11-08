"""Render the paper's quality/computation figure from its recorded snapshot."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parent
    snapshot = json.loads((root / "compute_snapshot.json").read_text())
    models = snapshot["models"]
    complete = [m for m in models if m["count_status"] == "complete"]
    baseline_id = snapshot["baseline_id"]
    two_stage_id = snapshot["two_stage_id"]
    baseline_per = next(m["per_percent"] for m in models if m["id"] == baseline_id)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 3.8))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.86, wspace=0.42)

    left.axhline(baseline_per, color="#777777", lw=1, ls="--", zorder=0)
    left.text(0.04, baseline_per + 0.045, f"{baseline_id} PER", ha="left", va="bottom",
              transform=left.get_yaxis_transform(), color="#666666", fontsize=10)
    colors = {"E01": "#247c84", "E02": "#b97a21", "E03": "#8d4273",
              "E04": "#596a80", "E05": "#293c51", "E06": "#087969"}
    frontier = []
    best_per = float("inf")
    for model in sorted(complete, key=lambda m: (m["total_gmac_per_audio_second"], m["per_percent"])):
        if model["per_percent"] < best_per:
            frontier.append(model)
            best_per = model["per_percent"]
    left.plot([m["total_gmac_per_audio_second"] for m in frontier],
              [m["per_percent"] for m in frontier], color="#849c9b", lw=1.1,
              ls=":", zorder=1, label="Observed nondominated points")
    for model in complete:
        x = model["total_gmac_per_audio_second"]
        y = model["per_percent"]
        marker = {baseline_id: "s", two_stage_id: "D"}.get(model["id"], "o")
        left.scatter(x, y, s=55, marker=marker, color=colors.get(model["id"], "#247c84"), zorder=3)
        offset = (5, -13) if model["id"] == two_stage_id else (5, 6)
        left.annotate(model["id"], (x, y), xytext=offset,
                      textcoords="offset points", fontsize=10)
    left.set(xlabel="Major GMAC / input-audio second", ylabel="Phone error rate (%)",
             title="(a) Full-corpus counts including decoder", xlim=(0.18, 1.02), ylim=(4.5, 6.05))
    left.grid(alpha=0.17)
    left.legend(loc="upper right", fontsize=8, frameon=False)

    comparisons = snapshot["paired_per_differences"]
    right.axvline(0, color="#777777", lw=1, ls="--", zorder=0)
    for index, comparison in enumerate(comparisons):
        delta = comparison["delta_per_pp"]
        low, high = comparison["ci95_pp"]
        right.errorbar(delta, index, xerr=[[delta - low], [high - delta]],
                       fmt="o", color="#344a65", ecolor="#344a65", capsize=4, ms=6, lw=1.6)
    right.set_yticks(range(len(comparisons)), [c["label"] for c in comparisons])
    right.set(xlabel="PER difference (percentage points)",
              title="(b) Paired quality comparisons", xlim=(-1.28, 0.36),
              ylim=(len(comparisons) - 0.4, -0.6))
    right.grid(axis="x", alpha=0.17)
    pending = snapshot["pending_full_counts"]
    status = f"; counts pending: {', '.join(pending)}" if pending else "; all counts complete"
    fig.text(0.065, 0.035, "test-clean: 2,620 utterances; beam 20" + status,
             fontsize=9, color="#555555")
    fig.text(0.61, 0.035, "95% intervals: 10,000 paired speaker resamples",
             fontsize=9, color="#555555")
    output = root / "figures/quality_compute.png"
    fig.savefig(output, dpi=220, metadata={"Software": "Matplotlib; source: compute_snapshot.json"})
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
