"""Plot the selected paper systems without altering historical evaluation archives."""

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phone69-paper-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def observed_frontier(points):
    """Return nondominated observed points for two lower-is-better objectives."""
    frontier = []
    best_per = float("inf")
    for point in sorted(points, key=lambda p: (p["gmac_per_audio_second"], p["per_percent"])):
        if point["per_percent"] < best_per:
            frontier.append(point)
            best_per = point["per_percent"]
    return frontier


def main():
    paper = Path(__file__).resolve().parent
    root = paper.parents[1]
    selection_path = paper / "fixed_duration_selection.json"
    selection = json.loads(selection_path.read_text())
    source_path = root / selection["source"]
    source = json.loads(source_path.read_text())
    duration = selection["duration_seconds"]
    models = {model["id"]: model for model in source["models"]}
    selected_ids = [entry["archive_id"] for entry in selection["selected"]]
    excluded_ids = [entry["archive_id"] for entry in selection["excluded"]]
    assert len(selected_ids) == len(set(selected_ids)) == 9
    assert not set(selected_ids) & set(excluded_ids)
    complete = {key for key, model in models.items() if model["status"] == "calibrated_theory"}
    assert set(selected_ids) | set(excluded_ids) == complete
    assert "E08" in selected_ids and "E01" in excluded_ids
    points = []
    for entry in selection["selected"]:
        model = models[entry["archive_id"]]
        assert model["status"] == "calibrated_theory"
        assert model["calibration_utterances"] == 2620
        matches = [r for r in model["results"] if r["duration_seconds"] == duration]
        assert len(matches) == 1
        result = matches[0]
        assert result["actual_beam20_macs"] is None
        gmac = sum(result["components_macs"].values()) / 1e9
        assert abs(gmac - result["encoder_compression_gmac"]) < 1e-10
        assert abs(gmac / duration - result["encoder_compression_gmac_per_audio_s"]) < 1e-10
        architecture = model["model"]
        points.append({
            **entry,
            "paper_id": model["paper_id"],
            "backbone_dim": architecture["time_reduction_input_dim"] * architecture["time_reduction_stride"],
            "per_percent": model["per_percent"],
            "gmac_30_seconds": gmac,
            "gmac_per_audio_second": gmac / duration,
            "checkpoint_sha256": model["checkpoint_sha256"],
            "config_sha256": model["config_sha256"],
        })
    frontier = observed_frontier(points)
    frontier_ids = [point["archive_id"] for point in frontier]
    assert frontier_ids == ["E10", "E05", "E02", "E08", "E06"]
    output = paper / "fixed_duration_curve_20261112"
    output.mkdir(exist_ok=True)
    snapshot = {
        "duration_seconds": duration,
        "source": selection["source"],
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": selection["scope"],
        "curve_policy": selection["curve_policy"],
        "selection_basis": selection["selection_basis"],
        "observed_frontier_archive_ids": frontier_ids,
        "models": points,
        "excluded": selection["excluded"],
        "pending": [key for key in models if key not in complete],
    }
    (output / "points.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    fields = ["archive_id", "paper_id", "label", "family", "backbone_dim", "per_percent", "gmac_30_seconds", "gmac_per_audio_second"]
    with (output / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.28, top=0.76)
    styles = {"fixed": ("#526175", "s", "Fixed stride"),
              "single": ("#277DA8", "o", "Single-stage learned-5"),
              "double": ("#15846C", "D", "Two-stage learned-5"),
              "mixed": ("#B77924", "P", "Mixed hard-3 / learned-5")}
    ax.plot([p["gmac_per_audio_second"] for p in frontier],
            [p["per_percent"] for p in frontier], color="#A2ABB3", ls="--", lw=1.5,
            zorder=1, label="Observed frontier (selected systems)")
    for family, (color, marker, label) in styles.items():
        group = [p for p in points if p["family"] == family]
        ax.scatter([p["gmac_per_audio_second"] for p in group],
                   [p["per_percent"] for p in group], marker=marker, color=color,
                   s=85 if family != "mixed" else 125, edgecolors="white", linewidths=0.8,
                   zorder=3, label=label)
    offsets = {"S1": (-10, 12, "right"), "S2": (14, 10, "left"),
               "E09": (10, -12, "left"), "E05": (8, -17, "left"),
               "E02": (8, 10, "left"), "E06": (8, -17, "left"),
               "E07": (-14, -17, "right"), "E10": (0, 12, "center"),
               "E08": (8, 10, "left")}
    for point in points:
        dx, dy, align = offsets[point["archive_id"]]
        label = point["label"]
        if point["family"] == "fixed":
            label += f" (D={point['backbone_dim']})"
        ax.annotate(label, (point["gmac_per_audio_second"], point["per_percent"]),
                    xytext=(dx, dy), textcoords="offset points", ha=align, fontsize=10,
                    color=styles[point["family"]][0])
    ax.set(xlabel="Theoretical encoder + compression GMAC / audio-s (30-s input)",
           ylabel="Full test-clean phone error rate (%)",
           xlim=(0.10, 1.23), ylim=(4.65, 5.89))
    ax.grid(alpha=0.2)
    top = ax.secondary_xaxis("top", functions=(lambda x: x * duration, lambda x: x / duration))
    top.set_xlabel("Theoretical encoder + compression GMAC / 30-s input", labelpad=10)
    top.spines["top"].set_visible(True)
    fig.suptitle("Compression placement: quality-computation trade-off", x=0.09,
                 y=0.97, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.09, 0.913, "Nine selected systems | 2,620 utterances | FP32, batch 1, beam 20", color="#526175", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=8.7)
    fig.text(0.09, 0.055, "* L2+L5 uses hard three-frame neighborhoods at L2 and learned five-frame fusion at L5.", fontsize=9, color="#526175")
    fig.text(0.09, 0.027, "Lines connect observed points only; they do not interpolate unmeasured PER. Decoder MACs are separate.", fontsize=9, color="#526175")
    fig.savefig(output / "quality_compute_30s.png", dpi=220)
    fig.savefig(output / "quality_compute_30s.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    lines = ["# 论文核心性能—计算量曲线（固定30秒）", "",
             "2026-11-12按用户选择：移除128维stride4；L2+L5进入核心位置比较。当前共九组已核验工作点。", "",
             "![核心曲线](quality_compute_30s.png)", "",
             "| 原归档ID | 系统 | 主干维度 | PER (%) | GMAC／30秒 | GMAC／音频秒 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for p in points:
        lines.append(f"| {p['archive_id']} | {p['label']} | {p['backbone_dim']} | {p['per_percent']:.4f} | {p['gmac_30_seconds']:.6f} | {p['gmac_per_audio_second']:.6f} |")
    lines += ["", "当前入选集合的观察前沿：**Mel+L5 → Mel → L2 → L2+L5* → L5**。L2+L5补充了L2与L5之间的质量—计算量工作点；L7作为晚层位置对照继续显示。", "",
              "*L2+L5*的第一阶段为hard三帧邻域，第二阶段为learned五帧融合。它是核心结构／插入位置对比，但融合规则与其他learned五帧系统不完全相同，不能将差异全部归因于位置。", "",
              "所有点均为独立训练并按dev-clean选择checkpoint的模型。连线只标记入选系统的观察前沿，不代表同一checkpoint可连续调节，不插值未知PER。未计算新配对区间，不宣称点间差异有统计显著性。", "",
              "横轴为校准保留率下的固定30秒编码器＋压缩主要MAC；纵轴为原始完整test-clean corpus PER，CPU/FP32/batch1/beam20。MAC计入投影、线性、卷积、注意力矩阵乘积、检测点积及融合；不计特征提取、归一化、激活、softmax、比较排序及访存。实际beam搜索predictor/joiner成本保留在历史全量计数中，不混入这条理论曲线。", "",
              "该图是用户筛选的探索性工作点比较，不是全部stride扫描或全部实验的前沿。不同模型的宽度、batch和更新次数差异仍需披露。后续预算比较应先在dev-clean冻结预算和候选，再在独立集确认。", "",
              "[入选与排除清单](../fixed_duration_selection.json) · [机器可读数据与来源哈希](points.json) · [CSV](points.csv) · [矢量PDF](quality_compute_30s.pdf) · [固定长度公式与范围](../../../docs/phone69_fixed_duration_macs.md)", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"selected": len(points), "excluded": len(excluded_ids),
                      "pending": len(snapshot["pending"]), "frontier": frontier_ids,
                      "report": str(output / "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
