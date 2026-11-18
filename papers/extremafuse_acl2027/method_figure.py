"""Draw the Mel+L5 transducer and the operation performed at each fusion site."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def make_method_figure(model, output):
    trace = model["trace_30s"]
    stages = trace["stages"]
    if model["id"] != "E10" or [s["after_layer"] for s in stages] != [0, 5]:
        raise ValueError("The architecture diagram requires the Mel+L5 checkpoint")
    mel, fifth = stages
    layer_frames = trace["layer_frames"]
    if (layer_frames != [mel["output_frames"]] * 5 + [fifth["output_frames"]] * 5
            or fifth["input_frames"] != mel["output_frames"]):
        raise ValueError("The layer-by-layer trace disagrees with the diagram")

    width, height = 458, 221
    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    ink, muted, rule = "#22323a", "#4b5d66", "#c8d2d4"
    teal, pale = "#006c6b", "#e2f2ef"

    def label(x, y, string, size=8, *, align="center", color=ink, bold=False):
        ax.text(x, y, string, ha=align, va="center", fontsize=size,
                fontfamily="DejaVu Sans", color=color, weight="bold" if bold else "normal")

    def box(x, y, w, h, title, detail=None, *, fusion=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=2.3",
            facecolor=pale if fusion else "#f6f8f8", edgecolor=teal if fusion else "#a0afb4",
            linewidth=1.0 if fusion else .65))
        color = teal if fusion else ink
        label(x + w / 2, y + h / 2 - (5.1 if detail else 0), title, 8.3, color=color, bold=True)
        if detail:
            label(x + w / 2, y + h / 2 + 5.6, detail, 7.7, color=color)

    def arrow(start, end, color=muted):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", color=color,
                                     linewidth=.9, mutation_scale=6.8, shrinkA=0, shrinkB=.4))

    def rule_line(xs, ys, color=rule):
        ax.plot(xs, ys, lw=.7, color=color, solid_capstyle="butt")

    label(2, 8, "(a)  Mel+L5 transducer", 10.8, align="left", bold=True)
    label(455, 8, "30-second reference input", 8.1, align="right", color=muted)
    width_model = model["backbone"]
    box(2, 21, 53, 31, "Mel input", "80 features")
    box(67, 21, 48, 31, "Mel fuse", "$d=80$", fusion=True)
    box(127, 21, 77, 31, f"Project to {width_model}", "Conformer 1–5")
    box(216, 21, 50, 31, "L5 fuse", f"$d={width_model}$", fusion=True)
    box(278, 21, 85, 31, "Conformer 6–10", "Project to 256")
    box(378, 21, 77, 31, "RNN-T joiner", "70 logits")
    for right, left in ((55, 67), (115, 127), (204, 216), (266, 278), (363, 378)):
        arrow((right, 36.5), (left, 36.5))
    for x, length in ((28.5, mel["input_frames"]), (165.5, layer_frames[0]),
                      (320.5, layer_frames[-1])):
        label(x, 61.5, f"{length:,} frames", 8.2, color=muted)
    label(233, 83, "Previous phones", 8.3, align="right")
    arrow((238, 83), (252, 83))
    box(252, 74.5, 119, 17, "2-layer LSTM predictor")
    rule_line([371, 416.5], [83, 83], muted)
    arrow((416.5, 83), (416.5, 52))

    rule_line([2, 455], [101, 101])
    rule_line([194, 194], [111, 216])
    label(2, 115, "(b)  Select fusion centers", 10.2, align="left", bold=True)
    label(205, 115, "(c)  Fuse a five-frame window", 10.2, align="left", bold=True)
    label(3, 132, r"Adjacent distance   $d_t=1-\cos(h_t,h_{t+1})$", 8.1,
          align="left", color=muted)
    # Eight distances lie between nine frame centers. The peak at 3 selects 4
    # and the valley at 6 selects 6, exactly as in the extrema detector.
    frame_x = [13 + 20 * i for i in range(9)]
    edge_x = [(a + b) / 2 for a, b in zip(frame_x, frame_x[1:])]
    edge_y = [171, 160, 146, 160, 168, 177, 165, 155]
    ax.plot(edge_x, edge_y, linewidth=1.5, color=muted, solid_capstyle="round")
    for index, center in ((2, 3), (5, 5)):
        x, y = edge_x[index], edge_y[index]
        arrow((x, y + 3), (frame_x[center], 186), teal)
        ax.plot(x, y, "o", ms=4.6, mfc=teal, mec="white", mew=.65)
    for i, x in enumerate(frame_x, start=1):
        box(x - 8, 187, 16, 15, str(i), fusion=i in (4, 6))
    label(3, 214, "Peak → next frame", 7.9, align="left", color=muted)
    label(89, 214, "Valley → current frame", 7.9, align="left", color=muted)

    label(205, 132, "Window for center 4", 8.1, align="left", color=muted)
    ax.add_patch(FancyBboxPatch((202.5, 142.5), 118, 22,
                                boxstyle="round,pad=0,rounding_size=2.3",
                                facecolor="white", edgecolor=muted, linewidth=.7))
    for i, j in enumerate(range(2, 7)):
        box(205 + 23 * i, 145, 21, 17, f"$h_{j}$", fusion=j == 4)
    # Both branches originate at the same window. Only the lower branch
    # transforms its features; the sum receives the original frame vectors.
    rule_line([320.5, 412], [153.5, 153.5], muted)
    arrow((412, 153.5), (412, 183))
    label(364, 146, "Original frames", 8, color=muted)
    arrow((261.5, 164.5), (261.5, 183))
    label(268, 172.5, "Concatenate", 7.8, align="left", color=muted)
    box(205, 183, 113, 30, "MLP   5d → 32 → 5", "Masked softmax")
    arrow((318, 198), (367, 198), teal)
    label(342, 189.5, r"Weights $\alpha_{4j}$", 7.8, color=teal)
    box(367, 183, 88, 30, "Weighted sum", r"Output $z_4\in\mathbb{R}^d$", fusion=True)

    fig.savefig(output / "extremafuse_network.pdf",
                metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None})
    fig.savefig(output / "extremafuse_network.png", dpi=360, facecolor="white")
    plt.close(fig)
