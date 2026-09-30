#!/usr/bin/env python3
"""A0-1: per-layer temporal redundancy statistics (research-plan v2.0 §6 Phase A0).

Runs a trained checkpoint (ideally compression.filter=none) over eval utterances
and reports, per encoder layer:

  - diff_mean / diff_std   : adjacent-frame L1 novelty signal statistics
  - cos_mean / cos_std     : adjacent-frame cosine similarity statistics
  - novelty_sparsity       : fraction of frames that are local maxima of the diff signal
  - still_ratio            : fraction of adjacent pairs with cosine similarity > 0.9

Motivation experiment for the paper (§Motivation): demonstrates that inter-layer
activations are highly temporally redundant and that redundancy varies by depth.

Usage (on server):
    python analysis/layer_redundancy.py configs/exp/a3-none-s42.yaml \
        --checkpoint /test21/conformer-rnnt-asr/exp/a3-none-s42/checkpoints/val-....ckpt \
        --max-samples 200 --out analysis/out/layer_redundancy.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from analysis._common import (
    capture_layer_activations,
    diff_signal,
    load_model,
    right_context_frames,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def layer_stats(frames: torch.Tensor, valid_len: int) -> dict:
    """Redundancy statistics of one layer's activations for one utterance."""
    x = frames[:valid_len]
    if x.shape[0] < 3:
        return {}
    diff = diff_signal(x)
    cos = F.cosine_similarity(x[1:], x[:-1], dim=-1).cpu().numpy()
    # local maxima of the diff signal (strict interior peaks)
    peaks = (diff[1:-1] > diff[:-2]) & (diff[1:-1] >= diff[2:])
    return {
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std()),
        "cos_mean": float(cos.mean()),
        "cos_std": float(cos.std()),
        "novelty_sparsity": float(peaks.sum() / max(len(diff) - 2, 1)),
        "still_ratio": float((cos > 0.9).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", default="analysis/out/layer_redundancy.csv")
    args = parser.parse_args()

    model, config, device, ckpt_path = load_model(args.config, args.checkpoint, args.cpu)
    rc = right_context_frames(config)
    logging.info("Checkpoint: %s | device: %s", ckpt_path, device)

    per_layer: dict[str, list[dict]] = defaultdict(list)
    dataloader = model.test_dataloader()

    with torch.no_grad():
        for idx, (batch, _transcripts) in enumerate(dataloader):
            if idx >= args.max_samples:
                break
            features = batch.features.to(device)
            lengths = batch.feature_lengths.to(device)
            acts, len_b1, len_c = capture_layer_activations(model, features, lengths)

            main_b1 = int(len_b1[0]) - rc  # exclude right context
            main_c = int(len_c[0]) - rc
            for name, tensor in acts.items():
                valid = main_c if name.startswith("block2") else main_b1
                stats = layer_stats(tensor[0].cpu(), valid)
                if stats:
                    per_layer[name].append(stats)
            if (idx + 1) % 20 == 0:
                logging.info("[%d/%d] done", idx + 1, args.max_samples)

    def layer_order(name: str) -> tuple:
        if name == "time_reduction":
            return (0, 0)
        if name.startswith("block1."):
            return (1, int(name.split(".L")[1]))
        if name == "block1_out":
            return (2, 0)
        if name == "compression_out":
            return (3, 0)
        return (4, int(name.split(".L")[1]))

    rows = []
    for name in sorted(per_layer, key=layer_order):
        samples = per_layer[name]
        row = {"layer": name, "n": len(samples)}
        for key in ("diff_mean", "diff_std", "cos_mean", "cos_std",
                    "novelty_sparsity", "still_ratio"):
            vals = [s[key] for s in samples]
            row[key] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
            row[f"_{key}_mean"] = float(np.mean(vals))
        rows.append(row)
        logging.info(
            "%-16s cos=%.3f  sparsity=%.3f  still=%.3f  diff=%.2f",
            name, row["_cos_mean_mean"], row["_novelty_sparsity_mean"],
            row["_still_ratio_mean"], row["_diff_mean_mean"],
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["layer", "n", "diff_mean", "diff_std", "cos_mean", "cos_std",
                  "novelty_sparsity", "still_ratio"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fieldnames} for r in rows])
    logging.info("Saved: %s", args.out)

    if HAS_MPL:
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        x = range(len(rows))
        labels = [r["layer"] for r in rows]
        ax1.plot(x, [r["_cos_mean_mean"] for r in rows], "o-", label="adjacent cosine sim")
        ax1.set_ylabel("cosine similarity")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2 = ax1.twinx()
        ax2.plot(x, [r["_novelty_sparsity_mean"] for r in rows], "s--",
                 color="darkred", label="novelty sparsity")
        ax2.plot(x, [r["_still_ratio_mean"] for r in rows], "^--",
                 color="gray", label="still ratio (cos>0.9)")
        ax2.set_ylabel("ratio")
        fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
        fig.tight_layout()
        png = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png, dpi=150)
        logging.info("Saved: %s", png)
    else:
        logging.warning("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
