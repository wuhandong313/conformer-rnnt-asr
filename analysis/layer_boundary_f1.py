#!/usr/bin/env python3
"""A0-2: novelty-peak ↔ phoneme-boundary alignment F1 per encoder layer.

For each encoder layer, detects peaks on the adjacent-frame L1 diff signal
(the "naive" detector of DynamicCompression) and measures how well they align
with phoneme boundaries from forced alignment, against uniform and random
boundary baselines with the same boundary count.

Requires MFA (Montreal Forced Aligner) TextGrids for the eval subset:
    mfa align ... --output_directory /path/to/mfa_out
    → one <fileid>.TextGrid per utterance with a "phones" tier.

Usage (on server):
    python analysis/layer_boundary_f1.py configs/exp/a3-none-s42.yaml \
        --checkpoint /test21/.../val-....ckpt \
        --textgrid-dir /test21/mfa_out/dev-clean \
        --max-samples 200 --tolerance 2 \
        --out analysis/out/layer_boundary_f1.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
from collections import defaultdict

import numpy as np
import torch

from analysis._common import (
    boundary_f1,
    capture_layer_activations,
    detect_peak_frames,
    diff_signal,
    encoder_frame_period,
    load_model,
    right_context_frames,
)
from conformer_rnnt.lightning_module import LIBRISPEECH


def parse_textgrid_phones(path: str) -> list[float]:
    """Extract phone-boundary times (seconds) from a TextGrid file.

    Handles both long ("intervals [i]:" blocks) and short (three-line groups)
    TextGrid formats; uses the tier named "phones" (MFA default).
    Every interval edge is a candidate boundary (phone↔phone, phone↔sil
    transitions included); the utterance start/end edges are dropped.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Restrict to the phones tier if tier names are present
    tier_match = re.search(
        r'name\s*=\s*"phones"(.*?)(?:item\s*\[\d+\]:|\Z)', text, re.DOTALL
    )
    segment = tier_match.group(1) if tier_match else text

    bounds = set()
    if "intervals [" in segment:
        pairs = re.findall(
            r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*"[^"]*"',
            segment,
        )
    else:
        pairs = re.findall(
            r'([\d.eE+-]+)\s*\n\s*([\d.eE+-]+)\s*\n\s*"[^"]*"', segment
        )
    for xmin, xmax in pairs:
        bounds.add(float(xmin))
        bounds.add(float(xmax))
    if len(bounds) > 2:
        bounds.discard(min(bounds))
        bounds.discard(max(bounds))
    return sorted(bounds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument("--textgrid-dir", required=True,
                        help="Directory with <fileid>.TextGrid from MFA.")
    parser.add_argument("--subset", default=None,
                        help="Eval subset (default: data.test_subset from config).")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--tolerance", type=int, default=2,
                        help="Matching tolerance in encoder frames (default ±2).")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", default="analysis/out/layer_boundary_f1.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, config, device, ckpt_path = load_model(args.config, args.checkpoint, args.cpu)
    rc = right_context_frames(config)
    period = encoder_frame_period(config)
    subset = args.subset or config["data"]["test_subset"]
    rng = np.random.default_rng(args.seed)
    logging.info("Checkpoint: %s | subset: %s | frame period: %.3fs", ckpt_path, subset, period)

    dataset = LIBRISPEECH(root=config["paths"]["dataset_path"], url=subset)

    # layer name -> list of per-utterance (f1, f1_uniform, f1_random)
    scores: dict[str, list[tuple]] = defaultdict(list)
    skipped = 0

    with torch.no_grad():
        for idx in range(min(args.max_samples, len(dataset))):
            fileid = dataset._walker[idx]
            tg_path = os.path.join(args.textgrid_dir, f"{fileid}.TextGrid")
            if not os.path.exists(tg_path):
                skipped += 1
                continue
            ref_bounds = parse_textgrid_phones(tg_path)
            if len(ref_bounds) < 2:
                skipped += 1
                continue
            ref_frames = np.unique(
                np.round(np.array(ref_bounds) / period).astype(int)
            )

            sample = dataset[idx]
            features, lengths = model._valid_extract_features([sample])
            acts, len_b1, len_c = capture_layer_activations(
                model, features.to(device), lengths.to(device)
            )
            main_b1 = int(len_b1[0]) - rc
            main_c = int(len_c[0]) - rc

            for name, tensor in acts.items():
                main = main_c if name.startswith("block2") else main_b1
                if main < 5:
                    continue
                ref = ref_frames[ref_frames < main]
                if len(ref) < 2:
                    continue
                diff = diff_signal(tensor[0, :main].cpu())
                pred = detect_peak_frames(diff)
                if len(pred) == 0:
                    continue

                f1, _, _ = boundary_f1(pred, ref, args.tolerance)
                uniform = np.linspace(0, main - 1, len(pred) + 2)[1:-1]
                f1_u, _, _ = boundary_f1(uniform, ref, args.tolerance)
                random = rng.choice(main, size=min(len(pred), main), replace=False)
                f1_r, _, _ = boundary_f1(random, ref, args.tolerance)
                scores[name].append((f1, f1_u, f1_r))

            if (idx + 1) % 20 == 0:
                logging.info("[%d] done (skipped %d)", idx + 1, skipped)

    if not scores:
        raise SystemExit(
            f"No utterances processed — check --textgrid-dir ({skipped} skipped). "
            "Run MFA first (see docstring)."
        )

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

    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        mannwhitneyu = None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "n", "f1", "f1_uniform", "f1_random",
                         "p_vs_uniform", "p_vs_random"])
        for name in sorted(scores, key=layer_order):
            vals = np.array(scores[name])
            f1m, f1u, f1r = vals.mean(axis=0)
            p_u = p_r = float("nan")
            if mannwhitneyu is not None:
                p_u = mannwhitneyu(vals[:, 0], vals[:, 1],
                                   alternative="greater").pvalue
                p_r = mannwhitneyu(vals[:, 0], vals[:, 2],
                                   alternative="greater").pvalue
            writer.writerow([name, len(vals), f"{f1m:.4f}", f"{f1u:.4f}",
                             f"{f1r:.4f}", f"{p_u:.2e}", f"{p_r:.2e}"])
            logging.info(
                "%-16s F1=%.3f (uniform %.3f, random %.3f, p=%.1e)",
                name, f1m, f1u, f1r, p_u,
            )
    logging.info("Saved: %s (skipped %d utterances without TextGrid)",
                 args.out, skipped)


if __name__ == "__main__":
    main()
