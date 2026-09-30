#!/usr/bin/env python3
"""A0-3: compression-rate distribution of each detector over eval utterances.

Uses block-1 outputs of a trained checkpoint and runs each filter family's
peak/valley detector (the exact ``DynamicCompression._detect_peaks_valleys``
code paths), reporting per-utterance keep rates. This fills the v1.1 blind
spot: the actual compression-rate distribution was never measured.

Detector families (detection is shared within a family):
    naive   — raw diff find_peaks (also used by "hanning")
    fir     — FIR lowpass → coarse peaks → refine on raw
    savgol  — Savitzky-Golay smoothing → find_peaks (also "savgol_hanning")

Usage (on server):
    python analysis/rate_distribution.py configs/exp/a3-none-s42.yaml \
        --checkpoint /test21/.../val-....ckpt --max-samples 500 \
        --out analysis/out/rate_distribution.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os

import numpy as np
import torch

from analysis._common import load_model, right_context_frames
from conformer_rnnt.compression import DynamicCompression
from conformer_rnnt.lightning_module import LIBRISPEECH

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

DETECTOR_FAMILIES = ["naive", "fir", "savgol"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument("--subset", default=None,
                        help="Eval subset (default: data.test_subset from config).")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", default="analysis/out/rate_distribution.csv")
    args = parser.parse_args()

    model, config, device, ckpt_path = load_model(args.config, args.checkpoint, args.cpu)
    model_cfg = config["model"]
    rc = right_context_frames(config)
    subset = args.subset or config["data"]["test_subset"]
    logging.info("Checkpoint: %s | subset: %s", ckpt_path, subset)

    # One detector instance per filter family (detection-only; weight predictor unused)
    detectors = {
        fam: DynamicCompression.from_legacy(
            right_context_length=model_cfg["right_context_length"],
            compression_filter=fam,
            feature_dim=model_cfg["time_reduction_input_dim"]
            * model_cfg["time_reduction_stride"],
        )
        for fam in DETECTOR_FAMILIES
    }

    dataset = LIBRISPEECH(root=config["paths"]["dataset_path"], url=subset)
    tr = model.model.transcriber

    rows = []
    with torch.no_grad():
        for idx in range(min(args.max_samples, len(dataset))):
            fileid = dataset._walker[idx]
            sample = dataset[idx]
            features, lengths = model._valid_extract_features([sample])
            features = features.to(device)
            lengths = lengths.to(device)

            # Frontend up to the compression point (same as encoder.forward)
            x, x_len = tr.time_reduction(tr.input_linear(features), lengths)
            x, len_b1 = tr.conformer1(x, x_len + rc)

            total = int(len_b1[0])
            main = total - rc
            if main < 5:
                continue
            diff_b = (
                torch.abs(x[0, 1:main] - x[0, :main - 1]).sum(dim=-1).cpu().numpy()
            )

            row = {"fileid": fileid, "total_frames": total, "main_frames": main}
            x_b = x[0, :main, :]
            diff_b_t = torch.abs(x_b[1:] - x_b[:-1]).sum(dim=-1)
            for fam, det in detectors.items():
                indices, _ = det.detector.detect(x_b, diff_b_t, diff_b, main)
                merged = indices.cpu().numpy()
                merged = merged[merged < main]
                kept = len(merged) + rc
                row[f"keep_{fam}"] = len(merged)
                row[f"rate_{fam}"] = kept / total
            rows.append(row)
            if (idx + 1) % 50 == 0:
                logging.info("[%d] done", idx + 1)

    if not rows:
        raise SystemExit("No utterances processed.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["fileid", "total_frames", "main_frames"]
    for fam in DETECTOR_FAMILIES:
        fieldnames += [f"keep_{fam}", f"rate_{fam}"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Saved: %s (%d utterances)", args.out, len(rows))

    logging.info("%-10s %8s %8s %8s %8s %8s", "detector", "mean", "std",
                 "p10", "p50", "p90")
    for fam in DETECTOR_FAMILIES:
        rates = np.array([r[f"rate_{fam}"] for r in rows])
        logging.info(
            "%-10s %8.3f %8.3f %8.3f %8.3f %8.3f",
            fam, rates.mean(), rates.std(),
            *np.percentile(rates, [10, 50, 90]),
        )

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(7, 4))
        for fam in DETECTOR_FAMILIES:
            rates = [r[f"rate_{fam}"] for r in rows]
            ax.hist(rates, bins=40, alpha=0.5, label=fam, density=True)
        ax.set_xlabel("keep rate (compressed length / original length)")
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        png = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png, dpi=150)
        logging.info("Saved: %s", png)
    else:
        logging.warning("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
