#!/usr/bin/env python3
"""Evaluate a trained Conformer-RNNT checkpoint.

Usage:
    python eval.py configs/server.yaml --checkpoint exp/checkpoints/val-epoch=02-step=4000.ckpt
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import time
from argparse import ArgumentParser
from typing import Any, Dict

import torch
import torchaudio
import yaml

from conformer_rnnt.lightning_module import LibriSpeechRNNTModule


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file '{path}' is empty.")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_per(reference: str, hypothesis: str) -> float:
    """Phoneme Error Rate — character-level edit distance on raw text."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return torchaudio.functional.edit_distance(reference, hypothesis) / len(reference)


# ═══════════════════════════════════════════════════════════════════════════
# Eval loop
# ═══════════════════════════════════════════════════════════════════════════


def run_eval(model: LibriSpeechRNNTModule, device: torch.device, per_sample_file: str = "eval_per_sample.csv", max_utts: int = 0) -> Dict[str, Any]:
    model = model.to(device)
    model.eval()
    dataloader = model.test_dataloader()

    total_per = 0.0
    total_rtf = 0.0
    n = 0
    per_sample = []

    with torch.no_grad():
        for idx, (batch, transcripts) in enumerate(dataloader):
            if max_utts and n >= max_utts:
                break
            actual = transcripts[0].lower()
            if model.strip_spaces:
                actual = actual.replace(" ", "")

            # Timing
            t0 = time.time()
            predicted = model(batch)
            t1 = time.time()

            # RTF: processing_time / audio_duration
            frame_dur = model._mel_kwargs["hop_length"] / model._mel_kwargs["sample_rate"]
            audio_dur = batch.feature_lengths[0].item() * frame_dur

            # PER: character-level edit distance on raw text
            per = compute_per(actual, predicted)
            rtf = (t1 - t0) / audio_dur if audio_dur > 0 else 0

            total_per += per
            total_rtf += rtf
            n += 1
            per_sample.append({"idx": idx, "PER": per, "RTF": rtf, "ref": actual, "hyp": predicted})

            if idx % 5 == 0:
                cum_per = total_per / n if n > 0 else 0
                cum_rtf = total_rtf / n if n > 0 else 0
                logging.info(
                    "[%4d] PER: %.3f (cum: %.3f)  RTF: %.4f (cum: %.4f)  |  ref: %s  |  hyp: %s",
                    idx, per, cum_per, rtf, cum_rtf,
                    actual[:60], predicted[:60],
                )

    avg_per = total_per / n if n > 0 else 0
    avg_rtf = total_rtf / n if n > 0 else 0

    logging.info("=" * 60)
    logging.info("Evaluation complete — %d samples", n)
    logging.info("PER: %.4f", avg_per)
    logging.info("RTF: %.4f", avg_rtf)
    logging.info("=" * 60)

    # Save per-sample CSV
    import csv
    with open(per_sample_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "PER", "RTF", "ref", "hyp"])
        writer.writeheader()
        writer.writerows(per_sample)
    logging.info("Per-sample results saved to: %s", per_sample_file)

    return {"per": avg_per, "rtf": avg_rtf, "num_samples": n}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def _find_latest_val_ckpt(checkpoint_dir: str) -> str | None:
    """Find the latest val-*.ckpt in the checkpoint directory."""
    pattern = os.path.join(checkpoint_dir, "val-*.ckpt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    parser = ArgumentParser(description="Evaluate a Conformer-RNNT checkpoint.")
    parser.add_argument("config", help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint", "-c", default=None,
        help="Path to .ckpt checkpoint. If not given, auto-loads latest val-*.ckpt.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if config says cuda.")
    parser.add_argument("--max-utts", type=int, default=0, help="Evaluate only the first N utterances (0 = full set).")
    parser.add_argument("--out", default="eval_per_sample.csv", help="Path for the per-sample CSV.")
    args = parser.parse_args()

    # Config
    config = load_config(args.config)

    # Logging
    debug = config.get("debug", False)
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

    # Build model from config (architecture, paths — all from YAML)
    logging.info("Building model from config …")
    model = LibriSpeechRNNTModule.from_config(config)
    logging.info(
        "Decoding: beam_width=%s (%s)",
        model.beam_width,
        "greedy" if model.beam_width == 0 else f"beam={model.beam_width}",
    )

    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        exp_dir = config.get("paths", {}).get("exp_dir", "./exp")
        checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        ckpt_path = _find_latest_val_ckpt(checkpoint_dir)
        if ckpt_path is None:
            logging.error("No val-*.ckpt found in %s. Specify --checkpoint.", checkpoint_dir)
            sys.exit(1)
        logging.info("Auto-selected checkpoint: %s", ckpt_path)

    # Load checkpoint
    logging.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    # Eval
    device = torch.device("cpu" if args.cpu else config.get("trainer", {}).get("accelerator", "cuda"))
    run_eval(model, device, per_sample_file=args.out, max_utts=args.max_utts)


if __name__ == "__main__":
    main()
