#!/usr/bin/env python3
"""Evaluate a trained Conformer-RNNT checkpoint.

Usage:
    python eval.py configs/server.yaml --checkpoint exp/checkpoints/val-epoch=02-step=4000.ckpt
"""

from __future__ import annotations

import logging
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


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word-level WER between two space-joined piece strings."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return torchaudio.functional.edit_distance(ref_words, hyp_words) / len(ref_words)


# ═══════════════════════════════════════════════════════════════════════════
# Eval loop
# ═══════════════════════════════════════════════════════════════════════════


def run_eval(model: LibriSpeechRNNTModule, device: torch.device, per_sample_file: str = "eval_per_sample.csv") -> Dict[str, Any]:
    model = model.to(device)
    model.eval()
    sp = model.sp_model
    dataloader = model.test_dataloader()

    total_wer = 0.0
    total_rtf = 0.0
    n = 0
    per_sample = []

    with torch.no_grad():
        for idx, (batch, transcripts) in enumerate(dataloader):
            actual = transcripts[0].lower().replace(" ", "")

            # Timing
            t0 = time.time()
            predicted = model(batch)
            t1 = time.time()

            # RTF: processing_time / audio_duration
            # each mel frame = hop_length / sample_rate seconds
            frame_dur = model._mel_kwargs["hop_length"] / model._mel_kwargs["sample_rate"]
            audio_dur = batch.feature_lengths[0].item() * frame_dur

            # Piece-level encoding for WER
            ref_pieces = " ".join(sp.encode_as_pieces(actual))
            hyp_pieces = " ".join(sp.encode_as_pieces(predicted))

            wer = compute_wer(ref_pieces, hyp_pieces)
            rtf = (t1 - t0) / audio_dur if audio_dur > 0 else 0

            total_wer += wer
            total_rtf += rtf
            n += 1
            per_sample.append({"idx": idx, "WER": wer, "RTF": rtf, "ref": actual, "hyp": predicted})

            if idx % 100 == 0:
                logging.info(
                    "[%4d] WER: %.3f  RTF: %.4f  |  ref: %s  |  hyp: %s",
                    idx, wer, rtf,
                    ref_pieces[:60], hyp_pieces[:60],
                )

    avg_wer = total_wer / n if n > 0 else 0
    avg_rtf = total_rtf / n if n > 0 else 0

    logging.info("=" * 60)
    logging.info("Evaluation complete — %d samples", n)
    logging.info("WER: %.4f", avg_wer)
    logging.info("RTF: %.4f", avg_rtf)
    logging.info("=" * 60)

    # Save per-sample CSV
    import csv
    with open(per_sample_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "WER", "RTF", "ref", "hyp"])
        writer.writeheader()
        writer.writerows(per_sample)
    logging.info("Per-sample results saved to: %s", per_sample_file)

    return {"wer": avg_wer, "rtf": avg_rtf, "num_samples": n}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = ArgumentParser(description="Evaluate a Conformer-RNNT checkpoint.")
    parser.add_argument("config", help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to the .ckpt checkpoint to evaluate.",
    )
    parser.add_argument("--device", default="cuda", help="Device: cuda | cpu.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Logging
    fmt = "%(asctime)s %(message)s" if args.debug else "%(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

    # Config
    config = load_config(args.config)

    # Build model from config
    logging.info("Building model from config …")
    model = LibriSpeechRNNTModule.from_config(config)

    # Load checkpoint
    logging.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    # Eval
    device = torch.device(args.device)
    run_eval(model, device)


if __name__ == "__main__":
    main()
