#!/usr/bin/env python3
"""Evaluate a trained Conformer-RNNT checkpoint.

Usage:
    python eval.py configs/server_phone69_460h_offline_native_stride4_none_s42_e50.yaml --checkpoint /path/to/model.ckpt --cpu
    python eval.py configs/server_phone69_460h_offline_native_stride4_none_s42_e50.yaml --checkpoint /path/to/model.ckpt --metric per
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
from conformer_rnnt.metrics import phone_error_counts


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




# ═══════════════════════════════════════════════════════════════════════════
# Eval loop
# ═══════════════════════════════════════════════════════════════════════════


def run_eval(
    model: LibriSpeechRNNTModule,
    device: torch.device,
    per_sample_file: str = "artifacts/evaluations/eval_per_sample.csv",
    max_utts: int = 0,
    metric: str = "per",
) -> Dict[str, Any]:
    """Evaluate fixed Phone69 labels; RTF is the mean of per-utterance ratios."""
    if metric != "per" or model.tokenizer_type != "phone69":
        raise ValueError("Evaluation requires tokenizer.type: phone69 and metric PER")
    model = model.to(device)
    model.eval()
    total_errors = total_phones = 0
    total_rtf = 0.0
    per_sample = []
    with torch.no_grad():
        for idx, (batch, transcripts) in enumerate(model.test_dataloader()):
            if max_utts and len(per_sample) >= max_utts:
                break
            actual = " ".join(model.tokenizer.tokenize(transcripts[0]))
            started = time.perf_counter()
            predicted = model(batch)
            elapsed = time.perf_counter() - started
            frame_seconds = model._mel_kwargs["hop_length"] / model._mel_kwargs["sample_rate"]
            audio_seconds = batch.feature_lengths[0].item() * frame_seconds
            rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0
            errors, phones = phone_error_counts(actual, predicted)
            rate = errors / phones if phones else (0.0 if errors == 0 else float("inf"))
            total_errors += errors
            total_phones += phones
            total_rtf += rtf
            per_sample.append(dict(idx=idx, PER=rate, phone_errors=errors,
                                   reference_phones=phones, RTF=rtf, ref=actual, hyp=predicted))
            cumulative = total_errors / total_phones if total_phones else 0.0
            logging.info("[%4d] PER: %.3f (cum: %.3f) RTF: %.4f | ref: %s | hyp: %s",
                         idx, rate, cumulative, rtf, actual[:60], predicted[:60])
    count = len(per_sample)
    corpus_per = total_errors / total_phones if total_phones else (0.0 if total_errors == 0 else float("inf"))
    mean_rtf = total_rtf / count if count else 0.0
    logging.info("Evaluation complete: %d samples; corpus PER %.4f (%d / %d)",
                 count, corpus_per, total_errors, total_phones)
    logging.info("Mean utterance RTF: %.4f (feature-duration denominator)", mean_rtf)
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(per_sample_file)), exist_ok=True)
    with open(per_sample_file, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["idx", "PER", "phone_errors", "reference_phones", "RTF", "ref", "hyp"])
        writer.writeheader()
        writer.writerows(per_sample)
    return dict(per=corpus_per, rtf=mean_rtf, num_samples=count,
                phone_errors=total_errors, reference_phones=total_phones)


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
    parser.add_argument("--out", default="artifacts/evaluations/eval_per_sample.csv", help="Path for the per-sample CSV.")
    parser.add_argument(
        "--metric",
        choices=("auto", "per"),
        default="auto",
        help="Phone69 evaluation reports corpus PER.",
    )
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
    metric = args.metric
    if metric == "auto":
        metric = "per"
    logging.info("Evaluation metric: %s", metric.upper())
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
    if config.get("model", {}).get("compression", {}).get("stages") is not None:
        model.on_load_checkpoint(ckpt)
    else:
        model.validate_tokenizer_checkpoint(ckpt)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)

    # Eval
    device = torch.device("cpu" if args.cpu else config.get("trainer", {}).get("accelerator", "cuda"))
    run_eval(
        model,
        device,
        per_sample_file=args.out,
        max_utts=args.max_utts,
        metric=metric,
    )


if __name__ == "__main__":
    main()
