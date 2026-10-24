#!/usr/bin/env python3
"""Evaluate a trained Conformer-RNNT checkpoint.

Usage:
    python eval.py configs/server.yaml --checkpoint exp/checkpoints/val-epoch=02-step=4000.ckpt
    python eval.py configs/server_lstm_word.yaml --checkpoint exp/checkpoints/best.ckpt --metric wer
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
from conformer_rnnt.metrics import format_wer_trace, phone_error_counts, word_error_counts


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


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute character error rate using character-level edit distance."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return torchaudio.functional.edit_distance(reference, hypothesis) / len(reference)


# ═══════════════════════════════════════════════════════════════════════════
# Eval loop
# ═══════════════════════════════════════════════════════════════════════════


def run_eval(
    model: LibriSpeechRNNTModule,
    device: torch.device,
    per_sample_file: str = "artifacts/evaluations/eval_per_sample.csv",
    max_utts: int = 0,
    metric: str = "wer",
    trace_wer: bool = False,
) -> Dict[str, Any]:
    if model.tokenizer_type in {"aishell_lexicon", "mandarin_syllable"}:
        evaluator = "eval_mandarin_syllables.py" if model.tokenizer_type == "mandarin_syllable" else "eval_aishell1.py"
        raise ValueError(f"Use scripts/evaluation/{evaluator}; Chinese evaluation does not report RTF")
    if metric not in {"wer", "cer", "per"}:
        raise ValueError(f"Unsupported evaluation metric: {metric}")
    if (model.tokenizer_type in {"ipa", "phone69", "aishell_lexicon"}) != (metric == "per"):
        raise ValueError("Fixed phone evaluation requires PER; PER requires tokenizer.type: ipa or phone69")
    if metric == "wer" and model.strip_spaces:
        raise ValueError("WER requires a word-text config with data.strip_spaces: false")
    if trace_wer and metric != "wer":
        raise ValueError("--trace-wer requires WER mode")

    model = model.to(device)
    model.eval()
    dataloader = model.test_dataloader()

    total_cer = 0.0
    total_word_errors = 0
    total_reference_words = 0
    total_phone_errors = 0
    total_reference_phones = 0
    total_rtf = 0.0
    n = 0
    per_sample = []

    with torch.no_grad():
        for idx, (batch, transcripts) in enumerate(dataloader):
            if max_utts and n >= max_utts:
                break
            actual = (
                " ".join(model.tokenizer.tokenize(transcripts[0]))
                if metric == "per" else transcripts[0].lower()
            )
            if metric == "cer" and model.strip_spaces:
                actual = actual.replace(" ", "")
            if trace_wer:
                # The test loader concatenates subsets in config order, batch_size=1.
                sample_idx = idx
                for subset in dataloader.dataset.datasets:
                    if sample_idx < len(subset):
                        metadata = subset.get_metadata(sample_idx)
                        logging.info("TRACE sample=%d audio=%s sample_rate=%s", idx, metadata[0], metadata[1])
                        break
                    sample_idx -= len(subset)
                logging.info(
                    "TRACE feature_shape=%s feature_lengths=%s target_lengths=%s beam_width=%s",
                    tuple(batch.features.shape), batch.feature_lengths.tolist(),
                    batch.target_lengths.tolist(), model.beam_width,
                )

            # Timing
            t0 = time.time()
            predicted = model(batch)
            t1 = time.time()

            # RTF: processing_time / audio_duration
            frame_dur = model._mel_kwargs["hop_length"] / model._mel_kwargs["sample_rate"]
            audio_dur = batch.feature_lengths[0].item() * frame_dur

            rtf = (t1 - t0) / audio_dur if audio_dur > 0 else 0

            total_rtf += rtf
            n += 1

            if metric == "wer":
                word_errors, reference_words = word_error_counts(actual, predicted)
                sample_wer = (
                    word_errors / reference_words
                    if reference_words
                    else (0.0 if word_errors == 0 else float("inf"))
                )
                total_word_errors += word_errors
                total_reference_words += reference_words
                per_sample.append({
                    "idx": idx,
                    "WER": sample_wer,
                    "word_errors": word_errors,
                    "reference_words": reference_words,
                    "RTF": rtf,
                    "ref": actual,
                    "hyp": predicted,
                })
                cumulative_rate = (
                    total_word_errors / total_reference_words
                    if total_reference_words
                    else 0.0
                )
                sample_rate = sample_wer
                metric_name = "WER"
                if trace_wer:
                    logging.info("\n%s", format_wer_trace(actual, predicted))
                    logging.info(
                        "TRACE cumulative WER = %d/%d = %.8f (%.4f%%)",
                        total_word_errors, total_reference_words, cumulative_rate,
                        100 * cumulative_rate,
                    )
            elif metric == "per":
                phone_errors, reference_phones = phone_error_counts(actual, predicted)
                sample_rate = (
                    phone_errors / reference_phones
                    if reference_phones else (0.0 if phone_errors == 0 else float("inf"))
                )
                total_phone_errors += phone_errors
                total_reference_phones += reference_phones
                per_sample.append({
                    "idx": idx,
                    "PER": sample_rate,
                    "phone_errors": phone_errors,
                    "reference_phones": reference_phones,
                    "RTF": rtf,
                    "ref": actual,
                    "hyp": predicted,
                })
                cumulative_rate = (
                    total_phone_errors / total_reference_phones
                    if total_reference_phones else (0.0 if total_phone_errors == 0 else float("inf"))
                )
                metric_name = "PER"
            else:
                sample_cer = compute_cer(actual, predicted)
                total_cer += sample_cer
                per_sample.append({
                    "idx": idx,
                    "CER": sample_cer,
                    "RTF": rtf,
                    "ref": actual,
                    "hyp": predicted,
                })
                cumulative_rate = total_cer / n
                sample_rate = sample_cer
                metric_name = "CER"

            if idx % 5 == 0:
                cum_rtf = total_rtf / n if n > 0 else 0
                logging.info(
                    "[%4d] %s: %.3f (cum: %.3f)  RTF: %.4f (cum: %.4f)  |  ref: %s  |  hyp: %s",
                    idx, metric_name, sample_rate, cumulative_rate, rtf, cum_rtf,
                    actual[:60], predicted[:60],
                )

    avg_rtf = total_rtf / n if n > 0 else 0

    logging.info("=" * 60)
    logging.info("Evaluation complete — %d samples", n)
    if metric == "wer":
        avg_rate = (
            total_word_errors / total_reference_words
            if total_reference_words
            else 0.0
        )
        logging.info(
            "WER: %.4f (%d word errors / %d reference words)",
            avg_rate,
            total_word_errors,
            total_reference_words,
        )
    elif metric == "per":
        avg_rate = (
            total_phone_errors / total_reference_phones
            if total_reference_phones else (0.0 if total_phone_errors == 0 else float("inf"))
        )
        logging.info(
            "PER: %.4f (%d phone errors / %d reference phones)",
            avg_rate, total_phone_errors, total_reference_phones,
        )
    else:
        avg_rate = total_cer / n if n > 0 else 0.0
        logging.info("CER: %.4f (mean per-utterance character error rate)", avg_rate)
    logging.info("RTF: %.4f", avg_rtf)
    logging.info("=" * 60)

    # Save per-sample CSV
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(per_sample_file)), exist_ok=True)
    with open(per_sample_file, "w", newline="") as f:
        fieldnames = {
            "wer": ["idx", "WER", "word_errors", "reference_words", "RTF", "ref", "hyp"],
            "per": ["idx", "PER", "phone_errors", "reference_phones", "RTF", "ref", "hyp"],
            "cer": ["idx", "CER", "RTF", "ref", "hyp"],
        }[metric]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample)
    logging.info("Per-sample results saved to: %s", per_sample_file)

    result = {metric: avg_rate, "rtf": avg_rtf, "num_samples": n}
    if metric == "wer":
        result.update({
            "word_errors": total_word_errors,
            "reference_words": total_reference_words,
        })
    elif metric == "per":
        result.update({
            "phone_errors": total_phone_errors,
            "reference_phones": total_reference_phones,
        })
    return result


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
    parser.add_argument("--trace-wer", action="store_true", help="Print input metadata, word tokens, DP rows, and WER alignment for every sample.")
    parser.add_argument(
        "--metric",
        choices=("auto", "wer", "cer", "per"),
        default="auto",
        help=(
            "auto selects PER for IPA/phone69, otherwise WER when data.strip_spaces is false or CER; "
            "WER lowercases and splits on whitespace, preserving punctuation."
        ),
    )
    args = parser.parse_args()

    # Config
    config = load_config(args.config)
    if config.get("tokenizer", {}).get("type") in {"aishell_lexicon", "mandarin_syllable"}:
        evaluator = "eval_mandarin_syllables.py" if config["tokenizer"]["type"] == "mandarin_syllable" else "eval_aishell1.py"
        raise SystemExit(f"Use scripts/evaluation/{evaluator} for frozen Chinese symbol errors, S/D/I, tones and deciles without RTF")

    # Logging
    debug = config.get("debug", False)
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

    # Build model from config (architecture, paths — all from YAML)
    logging.info("Building model from config …")
    model = LibriSpeechRNNTModule.from_config(config)
    metric = args.metric
    if metric == "auto":
        metric = "per" if model.tokenizer_type in {"ipa", "phone69", "aishell_lexicon"} else ("cer" if model.strip_spaces else "wer")
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
    model.validate_tokenizer_checkpoint(ckpt)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=model.tokenizer_type in {"ipa", "phone69", "aishell_lexicon"})

    # Eval
    device = torch.device("cpu" if args.cpu else config.get("trainer", {}).get("accelerator", "cuda"))
    run_eval(
        model,
        device,
        per_sample_file=args.out,
        max_utts=args.max_utts,
        metric=metric,
        trace_wer=args.trace_wer,
    )


if __name__ == "__main__":
    main()
