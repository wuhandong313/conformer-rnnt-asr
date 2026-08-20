#!/usr/bin/env python3
"""Evaluate a trained Conformer-RNNT checkpoint.

Usage:
    python eval.py configs/server.yaml --checkpoint exp/checkpoints/val-epoch=02-step=4000.ckpt
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import sys
import time
from argparse import ArgumentParser
from typing import Any, Dict

import torch
import torchaudio

from common import checkpoint_dir, load_config, model_ckpt_tag, setup_logging
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule


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


def run_eval(model: LibriSpeechRNNTModule, device: torch.device, per_sample_file: str = "eval_per_sample.csv") -> Dict[str, Any]:
    model = model.to(device)
    model.eval()
    dataloader = model.test_dataloader()

    total_per = 0.0
    total_rtf = 0.0
    n = 0
    per_sample = []

    # Medusa 投机解码接受率 — 跨样本聚合 (仅 spec_greedy 模式填充)
    med_drafts = 0
    med_accepted = 0
    med_per_head: Dict[int, Dict[str, int]] = {}  # head_idx -> {drafts, accepted}
    med_samples = 0  # 有 draft 统计的样本数

    with torch.no_grad():
        for idx, (batch, transcripts) in enumerate(dataloader):
            actual = transcripts[0].lower().replace(" ", "")

            # Timing
            t0 = time.time()
            predicted, stats = model(batch)
            t1 = time.time()

            # RTF: processing_time / audio_duration
            frame_dur = model._mel_kwargs["hop_length"] / model._mel_kwargs["sample_rate"]
            audio_dur = batch.feature_lengths[0].item() * frame_dur

            # Aggregate Medusa accept stats (spec_greedy only)
            if stats is not None and stats["total_drafts"] > 0:
                med_drafts += stats["total_drafts"]
                med_accepted += stats["total_accepted"]
                for i, ph in enumerate(stats["per_head"]):
                    head = med_per_head.setdefault(i, {"drafts": 0, "accepted": 0})
                    head["drafts"] += ph["drafts"]
                    head["accepted"] += ph["accepted"]
                med_samples += 1

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
    if med_drafts > 0:
        per_head_txt = " | ".join(
            f"h{i+1}: {d['accepted']}/{d['drafts']} ({100*d['accepted']/max(d['drafts'],1):.1f}%)"
            for i, d in sorted(med_per_head.items())
        )
        logging.info("Medusa accept rate (over %d spec samples):", med_samples)
        logging.info(
            "  overall: %d/%d (%.2f%%) | per-head: %s",
            med_accepted, med_drafts, 100 * med_accepted / med_drafts, per_head_txt,
        )
    else:
        logging.info(
            "Medusa accept stats unavailable (decoding.mode != spec_greedy, "
            "no Medusa heads, or all samples fell back to beam)."
        )
    logging.info("=" * 60)

    # Save per-sample CSV
    with open(per_sample_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "PER", "RTF", "ref", "hyp"])
        writer.writeheader()
        writer.writerows(per_sample)
    logging.info("Per-sample results saved to: %s", per_sample_file)

    summary: Dict[str, Any] = {
        "per": avg_per,
        "rtf": avg_rtf,
        "num_samples": n,
    }
    if med_drafts > 0:
        summary["medusa_accept_rate"] = med_accepted / med_drafts
        summary["medusa_total_drafts"] = med_drafts
        summary["medusa_total_accepted"] = med_accepted
        summary["medusa_per_head"] = [
            {"drafts": d["drafts"], "accepted": d["accepted"],
             "rate": d["accepted"] / max(d["drafts"], 1)}
            for _, d in sorted(med_per_head.items())
        ]
        summary["medusa_samples"] = med_samples
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def _extract_ckpt_snapshot(ckpt: Dict[str, Any]) -> Dict[str, Any] | None:
    """从 checkpoint dict 读取训练时写入的 _config_snapshot (由 train.py 的 CheckpointSnapshot 写入)。

    旧 checkpoint 没有该 key 时返回 None —— 无法校验, 但可正常加载。
    """
    snap = ckpt.get("_config_snapshot") if isinstance(ckpt, dict) else None
    if not isinstance(snap, dict):
        return None
    return snap


def _check_snapshot_mismatch(snapshot: Dict[str, Any] | None, config: Dict[str, Any]) -> None:
    """校验 checkpoint 训练时的真实配置 vs 当前 eval config, 不一致则告警。

    这是对「文件名可能和真实训练模式不符」的兜底: 即使文件名/自动加载选错,
    也能从 checkpoint 内嵌快照发现配置不匹配 (如 compression.filter、medusa.num_heads 等)。
    """
    if snapshot is None:
        logging.warning(
            "Checkpoint has no _config_snapshot (old checkpoint) — cannot verify "
            "it was trained with the current config. Results may be invalid if "
            "compression.filter / medusa.num_heads differ from the eval config."
        )
        return

    def _get(cfg: Dict[str, Any], dotted_path: str):
        cur = cfg
        for part in dotted_path.split(":"):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
        return cur

    pairs = [
        ("compression.filter",
         _get(snapshot, "model:compression:filter"),
         _get(config, "model:compression:filter")),
        ("medusa.num_heads",
         _get(snapshot, "model:medusa:num_heads"),
         _get(config, "model:medusa:num_heads")),
        ("medusa.freeze_base",
         _get(snapshot, "model:medusa:freeze_base"),
         _get(config, "model:medusa:freeze_base")),
        ("decoding.mode",
         _get(snapshot, "decoding:mode"),
         _get(config, "decoding:mode")),
    ]
    for name, snap_val, cfg_val in pairs:
        if snap_val != cfg_val:
            logging.warning(
                "SNAPSHOT MISMATCH: %s — checkpoint trained with %r, "
                "eval config has %r. Results may be invalid (e.g. frame "
                "alignment mismatch from wrong compression.filter).",
                name, snap_val, cfg_val,
            )


def _find_latest_val_ckpt(ckpt_dir: str, tag: str | None = None) -> str | None:
    """Find the latest val-*.ckpt in the checkpoint directory.

    When ``tag`` (e.g. ``"naive-h1-free"``) is given, only checkpoints matching
    ``val-{tag}-*.ckpt`` are candidates — avoids auto-loading a checkpoint
    from a different experiment (different medusa heads / compression mode).
    Falls back to any ``val-*.ckpt`` if no tagged one exists (legacy format).
    """
    if tag:
        tagged = glob.glob(os.path.join(ckpt_dir, f"val-{tag}-*.ckpt"))
        if tagged:
            return max(tagged, key=os.path.getmtime)
        logging.warning("No val-%s-*.ckpt found; falling back to any val-*.ckpt", tag)
    files = glob.glob(os.path.join(ckpt_dir, "val-*.ckpt"))
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
    args = parser.parse_args()

    # Config
    config = load_config(args.config)

    # Logging
    setup_logging(config)

    # Build model from config (architecture, paths — all from YAML)
    logging.info("Building model from config …")
    model = LibriSpeechRNNTModule.from_config(config)
    logging.info(
        "Decoding: mode=%s beam_width=%s",
        model.decoding_mode, model.beam_width,
    )

    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_dir = checkpoint_dir(config)
        ckpt_path = _find_latest_val_ckpt(ckpt_dir, model_ckpt_tag(config))
        if ckpt_path is None:
            logging.error("No val-*.ckpt found in %s. Specify --checkpoint.", ckpt_dir)
            sys.exit(1)
        logging.info("Auto-selected checkpoint: %s", ckpt_path)

    # Load checkpoint
    logging.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 校验 checkpoint 训练时的真实配置 vs 当前 eval config (内嵌快照)
    snapshot = _extract_ckpt_snapshot(ckpt)
    _check_snapshot_mismatch(snapshot, config)

    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    # Eval
    accel = config.get("trainer", {}).get("accelerator", "cuda")
    device_name = "cpu" if args.cpu else ("cuda" if accel == "gpu" else accel)
    run_eval(model, torch.device(device_name))


if __name__ == "__main__":
    main()
