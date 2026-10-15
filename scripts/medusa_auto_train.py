#!/usr/bin/env python3
"""Run staged Medusa experiments until the acceptance gate is met.

The script is intentionally a small orchestration layer around the repository's
existing ``train.py`` and ``eval.py`` entry points. It keeps each experiment in
its own directory, evaluates new validation checkpoints periodically, resumes
from the newest checkpoint after a training failure, and performs full dev/test
evaluation once a candidate passes the fixed-subset gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


REPO = Path(__file__).resolve().parents[1]


STAGES = [
    {"name": "stream_h1_kl0", "heads": 1, "mode": "stream_local", "lookahead": 1, "kl": 0.0, "freeze": True, "lr": 5e-4},
    {"name": "stream_h2_kl0", "heads": 2, "mode": "stream_local", "lookahead": 1, "kl": 0.0, "freeze": True, "lr": 5e-4},
    {"name": "stream_h2_kl001", "heads": 2, "mode": "stream_local", "lookahead": 1, "kl": 0.01, "freeze": True, "lr": 5e-4},
    {"name": "stream_h2_kl005", "heads": 2, "mode": "stream_local", "lookahead": 1, "kl": 0.05, "freeze": True, "lr": 5e-4},
    {"name": "predictor_h1_kl0", "heads": 1, "mode": "predictor_only", "lookahead": 0, "kl": 0.0, "freeze": True, "lr": 5e-4},
    {"name": "stream_h2_unfreeze", "heads": 2, "mode": "stream_local", "lookahead": 1, "kl": 0.01, "freeze": False, "lr": 5e-5},
]


def latest_checkpoint(directory: Path) -> Optional[Path]:
    candidates = list((directory / "checkpoints").glob("val-*.ckpt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_yaml(path: Path, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def make_config(
    template: Dict[str, Any],
    *,
    exp_dir: Path,
    checkpoint: Optional[Path],
    stage: Optional[Dict[str, Any]],
    subset: str = "dev-clean",
) -> Dict[str, Any]:
    config = copy.deepcopy(template)
    paths = config.setdefault("paths", {})
    paths["exp_dir"] = str(exp_dir)
    paths["from_checkpoint"] = str(checkpoint) if checkpoint else None

    model = config.setdefault("model", {})
    medusa = model.setdefault("medusa", {})
    if stage is None:
        medusa.update({
            "num_heads": 0,
            "input_mode": "stream_local",
            "lookahead_frames": 0,
            "draft_after_emit": False,
            "accept_ce_weight": 0.0,
            "freeze_base": False,
            "kl_weight": 0.0,
        })
    else:
        medusa.update({
            "num_heads": stage["heads"],
            "input_mode": stage["mode"],
            "lookahead_frames": stage["lookahead"],
            "draft_after_emit": True,
            "accept_ce_weight": 1.0,
            "freeze_base": stage["freeze"],
            "kl_weight": stage["kl"],
        })

    data = config.setdefault("data", {})
    data["val_subset"] = subset
    data["test_subset"] = subset
    decoding = config.setdefault("decoding", {})
    # A spec-greedy failure must fall back to the width-1 oracle, never to the
    # template's historical beam width, so PER comparisons stay meaningful.
    decoding["beam_width"] = 0
    decoding["mode"] = "beam" if stage is None else "spec_greedy"
    if stage is not None:
        decoding.setdefault("spec", {}).update({"max_draft": stage["heads"], "per_token_blank": 4})
        config.setdefault("training", {})["learning_rate"] = stage["lr"]
    return config


def run_process(command: Iterable[str], log_path: Path, *, check: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        result = subprocess.run(
            list(command), cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=False,
        )
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    return result.returncode


def evaluate_checkpoint(
    config_path: Path,
    checkpoint: Path,
    eval_dir: Path,
    max_samples: Optional[int],
    *,
    check_lossless: bool,
) -> Optional[Dict[str, Any]]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / f"{checkpoint.stem}.json"
    if metrics_path.exists():
        return load_json(metrics_path)
    csv_path = eval_dir / f"{checkpoint.stem}.csv"
    command = [
        sys.executable, "eval.py", str(config_path), "--checkpoint", str(checkpoint),
        "--output-json", str(metrics_path), "--per-sample-file", str(csv_path),
    ]
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    if check_lossless:
        command.append("--check-lossless")
    returncode = run_process(command, eval_dir / "artifacts/logs/eval.log")
    if returncode != 0 or not metrics_path.exists():
        return None
    return load_json(metrics_path)


def passes_gate(metrics: Dict[str, Any], baseline_per: float) -> bool:
    heads = metrics.get("medusa_per_head", [])
    head1 = heads[0].get("rate", 0.0) if heads else 0.0
    return (
        metrics.get("medusa_accept_rate", 0.0) >= 0.50
        and head1 >= 0.60
        and metrics.get("lossless_checked", 0) > 0
        and metrics.get("lossless_failures", 1) == 0
        and abs(metrics.get("per", float("inf")) - baseline_per) <= 0.005
    )


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def evaluate_full_winner(
    template: Dict[str, Any],
    winner_config: Dict[str, Any],
    checkpoint: Path,
    winner_dir: Path,
) -> None:
    for subset in ("dev-clean", "test-clean"):
        config = copy.deepcopy(winner_config)
        config.setdefault("data", {})["test_subset"] = subset
        config_path = winner_dir / f"winner_{subset}.yaml"
        write_yaml(config_path, config)
        evaluate_checkpoint(
            config_path, checkpoint, winner_dir / subset, None, check_lossless=True
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-config", default="configs/server.yaml")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--root-exp", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument(
        "--start-stage", type=int, default=0,
        help="Start at a zero-based stage index; earlier stages are preserved.",
    )
    parser.add_argument(
        "--resume-checkpoint", default=None,
        help="Checkpoint used to initialize the first selected stage.",
    )
    parser.add_argument(
        "--stage-epochs", type=int, default=None,
        help="Optional epoch cap for each staged experiment.",
    )
    parser.add_argument(
        "--stage-warmup-steps", type=int, default=None,
        help="Optional warmup length for staged head-only optimization.",
    )
    parser.add_argument(
        "--cycle-start", type=int, default=1,
        help="First cycle number; use a new number to preserve prior cycles.",
    )
    args = parser.parse_args()

    if not 0 <= args.start_stage < len(STAGES):
        parser.error(f"--start-stage must be in [0, {len(STAGES) - 1}]")
    if args.cycle_start < 1:
        parser.error("--cycle-start must be positive")

    root = Path(args.root_exp)
    root.mkdir(parents=True, exist_ok=True)
    template_path = (REPO / args.template_config).resolve() if not Path(args.template_config).is_absolute() else Path(args.template_config)
    with template_path.open() as handle:
        template = yaml.safe_load(handle)
    base_checkpoint = Path(args.base_checkpoint)

    baseline_dir = root / "baseline"
    baseline_config = make_config(
        template, exp_dir=baseline_dir, checkpoint=base_checkpoint, stage=None
    )
    baseline_config_path = baseline_dir / "config.yaml"
    write_yaml(baseline_config_path, baseline_config)
    baseline_metrics_path = baseline_dir / "baseline.json"
    if baseline_metrics_path.exists():
        baseline = load_json(baseline_metrics_path)
    else:
        baseline = evaluate_checkpoint(
            baseline_config_path,
            base_checkpoint,
            baseline_dir,
            args.max_samples,
            check_lossless=False,
        )
        if baseline is None:
            print("baseline evaluation failed; see", baseline_dir / "artifacts/logs/eval.log", flush=True)
            return 2
        baseline_metrics_path.write_text(json.dumps(baseline, indent=2, sort_keys=True))
    baseline_per = float(baseline["per"])
    print(f"baseline PER={baseline_per:.6f}", flush=True)

    resume_by_stage: Dict[str, Path] = {}
    resume_seed = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    cycle = args.cycle_start - 1
    while True:
        cycle += 1
        print(
            f"===== Medusa experiment cycle {cycle} "
            f"(start_stage={args.start_stage}) =====",
            flush=True,
        )
        had_checkpoint = False
        for stage_index, stage in enumerate(STAGES):
            if stage_index < args.start_stage:
                continue
            stage_name = stage["name"]
            stage_dir = root / f"cycle-{cycle:03d}" / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)
            resume = resume_by_stage.get(stage_name)
            if resume is None:
                resume = resume_seed if stage_index == args.start_stage and resume_seed else base_checkpoint
            config = make_config(
                template, exp_dir=stage_dir, checkpoint=resume, stage=stage
            )
            if args.stage_epochs is not None:
                config.setdefault("training", {})["epochs"] = args.stage_epochs
            if args.stage_warmup_steps is not None:
                config.setdefault("training", {})["warmup_steps"] = args.stage_warmup_steps
            config_path = stage_dir / "config.yaml"
            write_yaml(config_path, config)
            print(
                f"[{stage_name}] start from {resume} | "
                f"mode={stage['mode']} K={stage['heads']} KL={stage['kl']} "
                f"freeze={stage['freeze']}",
                flush=True,
            )

            train_log = stage_dir / "artifacts/logs/train.log"
            process = subprocess.Popen(
                [sys.executable, "train.py", str(config_path)],
                cwd=REPO,
                stdout=train_log.open("a"),
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
            evaluated = set()
            winner = None
            while True:
                checkpoint = latest_checkpoint(stage_dir)
                if checkpoint is not None and str(checkpoint) not in evaluated:
                    had_checkpoint = True
                    metrics = evaluate_checkpoint(
                        config_path,
                        checkpoint,
                        stage_dir / "eval",
                        args.max_samples,
                        check_lossless=True,
                    )
                    evaluated.add(str(checkpoint))
                    if metrics is not None:
                        status = {
                            "cycle": cycle,
                            "stage": stage_name,
                            "checkpoint": str(checkpoint),
                            "metrics": metrics,
                            "passes": passes_gate(metrics, baseline_per),
                        }
                        (stage_dir / "latest_status.json").write_text(
                            json.dumps(status, indent=2, sort_keys=True)
                        )
                        print(
                            f"[{stage_name}] ckpt={checkpoint.name} "
                            f"accept={metrics.get('medusa_accept_rate', 0):.4f} "
                            f"head1={metrics.get('medusa_per_head', [{}])[0].get('rate', 0):.4f} "
                            f"PER={metrics.get('per', float('nan')):.4f} "
                            f"lossless_failures={metrics.get('lossless_failures', -1)}",
                            flush=True,
                        )
                        if passes_gate(metrics, baseline_per):
                            winner = (checkpoint, config, metrics)
                            break

                if process.poll() is not None:
                    break
                time.sleep(max(30, args.poll_seconds))

            if winner is not None:
                checkpoint, winner_config, metrics = winner
                stop_process(process)
                winner_dir = root / "WINNER"
                winner_dir.mkdir(parents=True, exist_ok=True)
                (winner_dir / "winner.json").write_text(
                    json.dumps(
                        {"stage": stage_name, "checkpoint": str(checkpoint), "metrics": metrics},
                        indent=2,
                        sort_keys=True,
                    )
                )
                evaluate_full_winner(template, winner_config, checkpoint, winner_dir)
                print(f"ACCEPTANCE GATE PASSED: {checkpoint}", flush=True)
                return 0

            latest = latest_checkpoint(stage_dir)
            if latest is not None:
                had_checkpoint = True
                resume_by_stage[stage_name] = latest
                print(f"[{stage_name}] finished; next cycle resumes from {latest}", flush=True)
            elif process.returncode not in (0, None):
                print(f"[{stage_name}] failed without checkpoint; see {train_log}", flush=True)

        if not had_checkpoint:
            print("all stages failed without producing a checkpoint", flush=True)
            return 3
        print("no stage passed; continuing with the next cycle", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
