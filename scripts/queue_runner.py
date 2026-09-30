#!/usr/bin/env python3
"""Serial experiment queue runner — tmux-resident, idempotent.

Runs one experiment per manifest line, in order:

    python train.py <config>            # until the process exits
    → verify val-*.ckpt exists          # training actually produced a model
    → python eval.py <config> -c <ckpt> --out <run_dir>/eval_greedy.csv
    → (optional beam eval) eval.py --beam <W>
    → append one row per eval to results/results.csv

Idempotency: a (run_name, eval_mode) pair already present in results.csv is
skipped, so the runner can be killed/restarted at any point without redoing
finished work. A crashed train is simply restarted from scratch on the next
invocation (checkpoints are save_weights_only=True, so resume is not usable —
training-plan §A6 note).

Manifest format (TSV, '#' comment lines allowed):
    run_name<TAB>config_path<TAB>eval_modes
where eval_modes is a comma-separated list drawn from:
    greedy | beam<W>   (e.g. "greedy,beam20"; default "greedy")

Usage (inside tmux):
    tmux new -s queue 'python scripts/queue_runner.py scripts/manifest_a4.tsv'
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
CSV_FIELDS = ["run_name", "eval_mode", "per", "rtf", "n", "ckpt", "config",
              "timestamp"]

# How long (s) to wait after train exits before giving up on a ckpt appearing
# (NFS write latency can lag process exit slightly).
CKPT_WAIT_SECS = 120


def load_done() -> set:
    """Read (run_name, eval_mode) pairs already recorded in results.csv."""
    done = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["run_name"], row["eval_mode"]))
    return done


def append_result(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)
    logging.info("recorded: %s / %s -> PER=%s RTF=%s",
                 row["run_name"], row["eval_mode"], row["per"], row["rtf"])


def find_ckpt(config_path: str) -> str | None:
    """Latest val-*.ckpt under the config's exp_dir (mtime order)."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    exp_dir = cfg.get("paths", {}).get("exp_dir", "./exp")
    ckpts = glob.glob(os.path.join(exp_dir, "checkpoints", "val-*.ckpt"))
    return max(ckpts, key=os.path.getmtime) if ckpts else None


def run_cmd(cmd: list, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("RUN: %s (log: %s)", " ".join(shlex.quote(c) for c in cmd),
                 log_path)
    t0 = time.time()
    with open(log_path, "a") as logf:
        logf.write(f"\n===== {datetime.now():%F %T} $ {' '.join(cmd)} =====\n")
        logf.flush()
        ret = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              cwd=str(REPO))
    logging.info("exit=%d elapsed=%.0fs", ret, time.time() - t0)
    return ret


def parse_result_line(log_path: Path) -> dict | None:
    """Extract the machine-readable 'RESULT per=... ' line from eval log."""
    try:
        for line in reversed(log_path.read_text(errors="replace").splitlines()):
            if line.startswith("RESULT "):
                fields = dict(kv.split("=", 1) for kv in line[len("RESULT "):].split())
                return fields
    except OSError:
        pass
    return None


def process_run(run_name: str, config_path: str, eval_modes: list,
                done: set) -> None:
    config_path = str(REPO / config_path) if not os.path.isabs(config_path) \
        else config_path
    if not os.path.isfile(config_path):
        logging.error("[%s] config missing: %s — SKIPPED", run_name, config_path)
        return

    ckpt = find_ckpt(config_path)
    if ckpt is None:
        # ── Train ─────────────────────────────────────────────────────────
        logging.info("[%s] training …", run_name)
        ret = run_cmd([sys.executable, "train.py", config_path],
                      REPO / "results" / "logs" / f"{run_name}.train.log")
        if ret != 0:
            logging.error("[%s] train.py exited %d — SKIPPED", run_name, ret)
            return
        # Wait for ckpt to materialize (NFS lag tolerance).
        deadline = time.time() + CKPT_WAIT_SECS
        while time.time() < deadline:
            ckpt = find_ckpt(config_path)
            if ckpt:
                break
            time.sleep(10)
        if ckpt is None:
            logging.error("[%s] no val-*.ckpt after train exit — SKIPPED",
                          run_name)
            return
    logging.info("[%s] ckpt: %s", run_name, os.path.basename(ckpt))

    # ── Evals ─────────────────────────────────────────────────────────────
    for mode in eval_modes:
        if (run_name, mode) in done:
            logging.info("[%s] %s already recorded — skip", run_name, mode)
            continue
        if mode == "greedy":
            beam, out_csv = 0, "eval_greedy.csv"
        elif mode.startswith("beam"):
            beam, out_csv = int(mode[4:]), f"eval_{mode}.csv"
        else:
            logging.error("[%s] unknown eval mode '%s'", run_name, mode)
            continue
        eval_log = REPO / "results" / "logs" / f"{run_name}.{mode}.log"
        ret = run_cmd([sys.executable, "eval.py", config_path,
                       "--checkpoint", ckpt, "--beam", str(beam),
                       "--out", str(REPO / "results" / "logs"
                                    / f"{run_name}.{mode}.per_sample.csv")],
                      eval_log)
        fields = parse_result_line(eval_log) if ret == 0 else None
        if fields is None:
            logging.error("[%s] %s: no RESULT line (exit=%d) — not recorded",
                          run_name, mode, ret)
            continue
        append_result({
            "run_name": run_name, "eval_mode": mode,
            "per": fields["per"], "rtf": fields["rtf"], "n": fields["n"],
            "ckpt": fields["ckpt"], "config": config_path,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        done.add((run_name, mode))


def read_manifest(path: str) -> list:
    entries = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                logging.error("bad manifest line: %r", raw.rstrip())
                continue
            entries.append((parts[0], parts[1],
                            (parts[2] if len(parts) > 2 else "greedy").split(",")))
    return entries


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    logging.basicConfig(
        format="%(asctime)s %(message)s", level=logging.INFO,
        datefmt="%F %T",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(REPO / "results" / "queue_runner.log")],
    )
    entries = read_manifest(sys.argv[1])
    done = load_done()
    n_evals = sum(len(ms) for _n, _c, ms in entries)
    n_done = sum(1 for n, _c, ms in entries for m in ms if (n, m) in done)
    logging.info("manifest: %d runs / %d evals, %d already finished",
                 len(entries), n_evals, n_done)
    for i, (run_name, config_path, eval_modes) in enumerate(entries, 1):
        logging.info("=== [%d/%d] %s ===", i, len(entries), run_name)
        process_run(run_name, config_path, eval_modes, done)
    logging.info("queue drained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
