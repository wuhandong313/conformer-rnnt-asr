#!/usr/bin/env python3
"""Read-only Chinese queue probe. Compare file ages only to this server's UTC clock."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def snapshot(audit):
    import yaml
    now = datetime.now(timezone.utc)
    audit = Path(audit)

    def read(path):
        return json.loads(path.read_text()) if path.is_file() else None

    def age(path):
        return (now.timestamp() - path.stat().st_mtime) if path.exists() else None

    gate, queue = read(audit / "prelaunch_validation.json"), read(audit / "queue_state.json")
    configs = gate.get("configs", []) if gate else []
    if not configs:
        configs = sorted((REPO / "configs").glob("server_aishell1_lexicon_*_s42_e50.yaml"))
    models = []
    for path in configs:
        config = yaml.safe_load(Path(path).read_text())
        exp = REPO / config["paths"]["exp_dir"]
        run = read(exp / "run.json")
        logs = list((exp / "logs").glob("train_*.log"))
        checkpoints = list((exp / "checkpoints").glob("*.ckpt"))
        latest_log = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
        latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else None
        ages = {"log": age(latest_log) if latest_log else None,
                "progress": age(exp / "run.json"), "checkpoint": age(latest_ckpt) if latest_ckpt else None}
        pid = (run or {}).get("pid")
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                command = subprocess.check_output(["ps", "-p", str(pid), "-o", "args="], text=True)
                alive = "train_aishell1_one.py" in command and str(path) in command
            except (ProcessLookupError, PermissionError):
                pass
            except subprocess.CalledProcessError:
                pass
        models.append({"experiment": exp.name, "run": run, "process_alive": alive,
                       "complete_marker": (exp / "TRAINING_COMPLETE").exists(), "ages_seconds": ages,
                       "possible_stall": bool(run and run.get("status") == "training"
                                               and all(a is not None and a > 3600 for a in ages.values())),
                       "latest_log": str(latest_log) if latest_log else None,
                       "latest_checkpoint": str(latest_ckpt) if latest_ckpt else None})
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                       "--format=csv,noheader"], text=True).strip()
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                             "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu, processes = "unavailable", "unavailable"
    return {"server_utc": now.isoformat(), "phase": "aishell1_lexicon_20261024", "queue": queue,
            "models": models, "gpu_memory_and_compute": gpu, "gpu_processes": processes,
            "training_started": any(m["run"] for m in models), "automatic_restart": False,
            "notice": "A stale triad is a signal to inspect real processes/GPU, never authorization to restart"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", default="artifacts/runs/aishell1_lexicon_20261024")
    args = parser.parse_args()
    os.chdir(REPO)
    print(json.dumps(snapshot(args.audit_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
