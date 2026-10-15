"""Train the fixed-phone clean-100 baseline, then evaluate its best checkpoint.

Run from any directory with the server training Python. The recipe is fixed in
configs/server_phone69_100h_none.yaml; the training entry point is unchanged.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/server_phone69_100h_none.yaml"


def resume_metadata(state_path, cfg):
    """Require an explicit full checkpoint and an idle, unfinished run."""
    source = cfg["paths"].get("from_checkpoint")
    if not state_path.exists():
        if source:
            raise RuntimeError("Resume requires the original run.json")
        return {}, {}
    previous = json.loads(state_path.read_text())
    if not source or previous["status"] == "completed":
        raise RuntimeError("Existing experiment requires an explicit resume checkpoint")
    for key in ("pipeline_pid", "child_pid"):
        if previous.get(key) and Path(f"/proc/{previous[key]}").exists():
            raise RuntimeError(f"Previous {key} is still alive")
    import torch
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers"):
        raise RuntimeError("Resume checkpoint must include optimizer and scheduler state")
    progress = checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]
    old_budget = previous.get("max_token_limit", 4000)
    if old_budget != cfg["data"]["max_token_limit"] and progress["ready"] != progress["completed"]:
        raise RuntimeError("Changing the batch budget requires a completed epoch boundary")
    metadata = {"resume_checkpoint": str(source), "resume_global_step": checkpoint["global_step"],
                "resume_epoch": progress["completed"], "previous_max_token_limit": old_budget}
    return previous, metadata


def main():
    os.chdir(REPO)
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["training"]["epochs"] == 40 and not cfg["training"].get("early_stopping")
    assert cfg["model"]["compression"]["filter"] == "none"
    assert cfg["tokenizer"]["type"] == "phone69" and cfg["model"]["num_symbols"] == 69
    assert cfg["data"]["train_subset"] == "train-clean-100"
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], text=True).strip()
    if branch != "main":
        raise RuntimeError("This baseline must run on main")
    exp = (REPO / cfg["paths"]["exp_dir"]).resolve()
    exp.mkdir(parents=True, exist_ok=True)
    lock = (exp / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = exp / "run.json"
    previous, resumed = resume_metadata(state_path, cfg)
    if previous:
        history = exp / "run_history"
        history.mkdir(exist_ok=True)
        (history / f"attempt_{previous.get('attempt', 1)}.json").write_text(
            json.dumps(previous, indent=2) + "\n")
    logs = exp / "logs"
    logs.mkdir(exist_ok=True)
    state = {
        "status": "starting", "config": str(CONFIG.relative_to(REPO)),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "vocab_sha256": hashlib.sha256((REPO / cfg["tokenizer"]["vocab_path"]).read_bytes()).hexdigest(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "pipeline_pid": os.getpid(), "epochs": 40, "early_stopping": False,
        "train_subset": "train-clean-100", "val_subset": "dev-clean", "test_subset": "test-clean",
        "tokenizer": "phone69", "compression": "none", "seed": cfg["seed"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "original_started_at": (previous.get("original_started_at") or previous.get("started_at")
                                or datetime.now(timezone.utc).isoformat()),
        "attempt": (previous.get("attempt", 1) if previous else 0) + 1,
        "max_token_limit": cfg["data"]["max_token_limit"],
        **resumed,
    }

    def save(**updates):
        state.update(updates)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(state_path)
        print(json.dumps(updates), flush=True)

    def run_stage(stage, command, filename):
        with (logs / filename).open("a") as log:
            log.write(f"\n--- attempt {state['attempt']} at {state['started_at']} ---\n")
            log.flush()
            proc = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            save(status=stage, child_pid=proc.pid, active_log=str(logs / filename))
            rc = proc.wait()
        if rc:
            raise RuntimeError(f"{stage} exited with code {rc}; see {logs / filename}")

    try:
        save()
        run_stage("training", [sys.executable, "-u", "train.py", str(CONFIG.relative_to(REPO))], "train.log")
        import torch
        ckpt = torch.load(exp / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
        best = None
        for callback in ckpt.get("callbacks", {}).values():
            if isinstance(callback, dict) and callback.get("monitor") == "Losses/val_loss":
                best = callback
                break
        if not best or not Path(best["best_model_path"]).is_file():
            raise RuntimeError("No best validation checkpoint found in last.ckpt")
        checkpoint = best["best_model_path"]
        save(best_checkpoint=checkpoint, best_val_loss=float(best["best_model_score"]))
        del ckpt
        output = exp / "evaluations/test_clean_beam20.csv"
        output.parent.mkdir(exist_ok=True)
        run_stage("evaluating", [sys.executable, "-u", "eval.py", str(CONFIG.relative_to(REPO)),
                                 "--checkpoint", checkpoint, "--metric", "per", "--out", str(output)],
                  "test_clean_beam20.log")
        with output.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 2620:
            raise RuntimeError(f"Expected 2620 test-clean utterances, got {len(rows)}")
        errors = sum(int(row["phone_errors"]) for row in rows)
        phones = sum(int(row["reference_phones"]) for row in rows)
        metrics = {"per": errors / phones, "phone_errors": errors, "reference_phones": phones,
                   "utterances": len(rows), "mean_rtf": sum(float(r["RTF"]) for r in rows) / len(rows),
                   "checkpoint": checkpoint, "beam_width": cfg["decoding"]["beam_width"]}
        (exp / "evaluations/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        save(status="completed", metrics=metrics, finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        save(status="failed", error=str(error))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
