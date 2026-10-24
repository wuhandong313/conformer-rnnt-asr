#!/usr/bin/env python3
"""Inspect by default. Local-data preflight and formal execution require explicit modes."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DEFAULT_CONFIGS = [REPO / f"configs/server_mandarin_syllable_mix_{s}_s42_e50.yaml"
                   for s in ("stride4_none", "stride1_cosd_learned")]


def utc():
    return datetime.now(timezone.utc).isoformat()


def logged(command, path, *, lease=None, gpu_csv=None):
    """A unique log per invocation; sample memory independently from compute load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                   pass_fds=() if lease is None else (lease,))
        try:
            next_sample = 0.
            while process.poll() is None:
                if gpu_csv and time.monotonic() >= next_sample:
                    sample = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu",
                                             "--format=csv,noheader", "-i", "0"], capture_output=True, text=True)
                    with Path(gpu_csv).open("a") as gpu:
                        gpu.write(sample.stdout if sample.returncode == 0 else f"{utc()},sample_failed\n")
                    next_sample = time.monotonic() + 60
                time.sleep(2)
            return process.returncode
        except BaseException:
            # Only interrupt the child started by this invocation; preserve all outputs.
            process.send_signal(__import__("signal").SIGINT)
            process.wait()
            raise


def estimate_storage(batch_count, snapshot_bytes):
    # Full Adam state has two additional per-parameter tensors; initial state is empty.
    full_bytes = math.ceil(snapshot_bytes * 3.3)
    # Every500 step + each possible val200 best + epoch boundary + initial; unique files retained.
    per_model = 1 + 50 * (math.ceil(batch_count / 500) + math.ceil(batch_count / 200) + 1)
    required = 2 * per_model * full_bytes
    return {"train_batches_per_epoch": batch_count, "max_snapshots_per_model": per_model,
            "estimated_full_checkpoint_bytes": full_bytes, "required_checkpoint_bytes": required,
            "minimum_free_bytes": math.ceil(required * 1.2) + 2 * 2**30}


def preflight(paths, audit, *, wait=False):
    import yaml
    from conformer_rnnt.aishell.launch import BUDGETS, SCHEMES, profile, read_configs, source_fingerprints
    from conformer_rnnt.aishell.manifest import create_batch_plan
    from conformer_rnnt.aishell.protocol import file_sha, validate_bundle, write_json
    from conformer_rnnt.aishell.resources import gpu_lease

    originals = read_configs(paths)
    freeze = validate_bundle(originals[0]["paths"]["manifest_root"])
    if not Path(originals[0]["paths"]["dataset_path"]).is_dir():
        raise FileNotFoundError("Local audio missing. There is no download mode")
    audit = Path(audit).resolve()
    audit.mkdir(parents=True, exist_ok=False)
    gate = {"phase": profile(originals[0]) + "_20261024", "valid": False, "server_utc": utc(),
            "source_sha256": source_fingerprints(REPO), "bundle_fingerprint": freeze["fingerprint"],
            "original_configs": {str(Path(p).resolve()): file_sha(p) for p in paths}, "attempts": []}
    write_json(audit / "prelaunch_validation.json", gate)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    with (audit / "regression_tests.log").open("x") as stream:
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                                cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError("Regression tests failed; preserved preflight log")
    gate["tests_sha256"] = file_sha(audit / "regression_tests.log")
    with gpu_lease(wait=wait) as lease:
        for budget in BUDGETS:
            attempt = audit / f"budget_{budget}"
            attempt.mkdir()
            try:
                plan = create_batch_plan(originals[0]["paths"]["manifest_root"], budget)
            except ValueError as error:
                gate["attempts"].append({"budget": budget, "status": "unsafe_single_or_invalid_plan", "error": str(error)})
                break
            plan_path = attempt / "common_batch_plan.json"
            write_json(plan_path, plan)
            resolved, records, reduction_needed = [], [], False
            for original, scheme in zip(originals, SCHEMES):
                config = json.loads(json.dumps(original))
                config["data"]["max_token_limit"] = budget
                config["paths"]["batch_plan_path"] = str(plan_path)
                if config["tokenizer"]["type"] == "mandarin_syllable":
                    from mandarin_syllable_tokenizer import MandarinSyllableTokenizer
                    config["model"]["num_symbols"] = MandarinSyllableTokenizer(config["tokenizer"]["vocab_path"]).get_piece_size()
                config_path = attempt / (scheme + ".yaml")
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                resolved.append(config_path)
                report_path = attempt / (scheme + "_preflight.json")
                probe = "probe_mandarin_syllables.py" if config["tokenizer"]["type"] == "mandarin_syllable" else "probe_aishell1.py"
                code = logged([sys.executable, "-u", REPO / "scripts/server" / probe, config_path,
                               "--out", report_path, "--lease-fd", lease], attempt / (scheme + ".log"), lease=lease)
                report = json.loads(report_path.read_text())
                record = {**report, "config": str(config_path), "report": str(report_path), "report_sha256": file_sha(report_path)}
                records.append(record)
                if code:
                    if report.get("status") == "oom_or_memory_margin":
                        reduction_needed = True
                        break
                    raise RuntimeError(f"Preflight failed for {scheme}; no model/algorithm changes applied")
            gate["attempts"].append({"budget": budget, "models": records})
            write_json(audit / "prelaunch_validation.json", gate, replace=True)
            if reduction_needed:
                continue
            read_configs(resolved)
            if len(records) != 2 or records[0]["data_protocol"] != records[1]["data_protocol"]:
                raise ValueError("Pair does not share the identical frozen data/batch protocol")
            sizes = [Path(r["checkpoint_roundtrip"][0]["path"]).stat().st_size for r in records]
            storage = estimate_storage(len(plan["batches"]["train"]), max(sizes))
            storage["free_bytes"] = shutil.disk_usage(REPO).free
            if storage["free_bytes"] < storage["minimum_free_bytes"]:
                write_json(attempt / "disk_space_failure.json", storage)
                raise OSError("Insufficient disk space for immutable full checkpoint retention")
            if source_fingerprints(REPO) != gate["source_sha256"]:
                raise ValueError("Source changed during preflight; retained checks cannot approve launch")
            gate.update(valid=True, models=records, configs=list(map(str, resolved)), storage=storage,
                        data_protocol=records[0]["data_protocol"], server_utc=utc())
            write_json(audit / "prelaunch_validation.json", gate, replace=True)
            return gate
    raise RuntimeError("No common budget passed safely; retained all evidence, no training started")


def execute(gate_path, *, wait=False, resume=False):
    from conformer_rnnt.aishell.launch import read_configs
    from conformer_rnnt.aishell.protocol import write_json
    from conformer_rnnt.aishell.resources import gpu_lease
    from conformer_rnnt.aishell.runtime import validate_gate

    gate_path = Path(gate_path).resolve()
    gate = json.loads(gate_path.read_text())
    paths = gate["configs"]
    configs = read_configs(paths)
    if not all(c["execution"].get("allow_training") is True for c in configs):
        raise ValueError("Formal training remains disabled in the approved configurations")
    for path in paths:
        validate_gate(gate_path, path, REPO)
    if shutil.disk_usage(REPO).free < gate["storage"]["minimum_free_bytes"]:
        raise OSError("Disk space no longer satisfies the preflight retention estimate")
    audit = gate_path.parent
    state = {"status": "waiting_for_gpu", "pid": os.getpid(), "server_utc": utc(), "gate": str(gate_path)}
    state_path = audit / "queue_state.json"
    write_json(state_path, state, replace=True)
    with gpu_lease(wait=wait) as lease:
        results = []
        for path, config in zip(paths, configs):
            validate_gate(gate_path, path, REPO)
            exp = REPO / config["paths"]["exp_dir"]
            if not (exp / "TRAINING_COMPLETE").exists():
                runner = "train_mandarin_syllables.py" if config["tokenizer"]["type"] == "mandarin_syllable" else "train_aishell1_one.py"
                command = [sys.executable, "-u", REPO / "scripts/server" / runner, path,
                           "--gate", gate_path, "--execute", "--lease-fd", lease]
                if resume and (exp / "run.json").exists():
                    prior = json.loads((exp / "run.json").read_text())
                    command += ["--resume", prior["epoch_boundary_checkpoint"]]
                state.update(status="training", experiment=str(exp), server_utc=utc())
                write_json(state_path, state, replace=True)
                exp.mkdir(exist_ok=True)
                (exp / "logs").mkdir(exist_ok=True)
                # Log directory alone is created by the queue before a fresh runner starts.
                code = logged(command, exp / f"logs/train_{uuid.uuid4().hex[:12]}.log", lease=lease,
                              gpu_csv=exp / "logs/gpu_usage.csv")
                if code:
                    raise RuntimeError(f"Training failed with code {code}; preserved {exp}")
            run = json.loads((exp / "run.json").read_text())
            if run.get("status") != "trained" or run.get("epoch") != 50:
                raise ValueError("Completion marker disagrees with 50-epoch training state")
            destination = audit / "evaluations" / exp.name
            if not (destination / "EVALUATION_COMPLETE").exists():
                state.update(status="evaluating", experiment=str(exp), server_utc=utc())
                write_json(state_path, state, replace=True)
                evaluator = "eval_mandarin_syllables.py" if config["tokenizer"]["type"] == "mandarin_syllable" else "eval_aishell1.py"
                code = logged([sys.executable, "-u", REPO / "scripts/evaluation" / evaluator, path,
                               "--checkpoint", run["best_checkpoint"], "--out", destination],
                              audit / f"eval_{exp.name}_{uuid.uuid4().hex[:12]}.log", lease=lease)
                if code:
                    raise RuntimeError("Evaluation failed; preserved output, no automatic re-run")
            results.append(destination)
        from scripts.evaluation.eval_aishell1 import compare
        compare(results, audit / ("comparison_" + uuid.uuid4().hex[:12] + ".json"))
        state.update(status="complete", server_utc=utc(), results=list(map(str, results)))
        write_json(state_path, state, replace=True)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs=2, default=list(map(str, DEFAULT_CONFIGS)))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--audit-dir", help="New directory, required for a real-data preflight")
    parser.add_argument("--gate", help="Passed prelaunch_validation.json for execution")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Explicitly recover the same pair at full epoch boundaries")
    args = parser.parse_args()
    os.chdir(REPO)
    from conformer_rnnt.aishell.launch import read_configs, readiness
    from conformer_rnnt.aishell.protocol import write_json
    if not args.preflight and not args.execute:
        print(json.dumps(readiness(read_configs(args.configs)), ensure_ascii=False, indent=2))
        return
    if args.preflight and not args.audit_dir or args.execute and not args.gate:
        parser.error("--preflight requires a new --audit-dir; --execute requires --gate")
    try:
        result = (preflight(args.configs, args.audit_dir, wait=args.wait_for_gpu) if args.preflight else
                  execute(args.gate, wait=args.wait_for_gpu, resume=args.resume))
        print(json.dumps({k: v for k, v in result.items() if k in {"status", "valid", "configs"}}, indent=2))
    except BaseException as error:
        parent = Path(args.audit_dir) if args.preflight else Path(args.gate).parent
        if parent.exists():
            write_json(parent / ("failure_" + uuid.uuid4().hex[:12] + ".json"),
                       {"error": repr(error), "traceback": traceback.format_exc(), "server_utc": utc()})
            if args.execute:
                write_json(parent / "queue_state.json", {"status": "failed", "error": repr(error), "server_utc": utc()}, replace=True)
        raise


if __name__ == "__main__":
    main()
