#!/usr/bin/env python3
"""Frozen AISHELL test PER, S/D/I, tones, shared deciles and observed lengths; no RTF."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def evaluate(config_path, checkpoint_path, out, *, cpu=False, smoke=0):
    import torch
    import yaml
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from conformer_rnnt.aishell.evaluation import decile_assignment, sample_metrics, validate_predictions
    from conformer_rnnt.aishell.protocol import file_sha, write_json
    from conformer_rnnt.aishell.runtime import checkpoint_metadata, validate_gate

    config = yaml.safe_load(Path(config_path).read_text())
    if config["tokenizer"]["type"] != "aishell_lexicon" or config["decoding"] != {"beam_width": 20}:
        raise ValueError("AISHELL evaluation requires frozen original symbols and beam20")
    exp = Path(config["paths"]["exp_dir"]).resolve()
    if not smoke:
        run = json.loads((exp / "run.json").read_text())
        best = json.loads((exp / "best_checkpoint.json").read_text())
        if (not (exp / "TRAINING_COMPLETE").exists() or run.get("status") != "trained"
                or run.get("epoch") != 50 or Path(best["path"]).resolve() != Path(checkpoint_path).resolve()):
            raise ValueError("Full test evaluation requires this model's dev-best checkpoint after epoch50")
        validate_gate(run["gate"], config_path, REPO)
    module = LibriSpeechRNNTModule.from_config(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("preflight_only"):
        raise ValueError("Preflight checkpoints are not evaluation models")
    metadata = checkpoint_metadata(checkpoint, module, require_boundary=False)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cpu" if cpu else "cuda")
    module.to(device).eval()
    loader = module.test_dataloader()
    rows = loader.dataset.rows
    deciles = decile_assignment(rows)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    plan = {"status": "smoke_running" if smoke else "running", "server_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(Path(config_path).resolve()), "config_sha256": file_sha(config_path),
            "checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": file_sha(checkpoint_path),
            "checkpoint_metadata": metadata, "data_protocol": module.manifest_protocol,
            "model_definition": module.offline_native_definition, "beam_width": 20,
            "speech_rate": "reference_dictionary_symbols / real_audio_seconds",
            "tone_denominator": "toned pairs matching in minimum-edit toneless alignment",
            "rtf_reported": False, "expected_samples": len(rows), "sample_limit": smoke or None}
    write_json(out / "run_plan.json", plan)
    observed = []

    def lengths_hook(_module, inputs, outputs):
        observed.append((inputs[1].detach().cpu().tolist(), outputs[1].detach().cpu().tolist()))

    handle = module.model.transcriber.compression.register_forward_hook(lengths_hook)
    records = []
    try:
        with (out / "per_sample.jsonl").open("x", encoding="utf-8") as stream, torch.inference_mode():
            for index, (batch, _) in enumerate(loader):
                if smoke and index >= smoke:
                    break
                row = rows[index]
                observed.clear()
                prediction = module(batch)
                module.tokenizer.tokenize(prediction)
                if len(observed) != 1 or batch.feature_lengths.tolist() != [row["num_frames"]]:
                    raise ValueError("Decoder did not expose exactly one measured compression path")
                record = {"utterance_id": row["utterance_id"], "text": row["text"],
                          "ref": " ".join(row["phones"]), "hyp": prediction,
                          "duration_seconds": row["num_samples"] / 16000,
                          "reference_symbols_per_second": len(row["phones"]) * 16000 / row["num_samples"],
                          "decile": deciles[row["utterance_id"]], "mel_frames": row["num_frames"],
                          "pre_compression_frames": observed[0][0][0], "post_compression_frames": observed[0][1][0],
                          **sample_metrics(row["phones"], prediction.split())}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                write_json(out / "eval_state.json", {"status": "running", "samples": len(records),
                    "utterance_id": row["utterance_id"], "server_utc": datetime.now(timezone.utc).isoformat()}, replace=True)
                if index % 10 == 0:
                    print(f"sample={index + 1}/{len(rows)} id={row['utterance_id']} errors={record['phone_errors']}", flush=True)
        # Full evaluation uses the complete frozen order; smoke cannot emit a completion marker.
        summary = validate_predictions(rows[:len(records)] if smoke else rows, records, decile_rows=rows)
        summary.update(data_protocol=module.manifest_protocol, beam_width=20, rtf_reported=False,
                       status="smoke_only" if smoke else "complete")
        with (out / "per_sample_sdi.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        write_json(out / "summary.json", summary)
        write_json(out / "validation_report.json", {"valid": True, "full_test": not smoke,
            "samples": len(records), "per_sample_realignments": True, "frozen_references": True,
            "same_deciles_from_frozen_test": True, "rtf_reported": False})
        write_json(out / "eval_state.json", {"status": summary["status"], "samples": len(records),
                    "server_utc": datetime.now(timezone.utc).isoformat()}, replace=True)
        if not smoke:
            (out / "EVALUATION_COMPLETE").write_text("Frozen test, symbol edits, tones and deciles validated.\n")
        return summary
    except BaseException as error:
        write_json(out / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
                    "samples": len(records), "server_utc": datetime.now(timezone.utc).isoformat()})
        write_json(out / "eval_state.json", {"status": "failed", "samples": len(records), "error": repr(error),
                    "server_utc": datetime.now(timezone.utc).isoformat()}, replace=True)
        raise
    finally:
        handle.remove()


def compare(paths, out):
    from conformer_rnnt.aishell.evaluation import validate_predictions
    from conformer_rnnt.aishell.protocol import file_sha, read_jsonl, validate_bundle, write_json

    summaries, reference = [], None
    for path in map(Path, paths):
        if not (path / "EVALUATION_COMPLETE").exists():
            raise ValueError("Both full test evaluations must be complete")
        plan = json.loads((path / "run_plan.json").read_text())
        config = __import__("yaml").safe_load(Path(plan["config"]).read_text())
        freeze = validate_bundle(config["paths"]["manifest_root"])
        if (freeze["fingerprint"] != plan["data_protocol"]["bundle"]
                or file_sha(plan["config"]) != plan["config_sha256"]):
            raise ValueError("Evaluation configuration or frozen bundle changed")
        rows = list(read_jsonl(Path(config["paths"]["manifest_root"]) / "test.jsonl"))
        records = list(read_jsonl(path / "per_sample.jsonl"))
        summary = validate_predictions(rows, records)
        shared = {"protocol": plan["data_protocol"], "beam": plan["beam_width"],
                  "references": [(r["utterance_id"], r["ref"], r["decile"]) for r in records]}
        if reference is not None and shared != reference:
            raise ValueError("The pair differs in labels, normalization, common batches or test deciles")
        reference = shared
        summaries.append({"directory": str(path.resolve()), **summary})
    if len(summaries) != 2:
        raise ValueError("Exactly two model evaluations required")
    result = {"valid": True, "rtf_reported": False, "models": summaries}
    write_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", type=int, default=0, help="Incomplete subset only; cannot produce final results")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE_DIR", "DYNAMIC_DIR"))
    args = parser.parse_args()
    os.chdir(REPO)
    if args.compare:
        compare(args.compare, args.out)
    else:
        if not args.config or not args.checkpoint or args.smoke < 0:
            parser.error("config and --checkpoint are required; --smoke must be nonnegative")
        evaluate(args.config, args.checkpoint, args.out, cpu=args.cpu, smoke=args.smoke)


if __name__ == "__main__":
    main()
