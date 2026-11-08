#!/usr/bin/env python3
"""Evaluate the selected triple-compression checkpoint with the frozen CPU protocol."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
from itertools import islice
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


OUT = Path(__file__).resolve().parent
THREAD_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
N_UTTS = 2620
N_PHONES = 189023


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path, expected):
    if sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run():
    manifest_path = OUT / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest["schema"] != 1:
        raise ValueError("Unsupported manifest schema")
    entry = manifest["model"]
    protocol = manifest["protocol"]
    required_protocol = dict(device="cpu", threads=4, interop_threads=1,
                             data_workers=0, openmp_max_active_levels=1,
                             batch=1, beam=20, dtype="float32", warmup_utterances=10,
                             subset="test-clean", expected_phones=N_PHONES)
    if any(protocol.get(key) != value for key, value in required_protocol.items()):
        raise ValueError("Manifest differs from the established CPU evaluation protocol")
    if entry["id"] != "mel_layer5_layer8" or entry["expected_parameters"] != 4269973:
        raise ValueError("Unexpected evaluation model")
    if entry["training_commit"] != "eedae42d4c22fd11b643cc82e88adec31085cd86":
        raise ValueError("Unexpected training commit")
    if protocol.get("expected_utterances") != N_UTTS:
        raise ValueError("Unexpected test-clean utterance count")
    if any(os.environ.get(key) != "4" for key in THREAD_ENV):
        raise ValueError("CPU library thread environment must be set to four before launch")
    if os.environ.get("OMP_MAX_ACTIVE_LEVELS") != "1":
        raise ValueError("OMP_MAX_ACTIVE_LEVELS must be one before launch")
    if (OUT / "result.json").exists() or (OUT / "utterances.csv").exists():
        raise FileExistsError("Evaluation artifacts already exist; preserve this attempt")

    source = Path(manifest["source"]).resolve()
    for relative, expected in manifest["code_sha256"].items():
        file = (source / relative).resolve()
        if not file.is_relative_to(source):
            raise ValueError(f"Source file outside frozen source directory: {relative}")
        verified(file, expected)
    for key in ("config", "checkpoint", "training_config"):
        verified(entry[key], entry[key + "_sha256"])
    verified(manifest["reference_csv"], manifest["reference_csv_sha256"])
    runtime = manifest["runtime"]
    if Path(sys.executable).resolve() != Path(runtime["python"]).resolve():
        raise ValueError("Python executable differs from the frozen runtime")
    if platform.python_version() != runtime["python_version"] or platform.machine() != runtime["machine"]:
        raise ValueError("Python version or machine differs from the frozen runtime")
    for package in ("torch", "torchaudio"):
        if version(package) != runtime[package]:
            raise ValueError(f"{package} version differs from the frozen runtime")
    if runtime.get("openmp"):
        verified(runtime["openmp"]["path"], runtime["openmp"]["sha256"])

    sys.path.insert(0, str(source))
    import soundfile
    import torch
    import yaml
    from threadpoolctl import threadpool_info
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from conformer_rnnt.metrics import phone_error_counts

    module_path = Path(sys.modules[LibriSpeechRNNTModule.__module__].__file__).resolve()
    if not module_path.is_relative_to(source):
        raise ValueError(f"Model imported outside frozen source: {module_path}")
    torch.set_num_threads(protocol["threads"])
    torch.set_num_interop_threads(protocol["interop_threads"])
    if (torch.get_num_threads(), torch.get_num_interop_threads()) != (4, 1):
        raise ValueError("Unexpected PyTorch thread budget")
    cfg = yaml.safe_load(Path(entry["config"]).read_text(encoding="utf-8"))
    training_cfg = yaml.safe_load(Path(entry["training_config"]).read_text(encoding="utf-8"))
    if cfg["model"] != training_cfg["model"]:
        raise ValueError("Evaluation model configuration differs from the training recipe")
    if (cfg["data"]["test_subset"], cfg["tokenizer"]["type"], cfg["data"]["num_workers"],
            cfg["decoding"]["beam_width"]) != ("test-clean", "phone69", 0, 20):
        raise ValueError("Evaluation configuration differs from the established protocol")
    if cfg["model"]["compression"].get("execution_backend", "reference") != "reference":
        raise ValueError("This checkpoint must be evaluated with the reference backend")
    inputs = dict(global_stats_sha256=sha(cfg["paths"]["global_stats_path"]),
                  vocabulary_sha256=sha(cfg["tokenizer"]["vocab_path"]))
    verified(cfg["paths"]["global_stats_path"], manifest["global_stats_sha256"])
    verified(cfg["tokenizer"]["vocab_path"], manifest["vocab_sha256"])
    subset = Path(cfg["paths"]["dataset_path"]) / "LibriSpeech/test-clean"
    files = sorted(subset.glob("*/*/*.flac"), key=lambda file: file.stem)
    audio = manifest["audio"]
    if len(audio) != N_UTTS or [file.stem for file in files] != [item["uid"] for item in audio]:
        raise ValueError("Test-clean audio inventory or order differs from the frozen manifest")
    for file, item in zip(files, audio):
        verified(file, item["sha256"])
        metadata = soundfile.info(file)
        if (metadata.frames, metadata.samplerate) != (item["samples"], item["sample_rate"]):
            raise ValueError(f"Audio duration differs from the manifest: {item['uid']}")
    expected_audio = sum(item["samples"] / item["sample_rate"] for item in audio)
    if not math.isclose(expected_audio, protocol["expected_audio_seconds"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("Unexpected total test-clean audio duration")
    labels = {file.name: file for file in subset.glob("*/*/*-ph.trans.txt")}
    if set(labels) != set(manifest["chapter_labels"]):
        raise ValueError("Chapter label inventory differs from the frozen manifest")
    for name, expected in manifest["chapter_labels"].items():
        verified(labels[name], expected)
    with Path(manifest["reference_csv"]).open(newline="", encoding="utf-8") as stream:
        previous = list(csv.DictReader(stream))
    if len(previous) != N_UTTS or [row["uid"] for row in previous] != [item["uid"] for item in audio]:
        raise ValueError("Reference CSV utterance order differs from the manifest")
    if sum(int(row["reference_phones"]) for row in previous) != N_PHONES:
        raise ValueError("Unexpected reference phone count in comparison CSV")

    model = LibriSpeechRNNTModule.from_config(cfg)
    checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    checkpoint_step, checkpoint_epoch = checkpoint.get("global_step"), checkpoint.get("epoch")
    del checkpoint
    model.cpu().eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != entry["expected_parameters"]:
        raise ValueError(f"Unexpected model parameter count: {parameters}")
    definition = model.offline_native_definition
    if (definition["input_projection"], definition["stride"], definition["num_layers"], definition["layers"],
            [stage["after_layer"] for stage in definition["compression_stages"]]) != (128, 1, 10, [5, 3, 2], [0, 5, 8]):
        raise ValueError("Unexpected offline triple-compression definition")
    if model.beam_width != 20 or model.tokenizer_type != "phone69":
        raise ValueError("Unexpected decoder or tokenizer")
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise ValueError("Unexpected model parameter device or precision")
    pools = threadpool_info()
    omp_paths = {pool["filepath"] for pool in pools if pool["internal_api"] == "openmp"}
    if len(omp_paths) > 1 or any(pool["num_threads"] != 4 for pool in pools):
        raise ValueError(f"CPU library thread budget mismatch: {pools}")
    if runtime.get("openmp") and omp_paths != {runtime["openmp"]["path"]}:
        raise ValueError(f"OpenMP library differs from the frozen runtime: {omp_paths}")
    loader = model.test_dataloader()
    if len(loader) != N_UTTS or loader.batch_size != 1 or loader.num_workers != 0:
        raise ValueError("Test loader differs from the established protocol")
    if len(loader.dataset.datasets) != 1 or list(loader.dataset.datasets[0]._walker) != [item["uid"] for item in audio]:
        raise ValueError("Test loader utterance order differs from the frozen manifest")

    started = now()
    write_json(OUT / "progress.json", dict(status="warming_up", completed=0, total=N_UTTS, updated=now()))
    fields = ["idx", "uid", "audio_seconds", "mel_seconds", "wall_seconds", "cpu_seconds",
              "RTF", "phone_errors", "reference_phones", "PER", "ref", "hyp", "comparison_hyp_changed"]
    audio_s = wall_s = cpu_s = legacy_rtf_sum = 0.0
    errors_total = phones_total = changed_total = completed = 0
    result_csv = OUT / "utterances.csv"
    with torch.no_grad():
        for batch, _ in islice(loader, protocol["warmup_utterances"]):
            model(batch)
        write_json(OUT / "progress.json", dict(status="running", completed=0, total=N_UTTS, updated=now()))
        with result_csv.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, (batch, transcripts) in enumerate(loader):
                item = audio[index]
                duration = item["samples"] / item["sample_rate"]
                mel = int(batch.feature_lengths[0]) * cfg["data"]["hop_length"] / cfg["data"]["sample_rate"]
                reference = " ".join(model.tokenizer.tokenize(transcripts[0]))
                if int(previous[index]["idx"]) != index or previous[index]["ref"] != reference:
                    raise ValueError(f"Reference text differs from comparison CSV at {index}")
                c0 = time.process_time_ns()
                t0 = time.perf_counter_ns()
                hypothesis = model(batch)
                wall = (time.perf_counter_ns() - t0) / 1e9
                cpu = (time.process_time_ns() - c0) / 1e9
                if not 0 < wall < math.inf or not duration > 0 or not mel > 0:
                    raise ValueError(f"Invalid decode timing at {index}")
                errors, phones = phone_error_counts(reference, hypothesis)
                if phones != int(previous[index]["reference_phones"]) or phones <= 0:
                    raise ValueError(f"Reference phone count differs at {index}")
                changed = int(hypothesis != previous[index]["hyp"])
                writer.writerow(dict(zip(fields, (index, item["uid"], duration, mel, wall, cpu, wall / duration,
                                                    errors, phones, errors / phones, reference, hypothesis, changed))))
                audio_s += duration
                wall_s += wall
                cpu_s += cpu
                legacy_rtf_sum += wall / mel
                errors_total += errors
                phones_total += phones
                changed_total += changed
                completed = index + 1
                if completed % 100 == 0 or completed == N_UTTS:
                    stream.flush()
                    progress = dict(status="running", model=entry["id"], completed=completed, total=N_UTTS,
                                    phone_errors=errors_total, reference_phones=phones_total,
                                    corpus_PER=errors_total / phones_total, corpus_RTF=wall_s / audio_s, updated=now())
                    write_json(OUT / "progress.json", progress)
                    print(json.dumps(progress, ensure_ascii=False), flush=True)
    if completed != N_UTTS or phones_total != N_PHONES:
        raise ValueError("Incomplete evaluation")
    if not math.isclose(audio_s, protocol["expected_audio_seconds"], rel_tol=0, abs_tol=1e-6):
        raise ValueError("Completed evaluation has an unexpected total audio duration")
    result = dict(**entry, manifest_sha256=sha(manifest_path), runner_sha256=sha(__file__),
                  started=started, finished=now(), utterances=completed, reference_phones=phones_total,
                  phone_errors=errors_total, corpus_PER=errors_total / phones_total, audio_seconds=audio_s,
                  decode_wall_seconds=wall_s, decode_cpu_seconds=cpu_s, corpus_RTF=wall_s / audio_s,
                  cpu_seconds_per_audio_second=cpu_s / audio_s,
                  historical_mean_utterance_RTF=legacy_rtf_sum / completed,
                  comparison_changed_predictions=changed_total, parameters=parameters,
                  checkpoint_global_step=checkpoint_step, checkpoint_epoch=checkpoint_epoch,
                  offline_native_definition=definition, protocol=protocol, runtime=runtime,
                  threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                  data_workers=loader.num_workers, threadpools=pools, inputs=inputs,
                  csv=str(result_csv), csv_sha256=sha(result_csv), reference_csv=manifest["reference_csv"],
                  reference_csv_sha256=manifest["reference_csv_sha256"], source=str(source))
    write_json(OUT / "result.json", result)
    write_json(OUT / "progress.json", dict(status="complete", model=entry["id"], completed=completed,
                                          total=N_UTTS, result="result.json", updated=now()))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run()
    except BaseException as error:
        progress_path = OUT / "progress.json"
        progress = read_json(progress_path) if progress_path.exists() else dict(completed=0, total=N_UTTS)
        progress.update(status="failed", error_type=type(error).__name__, error=str(error), updated=now())
        write_json(progress_path, progress)
        raise
