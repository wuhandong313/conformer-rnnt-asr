"""Run the ten selected 460h Phone69 checkpoints serially on local CPU."""
import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "artifacts/research/phone69_460h_summary_20261107/experiments.json"
IDS = [f"E{i:02}" for i in range(1, 11)]
BASELINE = "E09"
N_UTTS = 2620
N_PHONES = 189023
WARMUP = 10
CPU_THREADS = 4
INTEROP_THREADS = 1
DEFAULT_OUT = ROOT / "artifacts/evaluations/phone69_460h_cpu_serial_20261108"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return Path(path).read_text(encoding="utf-8")


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def verified(path, expected):
    if sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def openmp_runtime():
    library = Path(sys.executable).resolve().parents[1] / "lib/libomp.dylib"
    if sys.platform == "darwin" and library.is_file():
        return {"path": str(library), "sha256": sha(library)}
    return None


def worker_environment(out, runtime):
    env = os.environ.copy()
    if runtime:
        library = Path(runtime["path"])
        verified(library, runtime["sha256"])
        # Torch and conda BLAS must resolve the same OpenMP library. Limit
        # dyld's override directory to this library so Accelerate is untouched.
        directory = out / "runtime_lib"
        directory.mkdir(exist_ok=True)
        link = directory / "libomp.dylib"
        if not link.exists():
            link.symlink_to(library)
        if link.resolve() != library.resolve():
            raise ValueError(f"Unexpected OpenMP runtime: {link}")
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(directory), env.get("DYLD_LIBRARY_PATH"))))
    return env


def code_fingerprints(source):
    """Fingerprint the files in the specified source tree, including its runner."""
    files = sorted((source / "conformer_rnnt").rglob("*.py"))
    files.extend(source / name for name in ("common.py", "phone69_tokenizer.py"))
    # Historical snapshots keep their frozen dependencies for reproducibility.
    files.extend(source.glob("ipa_*.py"))
    snapshot_runner = source / Path(__file__).resolve().relative_to(ROOT)
    if snapshot_runner.is_file():
        files.append(snapshot_runner)
    return {str(file.relative_to(source)): sha(file) for file in files}


def execution_fingerprints(source):
    """The timing job executes this runner with model code from source."""
    result = code_fingerprints(source)
    runner = Path(__file__).resolve()
    result[str(runner.relative_to(ROOT))] = sha(runner)
    return result


def prepare(out):
    import soundfile
    import yaml

    records = {e["id"]: e for e in json.loads(read(LEDGER))["experiments"]}
    entries = []
    dataset = None
    for name in IDS:
        e = records[name]
        ev = e["evaluation"]
        config = ROOT / ev["config_path"]
        verified(config, ev["config_sha256"])
        cfg = yaml.safe_load(read(config))
        ckname = Path(e["best_checkpoint"]["server_path"]).name
        ck = Path(cfg["paths"]["exp_dir"]) / ckname
        if not ck.exists():
            candidates = list((ROOT / "output").glob("local_cpu_*/models/*/" + ckname))
            if len(candidates) != 1:
                raise ValueError(f"{name}: cannot resolve {ckname}")
            ck = candidates[0]
        verified(ck, e["best_checkpoint"]["sha256"])
        verified(cfg["paths"]["global_stats_path"], ev["global_stats_sha256"])
        verified(cfg["tokenizer"]["vocab_path"], sha(ROOT / "configs/phone69_vocab.json"))
        verified(ROOT / ev["csv_path"], ev["csv_sha256"])
        training_recipe = ROOT / e["training_config_path"]
        verified(training_recipe, e["training_config_sha256"])
        subset = Path(cfg["paths"]["dataset_path"]) / "LibriSpeech/test-clean"
        if dataset is not None and subset.resolve() != dataset.resolve():
            raise ValueError(f"{name}: inconsistent dataset root")
        dataset = subset
        if (cfg["data"]["test_subset"], cfg["tokenizer"]["type"], cfg["data"]["num_workers"],
            cfg["decoding"]["beam_width"]) != ("test-clean", "phone69", 0, 20):
            raise ValueError(f"{name}: inconsistent evaluation protocol")
        entries.append(dict(id=name, name=e["name"], label=e["label"], baseline=(name == BASELINE),
                            training_config=str(training_recipe), training_config_sha256=e["training_config_sha256"],
                            config=str(config), config_sha256=ev["config_sha256"],
                            checkpoint=str(ck), checkpoint_sha256=e["best_checkpoint"]["sha256"],
                            old_predictions=ev["csv_path"], old_predictions_sha256=ev["csv_sha256"],
                            global_stats_sha256=ev["global_stats_sha256"], projection=e["input_projection_dim"],
                            backbone=e["backbone_dim"], output_dir=name))
    files = sorted(dataset.glob("*/*/*.flac"), key=lambda f: f.stem)
    if len(files) != N_UTTS or len({f.stem for f in files}) != N_UTTS:
        raise ValueError("Incomplete test-clean audio")
    audio = []
    for f in files:
        meta = soundfile.info(f)
        audio.append(dict(uid=f.stem, samples=meta.frames, sample_rate=meta.samplerate, sha256=sha(f)))
    snapshot = (out / "model_source").is_dir()
    source = out / "model_source" if snapshot else ROOT
    manifest = dict(schema=1, primary_baseline=BASELINE, models=entries,
                    ledger_sha256=sha(LEDGER), audio=audio,
                    chapter_labels={f.name:sha(f) for f in sorted(dataset.glob("*/*/*-ph.trans.txt"))},
                    code_sha256=execution_fingerprints(source),
                    protocol=dict(device="cpu", threads=CPU_THREADS, interop_threads=INTEROP_THREADS, data_workers=0,
                                  openmp_max_active_levels=1,
                                  batch=1, beam=20, dtype="float32", warmup_utterances=WARMUP,
                                  subset="test-clean", expected_phones=N_PHONES,
                                  timing="model(batch) on monotonic wall clock; corpus RTF uses FLAC duration"),
                    runtime=dict(python=sys.executable, python_version=platform.python_version(),
                                 machine=platform.machine(), os=platform.platform(),
                                 torch=package_version("torch"), torchaudio=package_version("torchaudio"),
                                 openmp=openmp_runtime(), model_source="snapshot" if snapshot else "checkout"))
    path = out / "manifest.json"
    if path.exists() and json.loads(read(path)) != manifest:
        raise RuntimeError("Inputs, code, or runtime changed since preparation; inspect the frozen manifest")
    if not path.exists():
        write_json(path, manifest)
    for e in entries:
        dest = out / e["id"]
        dest.mkdir(exist_ok=True)
        for key in ("config", "training_config"):
            path = dest / (key + ".yaml")
            if not path.exists():
                shutil.copyfile(e[key], path)
            verified(path, e[key + "_sha256"])
    return manifest


def package_version(name):
    from importlib.metadata import version
    return version(name)


def worker(out, name, attempt):
    manifest = json.loads(read(out / "manifest.json"))
    source = out / "model_source" if manifest["runtime"]["model_source"] == "snapshot" else ROOT
    if execution_fingerprints(source) != manifest["code_sha256"]:
        raise ValueError(f"{name}: benchmark sources differ from the recorded manifest")
    sys.path.insert(0, str(source))
    import torch
    import yaml
    from threadpoolctl import threadpool_info
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from conformer_rnnt.metrics import phone_error_counts

    module_path = Path(sys.modules[LibriSpeechRNNTModule.__module__].__file__).resolve()
    if not module_path.is_relative_to(source):
        raise ValueError(f"{name}: benchmark model imported from {module_path}, expected {source}")
    protocol = manifest["protocol"]
    torch.set_num_threads(protocol["threads"])
    torch.set_num_interop_threads(protocol["interop_threads"])
    threads = torch.get_num_threads()
    interop = torch.get_num_interop_threads()
    if (threads, interop) != (CPU_THREADS, INTEROP_THREADS):
        raise ValueError(f"{name}: unexpected PyTorch thread budget {(threads, interop)}")
    entry = next(e for e in manifest["models"] if e["id"] == name)
    dest = out / name
    verified(dest / "config.yaml", entry["config_sha256"])
    verified(entry["checkpoint"], entry["checkpoint_sha256"])
    cfg = yaml.safe_load(read(dest / "config.yaml"))
    verified(cfg["paths"]["global_stats_path"], entry["global_stats_sha256"])
    verified(ROOT / entry["old_predictions"], entry["old_predictions_sha256"])
    with (ROOT / entry["old_predictions"]).open(newline="") as old:
        previous = list(csv.DictReader(old))
    model = LibriSpeechRNNTModule.from_config(cfg)
    checkpoint = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.cpu().eval()
    pools = threadpool_info()
    omp_paths = {p["filepath"] for p in pools if p["internal_api"] == "openmp"}
    if len(omp_paths) > 1 or any(p["num_threads"] != threads for p in pools):
        raise ValueError(f"{name}: CPU library budget mismatch: {pools}")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError(f"{name}: unexpected parameter device or precision")
    loader = model.test_dataloader()
    if len(loader) != N_UTTS or loader.batch_size != 1 or loader.num_workers != 0:
        raise ValueError(f"{name}: test loader mismatch")
    dataset = loader.dataset.datasets[0]
    if list(dataset._walker) != [r["uid"] for r in manifest["audio"]] or len(previous) != N_UTTS:
        raise ValueError(f"{name}: unexpected utterance order")
    fields = ["idx","uid","audio_seconds","mel_seconds","wall_seconds","cpu_seconds",
              "RTF","phone_errors","reference_phones","PER","ref","hyp","historical_hyp_changed"]
    audio_s = wall_s = cpu_s = legacy_rtf_sum = 0.0
    errs = refs = changes = 0
    started = now()
    result_csv = attempt / "utterances.csv"
    with torch.no_grad(), result_csv.open("x", newline="") as stream:
        for batch, _ in islice(loader, WARMUP):
            model(batch)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, (batch, transcripts) in enumerate(loader):
            item = manifest["audio"][i]
            duration = item["samples"] / item["sample_rate"]
            mel = int(batch.feature_lengths[0]) * cfg["data"]["hop_length"] / cfg["data"]["sample_rate"]
            ref = " ".join(model.tokenizer.tokenize(transcripts[0]))
            if int(previous[i]["idx"]) != i or previous[i]["ref"] != ref:
                raise ValueError(f"{name}: reference mismatch at {i}")
            c0 = time.process_time_ns()
            t0 = time.perf_counter_ns()
            hyp = model(batch)
            wall = (time.perf_counter_ns() - t0) / 1e9
            cpu = (time.process_time_ns() - c0) / 1e9
            if not 0 < wall < math.inf or not duration > 0 or not mel > 0:
                raise ValueError(f"{name}: invalid timing at {i}")
            errors, phones = phone_error_counts(ref, hyp)
            changed = int(hyp != previous[i]["hyp"])
            writer.writerow(dict(zip(fields, (i,item["uid"],duration,mel,wall,cpu,wall/duration,
                                               errors,phones,errors/phones,ref,hyp,changed))))
            audio_s += duration
            wall_s += wall
            cpu_s += cpu
            legacy_rtf_sum += wall/mel
            errs += errors
            refs += phones
            changes += changed
            if (i+1)%100 == 0 or i+1 == N_UTTS:
                stream.flush()
                write_json(dest / "progress.json", dict(model=name, completed=i+1, total=N_UTTS, updated=now()))
    if i+1 != N_UTTS or refs != N_PHONES:
        raise ValueError(f"{name}: incomplete evaluation")
    result = dict(**entry, manifest_sha256=sha(out/"manifest.json"), started=started, finished=now(),
                  utterances=N_UTTS, reference_phones=refs, phone_errors=errs, corpus_PER=errs/refs,
                  audio_seconds=audio_s, decode_wall_seconds=wall_s, decode_cpu_seconds=cpu_s,
                  corpus_RTF=wall_s/audio_s, cpu_seconds_per_audio_second=cpu_s/audio_s,
                  historical_mean_utterance_RTF=legacy_rtf_sum/N_UTTS, changed_predictions=changes,
                  csv=str(result_csv.relative_to(out)), csv_sha256=sha(result_csv),
                  log=str((attempt/"run.log").relative_to(out)), threads=threads,
                  interop_threads=interop, data_workers=loader.num_workers, threadpools=pools)
    write_json(dest / "result.json", result)


def summarize(out, manifest):
    results = []
    for entry in manifest["models"]:
        path = out / entry["id"] / "result.json"
        if path.exists():
            r = json.loads(read(path))
            if any(r[k] != entry[k] for k in entry) or r["manifest_sha256"] != sha(out/"manifest.json"):
                raise ValueError(f"Result provenance mismatch: {path}")
            verified(out/r["csv"],r["csv_sha256"])
            results.append(r)
    base = next((r["corpus_RTF"] for r in results if r["id"]==BASELINE), None)
    for r in results:
        r["RTF_over_E09"] = r["corpus_RTF"]/base if base else None
    write_json(out/"summary.json", dict(baseline=BASELINE, completed=len(results), total=len(IDS), results=results))
    columns = ["id","name","baseline","projection","backbone","threads","interop_threads","data_workers",
               "corpus_PER","corpus_RTF","RTF_over_E09",
               "utterances","changed_predictions","config","config_sha256","training_config","training_config_sha256",
               "checkpoint","checkpoint_sha256","csv","csv_sha256","log"]
    tmp = out/"summary.csv.tmp"
    with tmp.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    tmp.replace(out/"summary.csv")
    return {r["id"] for r in results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--prepare",action="store_true",help="Verify and freeze inputs without running inference")
    parser.add_argument("--worker",choices=IDS,help=argparse.SUPPRESS)
    parser.add_argument("--attempt",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0,str(ROOT))
    for key in THREAD_ENV:
        os.environ[key] = str(CPU_THREADS)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    out=args.output.resolve()
    if args.worker:
        worker(out,args.worker,args.attempt)
        return
    out.mkdir(parents=True,exist_ok=True)
    with (out/"batch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=prepare(out)
        environment=worker_environment(out, manifest["runtime"]["openmp"])
        if args.prepare:
            print(f"Prepared {len(IDS)} checkpoints for {N_UTTS} utterances; baseline {BASELINE}.")
            return
        done=summarize(out,manifest)
        failures={}
        for entry in manifest["models"]:
            name=entry["id"]
            if name in done:
                continue
            dest=out/name
            number=max((int(d.name.split("_")[-1]) for d in dest.glob("attempt_*") if d.is_dir()),default=0)+1
            attempt=dest/f"attempt_{number:02}"
            attempt.mkdir()
            write_json(out/"progress.json",dict(status="running",model=name,attempt=number,
                                               completed=sorted(done),failures=failures,updated=now()))
            cmd=[sys.executable,"-u",__file__,"--output",str(out),"--worker",name,"--attempt",str(attempt)]
            with (attempt/"run.log").open("x") as log:
                run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=environment,
                                   pass_fds=(lock.fileno(),))
            if run.returncode:
                failures[name]=dict(exit_code=run.returncode,log=str((attempt/"run.log").relative_to(out)))
            done=summarize(out,manifest)
        write_json(out/"progress.json",dict(status="failed" if failures else "complete",
                                            completed=sorted(done),failures=failures,updated=now()))
        if failures or done!=set(IDS):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
