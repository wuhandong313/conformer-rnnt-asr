"""Count major inference MACs along the archived Phone69 beam-search paths."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

from run_phone69_460h_cpu_batch import N_UTTS, ROOT, THREAD_ENV, code_fingerprints, sha, verified

ARCHIVE = ROOT / "papers/phone69_460h_cpu_authoritative_20261108/supporting/original_archive"
COMPONENTS = ("encoder_weights", "attention_products", "projections", "fusion_mlp",
              "detector_dots", "fusion_weighted_sum", "predictor", "joiner")


def sample_indices(audio, per_quartile):
    if per_quartile is None:
        return list(range(len(audio)))
    # The first ten utterances served as warm-up in the archived timing study.
    ordered = sorted(range(10, len(audio)), key=lambda i: audio[i]["samples"] / audio[i]["sample_rate"])
    if not 1 <= per_quartile <= len(ordered) // 4:
        raise ValueError("Invalid sample size per duration quartile")
    rng = random.Random(42)
    selected = []
    for k in range(4):
        group = ordered[k * len(ordered) // 4:(k + 1) * len(ordered) // 4]
        selected.extend(rng.sample(group, per_quartile))
    return sorted(selected)


def observe(model, counts, lengths):
    """Attach observers to this model instance. Hooks never change a tensor."""
    from torch.nn import Linear
    from torchaudio.models.conformer import ConformerLayer
    from conformer_rnnt.compression.detectors import CosineDiffPeaksDetector
    from conformer_rnnt.compression.mixers import HardSelectMixer, LearnedWeightMixer
    from conformer_rnnt.compression.natural import NaturalPeakCompression

    def conformer_hook(weight_entries):
        def count(module, inputs):
            # The installed torchaudio ConformerLayer receives [time, batch, width].
            steps, batch, width = inputs[0].shape
            if batch != 1 or width != module.self_attn.embed_dim:
                raise ValueError("Unexpected Conformer layer input")
            lengths.append(steps)
            counts["encoder_weights"] += steps * weight_entries
            counts["attention_products"] += 2 * steps * steps * width
        return count

    def linear_hook(category):
        def count(module, inputs, output):
            counts[category] += output.numel() * module.in_features
            if category == "joiner":
                counts["joiner_calls"] += 1
                counts["joiner_rows"] += output.numel() // module.out_features
        return count

    def compression_hook(module, inputs, output):
        if module.passthrough:
            return
        if not isinstance(module.detector, CosineDiffPeaksDetector):
            raise ValueError("The counter expects the archived cosine-extrema detector")
        if module.execution_backend != "reference":
            raise ValueError("This counter covers the archived reference compression backend")
        n = int(inputs[1].item())
        k = int(output[1].item())
        if n <= 1:
            # The reference implementation returns these frames without invoking the mixer.
            return
        d = inputs[0].shape[-1]
        counts["detector_dots"] += max(0, n - 1) * d
        if isinstance(module.mixer, LearnedWeightMixer):
            counts["fusion_weighted_sum"] += k * module.mixer.window_size * d
        elif not isinstance(module.mixer, HardSelectMixer):
            raise ValueError("Unsupported fusion rule")

    encoder = model.model.transcriber
    for layer in encoder.modules():
        if isinstance(layer, ConformerLayer):
            # FFNs, packed Q/K/V and output projections, pointwise and depthwise
            # convolutions. Each matrix/kernel is applied once per time step.
            weights = sum(p.numel() for p in layer.parameters() if p.ndim >= 2)
            layer.register_forward_pre_hook(conformer_hook(weights))
        elif isinstance(layer, NaturalPeakCompression):
            layer.register_forward_hook(compression_hook)
    for module in (encoder.input_linear, encoder.output_linear):
        module.register_forward_hook(linear_hook("projections"))
    for module in encoder.compression.modules():
        if isinstance(module, Linear):
            module.register_forward_hook(linear_hook("fusion_mlp"))

    predictor = model.model.predictor
    predictor_weights = sum(m.weight.numel() for m in predictor.modules() if isinstance(m, Linear))

    def predictor_hook(module, inputs):
        # Every expanded hypothesis executes all LSTM gate and output matrices.
        tokens = inputs[0].numel()
        counts["predictor"] += tokens * predictor_weights
        counts["predictor_calls"] += 1
        counts["predictor_tokens"] += tokens

    predictor.register_forward_pre_hook(predictor_hook)
    model.model.joiner.linear.register_forward_hook(linear_hook("joiner"))

    def final_length(module, inputs, output):
        counts["output_steps"] += int(output[1].item())
    encoder.register_forward_hook(final_length)


def verify_corpus(archive, manifest):
    """Check that the files actually used are the frozen audio and labels."""
    import soundfile
    import yaml

    if len(manifest["audio"]) != N_UTTS or len({item["uid"] for item in manifest["audio"]}) != N_UTTS:
        raise ValueError("Expected 2,620 distinct test-clean recordings")
    roots = set()
    for entry in manifest["models"]:
        config_path = archive / entry["id"] / "config.yaml"
        verified(config_path, entry["config_sha256"])
        cfg = yaml.safe_load(config_path.read_text())
        if cfg["data"]["test_subset"] != "test-clean" or cfg["tokenizer"]["type"] != "phone69":
            raise ValueError(f"{entry['id']}: expected test-clean and Phone69")
        verified(cfg["tokenizer"]["vocab_path"], sha(archive / "inputs/phone69_vocab.json"))
        roots.add(Path(cfg["paths"]["dataset_path"]).resolve() / "LibriSpeech/test-clean")
    if len(roots) != 1:
        raise ValueError("The models do not use the same test corpus")
    root = roots.pop()
    for item in manifest["audio"]:
        speaker, chapter, _ = item["uid"].split("-")
        path = root / speaker / chapter / (item["uid"] + ".flac")
        verified(path, item["sha256"])
        header = soundfile.info(path)
        if (header.frames, header.samplerate) != (item["samples"], item["sample_rate"]):
            raise ValueError(f"Duration mismatch for {item['uid']}")
    for name, digest in manifest["chapter_labels"].items():
        speaker, chapter, _ = name.split("-", 2)
        verified(root / speaker / chapter / name, digest)


def validate_result(result, entry, audio, indices, provenance):
    expected = dict(id=entry["id"], config_sha256=entry["config_sha256"],
                    checkpoint_sha256=entry["checkpoint_sha256"], provenance=provenance,
                    utterances=len(indices), predictions_checked=len(indices))
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{entry['id']}: output provenance or coverage mismatch")
    totals = Counter()
    for i, row in zip(indices, result["rows"], strict=True):
        if row["idx"] != i or row["uid"] != audio[i]["uid"]:
            raise ValueError(f"{entry['id']}: unexpected utterance at index {i}")
        if len(row["layer_frames"]) != 10 or min(row["layer_frames"]) <= 0:
            raise ValueError(f"{entry['id']}: incomplete encoder trace at {i}")
        for key in (*COMPONENTS, "output_steps", "predictor_tokens", "joiner_rows", "joiner_calls"):
            value = row.get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f"{entry['id']}: invalid {key} at {i}")
            totals[key] += value
        if min(row.get(k, 0) for k in ("output_steps", "predictor", "joiner", "predictor_tokens", "joiner_rows")) <= 0:
            raise ValueError(f"{entry['id']}: incomplete decoder trace at {i}")
    duration = math.fsum(audio[i]["samples"] / audio[i]["sample_rate"] for i in indices)
    components = {k: totals[k] for k in COMPONENTS}
    if (result["components_macs"] != components or result["total_major_macs"] != sum(components.values())
            or not math.isclose(result["audio_seconds"], duration, rel_tol=1e-12)
            or not math.isclose(result["gmac_per_audio_second"], sum(components.values()) / 1e9 / duration,
                                rel_tol=1e-12)
            or result["encoder_output_steps"] != totals["output_steps"]
            or any(result[k] != totals[k] for k in ("predictor_tokens", "joiner_rows", "joiner_calls"))):
        raise ValueError(f"{entry['id']}: reported totals disagree with the utterance traces")


def measure(archive, entry, audio, indices):
    import torch
    import yaml
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule

    if not Path(sys.modules[LibriSpeechRNNTModule.__module__].__file__).resolve().is_relative_to(archive / "model_source"):
        raise ValueError("The model was not imported from the frozen source directory")
    folder = archive / entry["id"]
    config_file = folder / "config.yaml"
    verified(config_file, entry["config_sha256"])
    cfg = yaml.safe_load(config_file.read_text())
    verified(cfg["paths"]["global_stats_path"], entry["global_stats_sha256"])
    verified(entry["checkpoint"], entry["checkpoint_sha256"])
    original = json.loads((folder / "result.json").read_text())
    archived_predictions = archive / original["csv"]
    verified(archived_predictions, original["csv_sha256"])
    with archived_predictions.open(newline="") as f:
        references = list(csv.DictReader(f))
    model = LibriSpeechRNNTModule.from_config(cfg)
    state = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    model.on_load_checkpoint(state)
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    model.cpu().eval()
    loader = model.test_dataloader()
    if (model.beam_width, loader.batch_size, loader.num_workers) != (20, 1, 0):
        raise ValueError("Expected the archived offline batch-one, beam-20 protocol")
    if (len(references) != len(audio) or len(loader) != len(audio)
            or loader.dataset.datasets[0]._walker != [x["uid"] for x in audio]):
        raise ValueError("Dataset does not match the frozen utterance order")
    counts, lengths = Counter(), []
    observe(model, counts, lengths)
    records, totals = [], Counter()
    with torch.no_grad():
        for completed, i in enumerate(indices, 1):
            batch, texts = loader.collate_fn([loader.dataset[i]])
            expected = references[i]
            if expected["uid"] != audio[i]["uid"] or " ".join(model.tokenizer.tokenize(texts[0])) != expected["ref"]:
                raise ValueError(f"Reference mismatch at utterance {i}")
            counts.clear()
            lengths.clear()
            if model(batch) != expected["hyp"]:
                raise ValueError(f"{entry['id']} prediction differs from archive at utterance {i}")
            if len(lengths) != 10 or min(counts[k] for k in ("predictor","joiner","output_steps")) <= 0:
                raise ValueError("Incomplete execution trace")
            totals.update(counts)
            records.append(dict(idx=i, uid=expected["uid"], layer_frames=list(lengths), **counts))
            if completed % 250 == 0:
                print(f"{entry['id']}  {completed}/{len(indices)} utterances", flush=True)
    duration = math.fsum(audio[i]["samples"] / audio[i]["sample_rate"] for i in indices)
    macs = sum(totals[k] for k in COMPONENTS)
    return dict(id=entry["id"], config_sha256=entry["config_sha256"], checkpoint_sha256=entry["checkpoint_sha256"],
                utterances=len(records), audio_seconds=duration, total_major_macs=macs,
                gmac_per_audio_second=macs / duration / 1e9,
                components_macs={k:totals[k] for k in COMPONENTS},
                encoder_output_steps=totals["output_steps"], predictor_tokens=totals["predictor_tokens"],
                joiner_rows=totals["joiner_rows"], joiner_calls=totals["joiner_calls"], predictions_checked=len(records),
                rows=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Output directory, one JSON per model")
    parser.add_argument("--models", nargs="+", help="Experiment IDs, default E01 through E10")
    parser.add_argument("--sample-per-quartile", type=int, help="Pilot only; omit to count the entire corpus")
    args = parser.parse_args()
    archive = ARCHIVE
    manifest_path = archive / "manifest.json"
    authority = json.loads((archive / "AUTHORITATIVE.json").read_text())
    verified(manifest_path, authority["original_verification"]["manifest_sha256"])
    manifest = json.loads(manifest_path.read_text())
    source = archive / "model_source"
    if code_fingerprints(source) != manifest["code_sha256"]:
        raise ValueError("Model source does not match frozen manifest")
    sys.path.insert(0, str(source))
    for variable in THREAD_ENV:
        os.environ[variable] = "4"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    import torch
    import torchaudio
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for version, key in [(torch.__version__, "torch"), (torchaudio.__version__, "torchaudio")]:
        if version != manifest["runtime"][key]:
            raise ValueError(f"Use the frozen {key} environment ({manifest['runtime'][key]})")
    entries = {entry["id"]: entry for entry in manifest["models"]}
    if set(entries) != set(authority["scope"]["model_ids"]):
        raise ValueError("The model list differs from the authoritative archive")
    if args.models and not set(args.models).issubset(entries):
        parser.error("Unknown model ID")
    verify_corpus(archive, manifest)
    indices = sample_indices(manifest["audio"], args.sample_per_quartile)
    provenance = dict(manifest_sha256=sha(manifest_path), counter_sha256=sha(Path(__file__)),
                      index_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                      subset="test-clean", sample_per_quartile=args.sample_per_quartile,
                      torch=torch.__version__, torchaudio=torchaudio.__version__)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.models or entries:
        entry = entries[name]
        target = args.out / f"{name}.json"
        if target.exists():
            result = json.loads(target.read_text())
            validate_result(result, entry, manifest["audio"], indices, provenance)
        else:
            result = measure(archive, entry, manifest["audio"], indices)
            result["provenance"] = provenance
            validate_result(result, entry, manifest["audio"], indices, provenance)
            temp = target.with_suffix(f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(result, separators=(",", ":")) + "\n")
            temp.replace(target)
        print(f"{name}  {len(indices)} utterances  {result['gmac_per_audio_second']:.6f} GMAC/audio-s", flush=True)


if __name__ == "__main__":
    main()
