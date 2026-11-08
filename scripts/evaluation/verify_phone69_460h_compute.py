"""Verify the ten full-corpus count files and write their publication table."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from run_phone69_460h_cpu_batch import ROOT, sha

ARCHIVE = ROOT / "papers/phone69_460h_cpu_authoritative_20261108/supporting/original_archive"
STAGE_LENGTHS = ROOT / "output/phone69_460h/mel_layer5_compute_20261107/frame_lengths.csv"
PAIRED_ENCODER = ROOT / "artifacts/research/phone69_460h_summary_20261107/mel_layer5_compute.json"
CATEGORIES = ("encoder_weights", "attention_products", "projections", "fusion_mlp",
              "detector_dots", "fusion_weighted_sum", "predictor", "joiner")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_components(model, raw, row):
    """Calculate from the architecture, independently of the counting hooks."""
    stride = model["time_reduction_stride"]
    projection = model["time_reduction_input_dim"]
    width = projection * stride
    frames = row["layer_frames"]
    require(len(frames) == 10 and all(type(t) is int and t > 0 for t in frames), "layer lengths")
    comp = model["compression"]
    stages = comp.get("stages", [])
    if "detector" in comp:
        stages = [dict(after_layer=model["num_layers_block1"], **comp)]
    sites = {s["after_layer"]: s for s in stages}
    current = raw
    fusion = Counter()
    for site in range(11):
        if site in sites:
            # The corpus is batch-one, with no extra time padding at fusion.
            # All Mel-stage configurations in this archive use stride one.
            if site == 0:
                require(stride == 1, "Mel stride")
            k = frames[site] if site < 10 else row["output_steps"]
            require(0 < k <= current, "compression length")
            dim = model["input_dim"] if site == 0 else width
            if current > 1:
                fusion["detector_dots"] += (current - 1) * dim
                if sites[site]["mixer"]["type"] == "learned":
                    window = sites[site].get("window_size", 5)
                    fusion["fusion_mlp"] += k * (window * dim * 32 + 32 * window)
                    fusion["fusion_weighted_sum"] += k * window * dim
            current = k
        if site == 0:
            projected_steps = current
            current = (current + stride - 1) // stride
        if site < 10:
            require(frames[site] == current, f"stage {site + 1} input length")
    require(current == row["output_steps"], "encoder output length")
    per_frame = 4 * width * model["ffn_dim"] + 7 * width**2 + width * model["depthwise_conv_kernel_size"]
    # All archived predictors have two 128-wide LSTMs, a 128-wide embedding,
    # and a 256-wide output. The joiner projects to 69 labels and one blank.
    return dict(encoder_weights=per_frame * sum(frames),
                attention_products=2 * width * sum(t*t for t in frames),
                projections=model["input_dim"] * projection * projected_steps + width * 256 * current,
                fusion_mlp=fusion["fusion_mlp"], detector_dots=fusion["detector_dots"],
                fusion_weighted_sum=fusion["fusion_weighted_sum"],
                predictor=row["predictor_tokens"] * 294912, joiner=row["joiner_rows"] * 17920)


def verify_model(result_dir, entry, manifest, independent_lengths, paired_encoder):
    name = entry["id"]
    file = result_dir / f"{name}.json"
    result = json.loads(file.read_text())
    provenance = result["provenance"]
    audio = manifest["audio"]
    n = len(audio)
    require(result["id"] == name and result["checkpoint_sha256"] == entry["checkpoint_sha256"]
            and result["config_sha256"] == entry["config_sha256"], f"{name}: model identity")
    require(n == len(result["rows"]) == result["predictions_checked"] == result["utterances"] == 2620,
            f"{name}: corpus size")
    index_sha = hashlib.sha256(json.dumps(list(range(n))).encode()).hexdigest()
    require(provenance["sample_per_quartile"] is None and provenance["index_sha256"] == index_sha
            and provenance["manifest_sha256"] == sha(ARCHIVE / "manifest.json")
            and provenance["counter_sha256"] == sha(Path(__file__).with_name("measure_phone69_460h_compute.py")),
            f"{name}: provenance")
    cfg = yaml.safe_load((ARCHIVE / name / "config.yaml").read_text())["model"]
    original = json.loads((ARCHIVE / name / "result.json").read_text())
    with (ARCHIVE / original["csv"]).open(newline="") as f:
        references = list(csv.DictReader(f))
    totals = Counter()
    for i, (row, ref, item, length_record) in enumerate(
            zip(result["rows"], references, audio, independent_lengths, strict=True)):
        require(row["idx"] == i == int(ref["idx"]) == int(length_record["idx"])
                and row["uid"] == ref["uid"] == item["uid"], f"{name}: row {i}")
        raw = round(float(ref["mel_seconds"]) * 100)
        require(raw == int(length_record["input_mel_frames"]), f"{name}/{i}: input frames")
        expected = expected_components(cfg, raw, row)
        require(all(type(row.get(k, 0)) is int and row.get(k, 0) == v for k, v in expected.items()),
                f"{name}/{i}: MAC breakdown")
        if name in ("E05", "E10"):
            mel = int(length_record["two_stage_mel_frames"])
            l5 = int(length_record["two_stage_layer5_frames"])
            lengths = [mel] * 10 if name == "E05" else [mel] * 5 + [l5] * 5
            require(row["layer_frames"] == lengths, f"{name}/{i}: independent frame trace")
        totals.update(expected)
    require(dict(totals) == result["components_macs"]
            and sum(totals.values()) == result["total_major_macs"], f"{name}: total MACs")
    if name in ("E05", "E10"):
        key = "mel" if name == "E05" else "two_stage"
        shared_scope = sum(totals[k] for k in CATEGORIES[:4])
        require(entry["checkpoint_sha256"] == paired_encoder["checkpoint_sha256"][key]
                and entry["config_sha256"] == paired_encoder["config_sha256"][key]
                and shared_scope == paired_encoder["totals_macs"][key], f"{name}: independent encoder totals")
    pilot_file = result_dir.parent / "pilot24" / f"{name}.json"
    pilot = json.loads(pilot_file.read_text())
    require(len(pilot["rows"]) == 24 and all(r == result["rows"][r["idx"]] for r in pilot["rows"]),
            f"{name}: pilot overlap")
    duration = math.fsum(item["samples"] / item["sample_rate"] for item in audio)
    denominator = duration * 1e9
    gmac = sum(totals.values()) / denominator
    require(math.isclose(result["audio_seconds"], duration, rel_tol=1e-12)
            and math.isclose(result["gmac_per_audio_second"], gmac, rel_tol=1e-12), f"{name}: duration/GMAC")
    search = totals["predictor"] + totals["joiner"]
    metric = dict(id=name, configuration=entry["name"], utterances=n, audio_seconds=duration,
                  PER_percent=100*original["corpus_PER"], encoder_gmac_per_audio_s=(sum(totals.values())-search)/denominator,
                  search_gmac_per_audio_s=search/denominator, total_gmac_per_audio_s=gmac)
    check = dict(id=name, file_sha256=sha(file), config_sha256=entry["config_sha256"],
                 checkpoint_sha256=entry["checkpoint_sha256"], rows_verified=n, pilot_rows_identical=24,
                 total_major_macs=sum(totals.values()))
    return metric, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Completed 'full' directory next to 'pilot24'")
    out = parser.parse_args().out
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    with STAGE_LENGTHS.open(newline="") as f:
        independent_lengths = list(csv.DictReader(f))
    paired_encoder = json.loads(PAIRED_ENCODER.read_text())["record"]
    metrics, checks = zip(*(verify_model(out, entry, manifest, independent_lengths, paired_encoder)
                            for entry in manifest["models"]))
    baseline = next(r["total_gmac_per_audio_s"] for r in metrics if r["id"] == "E09")
    for m in metrics:
        m["GMAC_over_E09"] = m["total_gmac_per_audio_s"] / baseline
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)
    report = dict(status="passed", scope="major model MACs per second of input audio",
                  manifest_sha256=sha(ARCHIVE / "manifest.json"), independent_lengths_sha256=sha(STAGE_LENGTHS),
                  summary_sha256=sha(out / "summary.csv"), checked_model_utterances=sum(x["rows_verified"] for x in checks),
                  models=checks)
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(checks)} models and {report['checked_model_utterances']} model/utterance traces.")


if __name__ == "__main__":
    main()
