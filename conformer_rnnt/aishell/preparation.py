"""Explicit local-data audit, constrained labels, statistics and bundle freezing."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import wave

from .lexicon import Lexicon
from .protocol import (FRONTEND, G2PM_VERSION, LEXICON_SHA256, OFFICIAL_COUNTS, feature_frames,
                       file_sha, fingerprint, read_jsonl, relative_file, validate_label_protocol, write_json)


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": values[0], "max": values[-1],
            "mean": sum(values) / len(values),
            **{f"p{q}": values[max(0, math.ceil(q / 100 * len(values)) - 1)] for q in (50, 95, 99)}}


def audit_audio(data_root, out, *, expected_counts=None):
    root, out = Path(data_root).resolve(), Path(out)
    expected = OFFICIAL_COUNTS if expected_counts is None else expected_counts
    source = root / "transcript/aishell_transcript_v0.8.txt"
    if not source.is_file() or not (root / "wav").is_dir():
        raise FileNotFoundError("Local official AISHELL audio/transcript is missing; this command never downloads")
    out.mkdir(parents=True, exist_ok=False)
    transcripts, errors, extras, rows = {}, [], [], []
    for number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 2 or parts[0] in transcripts:
            errors.append({"kind": "invalid_or_duplicate_transcript", "line": number})
        else:
            transcripts[parts[0]] = parts[1:]
    seen, pcm_splits, speakers, found = set(), defaultdict(set), defaultdict(set), set()
    for path in sorted((root / "wav").rglob("*.wav")):
        relative = path.relative_to(root)
        uid = path.stem
        if uid in seen:
            errors.append({"kind": "duplicate_audio_id", "utterance_id": uid, "path": str(relative)})
        seen.add(uid)
        parts = path.relative_to(root / "wav").parts
        if len(parts) != 3 or parts[0] not in expected:
            errors.append({"kind": "unexpected_audio_layout", "path": str(relative)})
            continue
        split, speaker = parts[:2]
        try:
            digest, read_count = hashlib.sha256(), 0
            with wave.open(str(path), "rb") as reader:
                if (reader.getframerate() != 16000 or reader.getnchannels() != 1
                        or reader.getsampwidth() != 2 or reader.getcomptype() != "NONE"):
                    raise ValueError("expected mono 16kHz uncompressed PCM16")
                count = reader.getnframes()
                frames = feature_frames(count)
                while True:
                    block = reader.readframes(65536)
                    if not block:
                        break
                    read_count += len(block)
                    digest.update(block)
                if read_count != count * 2:
                    raise ValueError("truncated PCM payload")
            pcm_splits[digest.hexdigest()].add(split)
        except (ValueError, wave.Error, EOFError, OSError) as error:
            errors.append({"kind": "invalid_audio", "path": str(relative), "error": str(error)})
            continue
        if uid not in transcripts:
            extras.append({"utterance_id": uid, "path": str(relative), "reason": "no_official_transcript"})
            continue
        found.add(uid)
        speakers[split].add(speaker)
        rows.append({"utterance_id": uid, "split": split, "speaker_id": speaker,
                     "audio_path": str(relative), "sample_rate": 16000, "num_samples": count,
                     "num_frames": frames, "audio_pcm_sha256": digest.hexdigest(),
                     "words": transcripts[uid], "text": "".join(transcripts[uid])})
    for uid in sorted(set(transcripts) - found):
        errors.append({"kind": "transcript_without_valid_audio", "utterance_id": uid})
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = sorted(speakers[a] & speakers[b])
        if overlap:
            errors.append({"kind": "speaker_leakage", "splits": [a, b], "speakers": overlap})
    for digest, splits in pcm_splits.items():
        if len(splits) > 1:
            errors.append({"kind": "identical_audio_across_splits", "pcm_sha256": digest, "splits": sorted(splits)})
    counts = {s: sum(r["split"] == s for r in rows) for s in expected}
    if counts != expected:
        errors.append({"kind": "official_count_mismatch", "actual": counts, "expected": expected})
    with (out / "audio_index.jsonl").open("x", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda r: (r["split"], r["utterance_id"])):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"version": 1, "valid": not errors, "counts": counts, "frontend": FRONTEND,
              "transcript_sha256": file_sha(source), "audio_index_sha256": file_sha(out / "audio_index.jsonl"),
              "unpaired_audio": extras, "errors": errors,
              "splits": {s: {"duration_seconds": sum(r["num_samples"] / 16000 for r in rows if r["split"] == s),
                              "duration_distribution": distribution([r["num_samples"] / 16000 for r in rows if r["split"] == s]),
                              "character_length_distribution": distribution([len(r["text"]) for r in rows if r["split"] == s]),
                              "speakers": len(speakers[s])} for s in expected}}
    write_json(out / "audio_audit.json", report)
    return report


def prepare_labels(audio_audit_dir, lexicon_path, predictor, out, *, resolutions=None, tokenizer=None):
    audit_root, out = Path(audio_audit_dir), Path(out)
    audit = json.loads((audit_root / "audio_audit.json").read_text())
    if not audit["valid"] or audit["audio_index_sha256"] != file_sha(audit_root / "audio_index.jsonl"):
        raise ValueError("Audio audit must pass, with unchanged audio index, before label preparation")
    lexicon = Lexicon(lexicon_path)
    if lexicon.sha256 != LEXICON_SHA256:
        raise ValueError("Official lexicon bytes differ from the pinned source; review a version change explicitly")
    if tokenizer is None:
        from aishell_lexicon_tokenizer import AISHELLLexiconTokenizer
        tokenizer = AISHELLLexiconTokenizer()
    if lexicon.symbols != list(tokenizer.tokens):
        raise ValueError("Raw dictionary symbols do not match the frozen tokenizer inventory")
    if (predictor.metadata.get("package") != "g2pM" or predictor.metadata.get("version") != G2PM_VERSION
            or not predictor.metadata.get("package_and_model_files")):
        raise ValueError("Predictor requires pinned g2pM/model provenance")
    out.mkdir(parents=True, exist_ok=False)
    decisions = resolutions or {}
    used, issue_counts, summaries, counts, total_issues, unresolved = set(), Counter(), {}, Counter(), 0, 0
    streams = {s: (out / (s + ".jsonl")).open("x", encoding="utf-8") for s in OFFICIAL_COUNTS}
    lengths = defaultdict(list)
    try:
        with (out / "issues.jsonl").open("x", encoding="utf-8") as issue_stream:
            for row in read_jsonl(audit_root / "audio_index.jsonl"):
                predicted = predictor(row["text"])
                result = lexicon.select_sentence(row["utterance_id"], row["words"], predicted,
                                                 split=row["split"], audio_path=row["audio_path"], resolutions=decisions)
                row.update(phones=result["phones"], label_provenance=result["label_provenance"])
                if row["phones"] is not None:
                    tokenizer.tokenize(" ".join(row["phones"]))
                    lengths[row["split"]].append(len(row["phones"]))
                for issue in result["issues"]:
                    issue_stream.write(json.dumps(issue, ensure_ascii=False) + "\n")
                    total_issues += 1
                    unresolved += not issue["resolved"]
                    if issue["resolved"]:
                        used.add(issue["issue_id"])
                    key = issue["word"] + ":" + issue["kind"]
                    issue_counts[key] += 1
                    summaries.setdefault(key, []).extend([issue["issue_id"]] if len(summaries.get(key, [])) < 10 else [])
                streams[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["split"]] += 1
    finally:
        for stream in streams.values():
            stream.close()
    stale = sorted(set(decisions) - used)
    if stale:
        unresolved += len(stale)
    protocol = {"version": 1, "lexicon_sha256": lexicon.sha256, "tokenizer_fingerprint": tokenizer.fingerprint,
                "predictor": predictor.metadata, "selection": "exact_context_match_to_original_dictionary_candidates",
                "comparison_mapping_version": 1, "word_boundaries": "official_transcript_unchanged",
                "resolutions": decisions, "stale_resolution_ids": stale,
                "preprocessing_source_sha256": {name: file_sha(Path(__file__).parent / name)
                                                 for name in ("lexicon.py", "preparation.py", "protocol.py")}}
    write_json(out / "label_protocol.json", protocol)
    write_json(out / "lexicon_audit.json", lexicon.audit)
    report = {"version": 1, "valid": unresolved == 0, "counts": dict(counts), "issues": total_issues,
              "unresolved_issues": unresolved, "groups": [{"key": k, "count": issue_counts[k], "example_issue_ids": summaries[k]}
                                                           for k in sorted(issue_counts)],
              "target_length_distribution": {s: distribution(lengths[s]) for s in OFFICIAL_COUNTS},
              "draft_sha256": {s: file_sha(out / (s + ".jsonl")) for s in OFFICIAL_COUNTS}}
    write_json(out / "label_audit.json", report)
    shutil.copyfile(audit_root / "audio_audit.json", out / "audio_audit.json")
    return report


def compute_statistics(data_root, train_manifest):
    import torch
    import torchaudio
    from common import piecewise_linear_log
    from .manifest import pcm_audio

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80,
        center=True, pad=0, pad_mode="reflect", power=2.0, normalized=False, norm=None, mel_scale="htk")
    count, mean, m2 = 0, torch.zeros(80, dtype=torch.float64), torch.zeros(80, dtype=torch.float64)
    with torch.no_grad():
        for row in read_jsonl(train_manifest):
            if row["split"] != "train":
                raise ValueError("Only official training frames may enter statistics")
            audio, digest = pcm_audio(relative_file(data_root, row["audio_path"]))
            if digest != row["audio_pcm_sha256"] or audio.shape[1] != row["num_samples"]:
                raise ValueError("Audio changed since audit")
            frames = piecewise_linear_log(mel(audio.squeeze(0)).T).double()
            if frames.shape[0] != row["num_frames"] or not torch.isfinite(frames).all():
                raise ValueError("Invalid frontend frames")
            n = frames.shape[0]
            local_mean = frames.mean(0)
            local_m2 = ((frames - local_mean) ** 2).sum(0)
            delta = local_mean - mean
            total = count + n
            m2 += local_m2 + delta.square() * count * n / total
            mean += delta * n / total
            count = total
    variance = m2 / count if count else m2
    if count < 2 or not torch.isfinite(variance).all() or not (variance > 0).all():
        raise ValueError("Training statistics require finite, positive channel variance; no silent epsilon")
    return {"mean": mean.tolist(), "invstddev": variance.rsqrt().tolist(), "num_frames": count,
            "split": "train", "frontend": FRONTEND, "variance_denominator": "population_frame_count"}


def freeze_bundle(data_root, labels_root, destination, *, expected_counts=None):
    labels, out = Path(labels_root), Path(destination)
    audit = json.loads((labels / "label_audit.json").read_text())
    audio = json.loads((labels / "audio_audit.json").read_text())
    expected = OFFICIAL_COUNTS if expected_counts is None else expected_counts
    if not audit["valid"] or audit["unresolved_issues"] or audit["counts"] != expected or not audio["valid"]:
        raise ValueError("Unresolved issues or incomplete official splits prevent freezing")
    protocol = json.loads((labels / "label_protocol.json").read_text())
    validate_label_protocol(protocol)
    if audio["counts"] != expected:
        raise ValueError("Audio audit and label counts differ")
    for split in expected:
        if file_sha(labels / (split + ".jsonl")) != audit["draft_sha256"][split]:
            raise ValueError("Draft labels changed since their audit")
        rows = list(read_jsonl(labels / (split + ".jsonl")))
        if len(rows) != expected[split] or any(not r.get("phones") or r["split"] != split for r in rows):
            raise ValueError("Incomplete labeled split")
    statistics = compute_statistics(data_root, labels / "train.jsonl")
    out.mkdir(parents=True, exist_ok=False)
    for name in [s + ".jsonl" for s in expected] + ["label_protocol.json", "audio_audit.json", "label_audit.json", "lexicon_audit.json"]:
        shutil.copyfile(labels / name, out / name)
    write_json(out / "statistics.json", statistics)
    files = {p.name: file_sha(p) for p in out.iterdir() if p.is_file()}
    protocol = json.loads((out / "label_protocol.json").read_text())
    freeze = {"version": 1, "status": "frozen", "counts": expected, "unresolved_issues": 0,
              "lexicon_sha256": protocol["lexicon_sha256"], "tokenizer_fingerprint": protocol["tokenizer_fingerprint"],
              "frontend": FRONTEND, "files": files}
    freeze["fingerprint"] = fingerprint(freeze)
    write_json(out / "freeze.json", freeze)
    return freeze
