"""Local AISHELL-3 + THCHS-30 labels, audio audit and shared frozen bundles."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import unicodedata
import wave

from mandarin_syllable_tokenizer import MandarinSyllableTokenizer, SYLLABLE, validate_syllables
from conformer_rnnt.aishell.protocol import (FRONTEND, feature_frames, file_sha, fingerprint,
                                           read_jsonl, relative_file, write_json)

CORPORA = ["aishell3", "thchs30"]
RESAMPLING = {"lowpass_filter_width": 6, "rolloff": .99, "resampling_method": "sinc_interp_hann", "beta": None}
THCHS_COUNTS = {"train": 10000, "dev": 893, "test": 2495}


def punctuation(value):
    return bool(value) and all(unicodedata.category(c).startswith("P") for c in value)


def parse_aishell3(line):
    """Keep released readings, including tones and erhua; never regenerate Pinyin."""
    name, content = line.rstrip("\r\n").split("\t", 1)
    if Path(name).name != name or not name:
        raise ValueError("Unsafe AISHELL-3 recording ID")
    fields, words, syllables, excluded = content.split(), [], [], []
    i = 0
    while i < len(fields):
        word = fields[i]
        if punctuation(word) and (i + 1 == len(fields) or SYLLABLE.fullmatch(fields[i + 1]) is None):
            excluded.append(word)
            i += 1
            continue
        if i + 1 == len(fields) or SYLLABLE.fullmatch(fields[i + 1]) is None:
            raise ValueError(f"Invalid alternating character/Pinyin fields at position {i}")
        words.append(word)
        syllables.append(fields[i + 1])
        i += 2
    validate_syllables(syllables)
    uid = name[:-4] if name.lower().endswith(".wav") else name
    return {"source_id": uid, "filename": name if name.lower().endswith(".wav") else name + ".wav",
            "words": words, "text": "".join(words), "syllables": syllables, "excluded_punctuation": excluded,
            "original_label_line": line.rstrip("\r\n")}


def parse_thchs30(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 2:
        raise ValueError("THCHS-30 requires its original second annotation line")
    syllables = validate_syllables(lines[1].split())
    return {"words": lines[0].split(), "text": "".join(lines[0].split()), "syllables": syllables,
            "original_text_line": lines[0], "original_syllable_line": lines[1],
            "label_source": {"file_sha256": file_sha(path), "line": 2}}


def _audio_row(data_root, path, corpus, source_split, uid, speaker, labels):
    import torch
    import torchaudio
    from conformer_rnnt.aishell.manifest import pcm_audio
    path = Path(path).resolve()
    relative = path.relative_to(Path(data_root).resolve())
    expected_rate = 44100 if corpus == "aishell3" else 16000
    waveform, raw_sha = pcm_audio(path, expected_rate=expected_rate)
    original_samples = waveform.shape[1]
    if expected_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, expected_rate, 16000, **RESAMPLING)
    samples = (original_samples * 16000 + expected_rate - 1) // expected_rate
    if waveform.shape[1] != samples or not torch.isfinite(waveform).all():
        raise ValueError("Invalid canonical audio or resampling length")
    return {**labels, "utterance_id": corpus + "/" + uid, "corpus": corpus,
            "official_split": source_split, "split": source_split, "speaker_id": corpus + "/" + speaker,
            "audio_path": str(relative), "sample_rate": 16000, "num_samples": samples,
            "original_sample_rate": expected_rate, "original_num_samples": original_samples,
            "duration_seconds": original_samples / expected_rate, "num_frames": feature_frames(samples),
            "resampling": RESAMPLING if expected_rate != 16000 else None,
            "audio_pcm_sha256": raw_sha, "canonical_audio_sha256": hashlib.sha256(waveform.numpy().tobytes()).hexdigest()}


def prepare_mix(data_root, aishell3_root, thchs30_root, out, *, seed=42, dev_fraction=.1,
                expected_aishell3_total=88035, expected_thchs_counts=None, inventory=None):
    """No downloader. Invalid/missing/OOV rows block freezing and are reported."""
    a3, th, out = Path(aishell3_root), Path(thchs30_root), Path(out)
    required = [a3 / part / "content.txt" for part in ("train", "test")] + [th / "data"] + [th / p for p in THCHS_COUNTS]
    if any(not p.exists() for p in required):
        raise FileNotFoundError("Both local corpora and original label files are required; nothing is downloaded")
    if not 0 < dev_fraction < 1:
        raise ValueError("A held-out AISHELL-3 development speaker fraction is required")
    out.mkdir(parents=True, exist_ok=False)
    rows, errors, seen, source_hashes, source_counts = [], [], set(), {}, Counter()

    def add(path, corpus, split, uid, speaker, labels):
        key = corpus + "/" + uid
        if key in seen:
            errors.append({"kind": "duplicate_id", "utterance_id": key}); return
        seen.add(key)
        try:
            rows.append(_audio_row(data_root, path, corpus, split, uid, speaker, labels))
        except (ValueError, OSError, EOFError, wave.Error) as error:
            errors.append({"kind": "invalid_audio", "utterance_id": key, "error": str(error)})

    for split in ("train", "test"):
        source = a3 / split / "content.txt"
        source_hashes["aishell3/" + split + "/content.txt"] = file_sha(source)
        source_sha = source_hashes["aishell3/" + split + "/content.txt"]
        for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            source_counts["aishell3/" + split] += 1
            try:
                label = parse_aishell3(line)
                uid, filename = label.pop("source_id"), label.pop("filename")
                label["label_source"] = {"file_sha256": source_sha, "line": line_number, "field": "official_pinyin"}
                add(a3 / split / "wav" / uid[:7] / filename, "aishell3", split, uid, uid[:7], label)
            except ValueError as error:
                errors.append({"kind": "invalid_aishell3_label", "split": split, "line": line_number, "error": str(error)})
    th_expected = THCHS_COUNTS if expected_thchs_counts is None else expected_thchs_counts
    for split in THCHS_COUNTS:
        for audio in sorted((th / split).glob("*.wav")):
            uid = audio.stem
            source_counts["thchs30/" + split] += 1
            annotation = th / "data" / (audio.name + ".trn")
            try:
                label = parse_thchs30(annotation)
                source_hashes["thchs30/" + uid] = file_sha(annotation)
                add(audio, "thchs30", split, uid, uid.split("_")[0], label)
            except (ValueError, OSError) as error:
                errors.append({"kind": "invalid_thchs30_label", "utterance_id": uid, "error": str(error)})
    if sum(v for k, v in source_counts.items() if k.startswith("aishell3/")) != expected_aishell3_total:
        errors.append({"kind": "aishell3_release_count_mismatch", "expected_total": expected_aishell3_total})
    if {p: source_counts["thchs30/" + p] for p in THCHS_COUNTS} != th_expected:
        errors.append({"kind": "thchs30_release_count_mismatch", "expected": th_expected})
    speakers = sorted({r["speaker_id"] for r in rows if r["corpus"] == "aishell3" and r["split"] == "train"})
    dev_speakers = []
    if len(speakers) < 2:
        errors.append({"kind": "insufficient_aishell3_training_speakers"})
    else:
        n = max(1, min(len(speakers) - 1, round(len(speakers) * dev_fraction)))
        dev_speakers = sorted(random.Random(seed).sample(speakers, n))
    for row in rows:
        if row["corpus"] == "aishell3" and row["speaker_id"] in dev_speakers and row["split"] == "train":
            row["split"] = "dev"
    leakage = []
    for corpus in CORPORA:
        parts = {p: {r["speaker_id"] for r in rows if r["corpus"] == corpus and r["split"] == p} for p in THCHS_COUNTS}
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = sorted(parts[a] & parts[b])
            if overlap:
                item = {"corpus": corpus, "splits": [a, b], "speakers": overlap,
                        "allowed_by_official_split": corpus == "thchs30" and [a, b] == ["train", "dev"]}
                leakage.append(item)
                if not item["allowed_by_official_split"]:
                    errors.append({"kind": "speaker_leakage", **item})
    hashes = defaultdict(set)
    for row in rows:
        hashes[row["canonical_audio_sha256"]].add(row["split"])
    for sha, splits in hashes.items():
        if len(splits) > 1:
            errors.append({"kind": "identical_audio_across_splits", "canonical_sha256": sha, "splits": sorted(splits)})
    for corpus, root in (("aishell3", a3), ("thchs30", th)):
        if corpus == "aishell3":
            wavs = {p.resolve() for p in root.glob("*/wav/*/*.wav")}
        else:
            wavs = {p.resolve() for p in (root / "data").glob("*.wav")}
        paired = {(Path(data_root) / r["audio_path"]).resolve() for r in rows if r["corpus"] == corpus}
        if wavs - paired:
            errors.append({"kind": "unpaired_audio", "corpus": corpus, "paths": sorted(str(p) for p in wavs - paired)})
    training_tokens = sorted({t for r in rows if r["split"] == "train" for t in r["syllables"]})
    tokens = json.loads(Path(inventory).read_text())["tokens"] if inventory else training_tokens
    spec = {"version": 1, "type": "mandarin_syllable", "tokenization": "whitespace",
            "normalization": "none_preserve_original", "tokens": tokens,
            "inventory_source": {"type": "explicit_reviewed_inventory", "sha256": file_sha(inventory)} if inventory else
                                {"type": "combined_training_labels_only"}}
    write_json(out / "syllable_vocab.json", spec)
    tokenizer = MandarinSyllableTokenizer(out / "syllable_vocab.json")
    oov = Counter((r["corpus"], r["split"], t) for r in rows for t in r["syllables"] if t not in tokenizer.token_to_id)
    if oov:
        errors.append({"kind": "unseen_syllables", "items": [{"corpus": c, "split": s, "syllable": t, "count": count}
                                                              for (c, s, t), count in sorted(oov.items())]})
    for split in THCHS_COUNTS:
        with (out / (split + ".jsonl")).open("x", encoding="utf-8") as stream:
            for row in sorted((r for r in rows if r["split"] == split), key=lambda r: r["utterance_id"]):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    protocol = {"version": 1, "dataset": "aishell3_thchs30_syllables", "corpora": CORPORA,
                "target_type": "tone_marked_syllables", "g2p_used": False, "dictionary_used": False,
                "source_files_sha256": source_hashes, "tokenizer_fingerprint": tokenizer.fingerprint,
                "expected_release_counts": {"aishell3_total": expected_aishell3_total, "thchs30": th_expected},
                "resampling": RESAMPLING, "frontend": FRONTEND,
                "split_policy": {"seed": seed, "aishell3_dev_fraction_of_training_speakers": dev_fraction,
                                 "aishell3_dev_speakers": dev_speakers, "thchs30": "official_train_dev_test_unchanged"},
                "source_runtime": {"torch": str(__import__("torch").__version__), "torchaudio": __import__("torchaudio").__version__}}
    protocol["preprocessing_source_sha256"] = {
        "preparation": file_sha(__file__),
        "tokenizer": file_sha(Path(__file__).resolve().parents[2] / "mandarin_syllable_tokenizer.py"),
    }
    write_json(out / "label_protocol.json", protocol)
    counts = {p: sum(r["split"] == p for r in rows) for p in THCHS_COUNTS}
    from conformer_rnnt.aishell.preparation import distribution
    report = {"version": 1, "valid": not errors, "counts": counts, "source_counts": dict(source_counts),
              "counts_by_corpus": {c: {p: sum(r["corpus"] == c and r["split"] == p for r in rows) for p in THCHS_COUNTS} for c in CORPORA},
              "errors": errors, "speaker_overlap": leakage, "vocabulary_size": len(tokens),
              "protocol_sha256": file_sha(out / "label_protocol.json"), "vocabulary_sha256": file_sha(out / "syllable_vocab.json"),
              "duration_seconds_by_corpus": {c: {p: sum(r["duration_seconds"] for r in rows if r["corpus"] == c and r["split"] == p)
                                                   for p in THCHS_COUNTS} for c in CORPORA},
              "syllable_length_distribution": {c: distribution([len(r["syllables"]) for r in rows if r["corpus"] == c]) for c in CORPORA},
              "tone_suffix_counts": {c: dict(Counter(t[-1] for r in rows if r["corpus"] == c for t in r["syllables"])) for c in CORPORA},
              "draft_sha256": {p: file_sha(out / (p + ".jsonl")) for p in THCHS_COUNTS}}
    write_json(out / "audio_audit.json", report)
    write_json(out / "label_audit.json", report)
    return report


def freeze_mix(data_root, draft, destination):
    from conformer_rnnt.aishell.preparation import compute_statistics
    draft, out = Path(draft), Path(destination)
    audit = json.loads((draft / "audio_audit.json").read_text())
    if audit.get("valid") is not True or audit.get("errors"):
        raise ValueError("Audio/label/OOV audit must pass before freezing the combined corpora")
    if (json.loads((draft / "label_audit.json").read_text()) != audit
            or file_sha(draft / "label_protocol.json") != audit["protocol_sha256"]
            or file_sha(draft / "syllable_vocab.json") != audit["vocabulary_sha256"]):
        raise ValueError("Audit, label protocol or inventory changed before freezing")
    for split, sha in audit["draft_sha256"].items():
        if file_sha(draft / (split + ".jsonl")) != sha:
            raise ValueError("Draft labels changed since audit")
    stats = compute_statistics(data_root, draft / "train.jsonl")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl", "label_protocol.json", "audio_audit.json", "label_audit.json", "syllable_vocab.json"):
        shutil.copyfile(draft / name, out / name)
    write_json(out / "statistics.json", stats)
    tokenizer = MandarinSyllableTokenizer(out / "syllable_vocab.json")
    freeze = {"version": 2, "status": "frozen", "dataset": "aishell3_thchs30_syllables", "target_type": "tone_marked_syllables",
              "counts": audit["counts"], "corpora": CORPORA, "counts_by_corpus": audit["counts_by_corpus"],
              "tokenizer_fingerprint": tokenizer.fingerprint, "vocab_including_blank": tokenizer.get_piece_size() + 1,
              "frontend": FRONTEND, "unresolved_issues": 0, "files": {p.name: file_sha(p) for p in out.iterdir()}}
    freeze["fingerprint"] = fingerprint(freeze)
    write_json(out / "freeze.json", freeze)
    validate_syllable_bundle(out, freeze)
    return freeze


def validate_syllable_bundle(root, freeze, *, expected_counts=None):
    root = Path(root)
    if (freeze.get("version") != 2 or freeze.get("status") != "frozen" or freeze.get("target_type") != "tone_marked_syllables"
            or freeze.get("corpora") != CORPORA or freeze.get("frontend") != FRONTEND or freeze.get("unresolved_issues") != 0
            or freeze.get("fingerprint") != fingerprint({k: v for k, v in freeze.items() if k != "fingerprint"})):
        raise ValueError("Invalid combined syllable bundle definition")
    required = {"train.jsonl", "dev.jsonl", "test.jsonl", "statistics.json", "label_protocol.json", "audio_audit.json", "label_audit.json", "syllable_vocab.json"}
    if not required.issubset(freeze["files"]):
        raise ValueError("Incomplete combined syllable bundle")
    for name, sha in freeze["files"].items():
        if file_sha(relative_file(root, name)) != sha:
            raise ValueError("Frozen syllable payload changed: " + name)
    tokenizer = MandarinSyllableTokenizer(root / "syllable_vocab.json")
    protocol = json.loads((root / "label_protocol.json").read_text())
    audit = json.loads((root / "audio_audit.json").read_text())
    labels = json.loads((root / "label_audit.json").read_text())
    if (tokenizer.fingerprint != freeze["tokenizer_fingerprint"] or tokenizer.get_piece_size() + 1 != freeze["vocab_including_blank"]
            or protocol.get("target_type") != "tone_marked_syllables" or protocol.get("g2p_used") is not False
            or protocol.get("dictionary_used") is not False or protocol.get("corpora") != CORPORA
            or protocol.get("tokenizer_fingerprint") != tokenizer.fingerprint
            or not audit.get("valid") or audit.get("errors") or not labels.get("valid") or labels.get("errors")
            or audit["counts"] != freeze["counts"] or labels["counts"] != freeze["counts"]):
        raise ValueError("Combined syllable data/tokenizer/audits disagree")
    if expected_counts is not None and freeze["counts"] != expected_counts:
        raise ValueError("Unexpected combined corpus counts")
    for split in THCHS_COUNTS:
        rows = list(read_jsonl(root / (split + ".jsonl")))
        if (len(rows) != freeze["counts"][split] or not rows or any(r["corpus"] not in CORPORA for r in rows)
                or any(sum(r["corpus"] == corpus for r in rows) != freeze["counts_by_corpus"][corpus][split] for corpus in CORPORA)):
            raise ValueError("Incomplete combined split")
        for row in rows:
            if row["split"] != split or tokenizer.tokenize(" ".join(row["syllables"])) != row["syllables"]:
                raise ValueError("Invalid frozen syllable row")
    return freeze
