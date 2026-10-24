"""Portable hashes and frozen data-bundle validation (standard library only)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

OFFICIAL_COUNTS = {"train": 120098, "dev": 14326, "test": 7176}
LEXICON_SHA256 = "a01567fb7d0255164f7f97998cf1616ea40b30f1b3bccfd958df1a09e32f404c"
G2PM_VERSION = "0.1.2.5"
FRONTEND = {
    "sample_rate": 16000, "n_fft": 400, "win_length": 400, "hop_length": 160,
    "n_mels": 80, "center": True, "pad": 0, "pad_mode": "reflect", "power": 2.0,
    "normalized": False, "norm": None, "mel_scale": "htk",
    "f_min": 0.0, "f_max": None, "effective_f_max": 8000.0,
    "window": "hann_periodic", "transform": "existing_piecewise_linear_log",
}


def fingerprint(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value, *, replace=False):
    """Immutable records by default; state/pointer callers explicitly opt in."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if not replace:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(raw)
        return
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False, prefix=path.name + ".") as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    os.replace(temporary, path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as error:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from error


def relative_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Path must remain relative to data root: {relative}")
    return path


def feature_frames(num_samples: int) -> int:
    if type(num_samples) is not int or num_samples <= FRONTEND["n_fft"] // 2:
        raise ValueError("Audio is too short for the unchanged centered reflect-padded STFT")
    return num_samples // FRONTEND["hop_length"] + 1


def validate_bundle(root, *, expected_counts=None):
    """Verify every payload, official split counts and a successful label audit."""
    root = Path(root)
    freeze = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    if freeze.get("dataset") == "aishell3_thchs30_syllables":
        from conformer_rnnt.mandarin.preparation import validate_syllable_bundle
        return validate_syllable_bundle(root, freeze, expected_counts=expected_counts)
    payload = {k: v for k, v in freeze.items() if k != "fingerprint"}
    if freeze.get("fingerprint") != fingerprint(payload):
        raise ValueError("Frozen bundle metadata fingerprint differs")
    counts = OFFICIAL_COUNTS if expected_counts is None else expected_counts
    if (freeze.get("version") != 1 or freeze.get("status") != "frozen"
            or freeze.get("counts") != counts or freeze.get("unresolved_issues") != 0
            or freeze.get("frontend") != FRONTEND or freeze.get("lexicon_sha256") != LEXICON_SHA256):
        raise ValueError("Bundle is not an audited, complete official AISHELL-1 bundle")
    required = {"train.jsonl", "dev.jsonl", "test.jsonl", "statistics.json", "label_protocol.json", "audio_audit.json", "label_audit.json", "lexicon_audit.json"}
    hashes = freeze.get("files", {})
    if not required.issubset(hashes):
        raise ValueError("Frozen bundle is missing required payloads")
    for name, expected in hashes.items():
        if file_sha(relative_file(root, name)) != expected:
            raise ValueError(f"Frozen payload changed: {name}")
    protocol = json.loads((root / "label_protocol.json").read_text())
    validate_label_protocol(protocol)
    audio = json.loads((root / "audio_audit.json").read_text())
    labels = json.loads((root / "label_audit.json").read_text())
    if (protocol["tokenizer_fingerprint"] != freeze["tokenizer_fingerprint"]
            or audio.get("valid") is not True or audio.get("counts") != counts
            or labels.get("valid") is not True or labels.get("counts") != counts
            or labels.get("unresolved_issues") != 0):
        raise ValueError("Frozen audio/label audits do not approve the same complete splits")
    return freeze


def validate_label_protocol(protocol):
    predictor = protocol.get("predictor", {})
    if (protocol.get("version") != 1 or protocol.get("lexicon_sha256") != LEXICON_SHA256
            or protocol.get("selection") != "exact_context_match_to_original_dictionary_candidates"
            or protocol.get("comparison_mapping_version") != 1
            or protocol.get("word_boundaries") != "official_transcript_unchanged"
            or protocol.get("stale_resolution_ids") != []
            or predictor.get("package") != "g2pM" or predictor.get("version") != G2PM_VERSION
            or not predictor.get("package_and_model_files") or not protocol.get("preprocessing_source_sha256")):
        raise ValueError("Label-selection provenance is incomplete or not the pinned g2pM protocol")
