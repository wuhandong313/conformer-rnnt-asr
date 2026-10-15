#!/usr/bin/env python3
"""Prepare consistent LibriSpeech labels as fixed, complete IPA phones.

Default: original words → CMUdict/g2p-en → IPA, ignoring mixed legacy labels.
Alternatively --source arpabet converts existing CMU labels using only stdlib.
Original word and legacy phoneme files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from ipa_tokenizer import DEFAULT_VOCAB, FixedIPATokenizer
from ipa_phonemizer import WordPhonemizer


SUBSETS = (
    "train-clean-100", "train-clean-360", "train-other-500",
    "dev-clean", "dev-other", "test-clean", "test-other",
)


def read_transcripts(path: Path) -> dict[str, str]:
    transcripts = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(f"{path}:{number}: expected utterance ID and nonempty transcript")
        utterance, text = parts
        if utterance in transcripts:
            raise ValueError(f"{path}:{number}: duplicate utterance ID {utterance}")
        transcripts[utterance] = text
    if not transcripts:
        raise ValueError(f"Empty transcript file: {path}")
    return transcripts


def convert_file(path: Path, tokenizer: FixedIPATokenizer, phonemizer=None):
    source = read_transcripts(path)
    converted = {}
    for utterance, text in source.items():
        try:
            if phonemizer is not None:
                text = phonemizer.arpabet(text)
                source[utterance] = text
            ipa = tokenizer.convert_arpabet(text)
            # Require one source phone to produce exactly one vocabulary token.
            if len(tokenizer.encode(ipa)) != len(text.split()):
                raise ValueError("Source/IPA phone counts differ")
            converted[utterance] = ipa
        except ValueError as error:
            raise ValueError(f"{path} [{utterance}]: {error}") from error
    content = "".join(f"{utterance} {text}\n" for utterance, text in converted.items())
    return source, converted, content


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            name = handle.name
            handle.write(content)
        os.chmod(name, 0o644)
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def prepare_dataset(
    dataset_root: str | Path,
    subsets=SUBSETS,
    vocab_path: str | Path = DEFAULT_VOCAB,
    check_only: bool = False,
    overwrite: bool = False,
    source_type: str = "words",
    phonemizer=None,
) -> dict:
    """Audit all selected labels/IDs before writing any new IPA transcript.

    ``check_only`` validates sources and any existing IPA files, without writing.
    Existing outputs must match exactly unless ``overwrite`` is explicitly set.
    ``dataset_root`` is the parent of ``LibriSpeech/``, as in the training config.
    """
    tokenizer = FixedIPATokenizer(vocab_path)
    if source_type not in {"words", "arpabet"}:
        raise ValueError("source_type must be words or arpabet")
    if source_type == "words" and phonemizer is None:
        phonemizer = WordPhonemizer(tokenizer)
    root = Path(dataset_root) / "LibriSpeech"
    subsets = tuple(subsets)
    if not subsets or len(set(subsets)) != len(subsets):
        raise ValueError("Select at least one distinct LibriSpeech subset")
    if any(subset not in SUBSETS for subset in subsets):
        raise ValueError(f"Unsupported subset; choose from {SUBSETS}")
    report = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "vocab_sha256": tokenizer.fingerprint,
        "num_symbols": tokenizer.get_piece_size(),
        "blank_id": tokenizer.blank_id,
        "mode": "check-only" if check_only else "prepare",
        "source_type": source_type,
        "subsets": {},
    }
    source_inventory = Counter()
    ipa_inventory = Counter()
    manifest = []
    source_digest = hashlib.sha256()
    for subset in subsets:
        directory = root / subset
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing LibriSpeech subset: {directory}")
        audio = list(directory.glob("*/*/*.flac"))
        if not audio:
            raise ValueError(f"No audio files found in {directory}")
        # Derive expected chapters from audio, so a missing entire -ph file fails.
        chapters = sorted({path.parent for path in audio})
        stats = {"files": 0, "utterances": 0, "phones": 0,
                 "existing_ipa_files": 0, "written_ipa_files": 0}
        for chapter in chapters:
            stem = f"{chapter.parent.name}-{chapter.name}"
            source_suffix = "-ph" if source_type == "arpabet" else ""
            source_path = chapter / f"{stem}{source_suffix}.trans.txt"
            output_path = chapter / f"{stem}-ipa.trans.txt"
            source, converted, content = convert_file(source_path, tokenizer, phonemizer)
            audio_ids = {path.stem for path in chapter.glob("*.flac")}
            word_ids = set(read_transcripts(chapter / f"{stem}.trans.txt"))
            if set(source) != audio_ids or set(source) != word_ids:
                missing = sorted((audio_ids | word_ids) - set(source))[:5]
                extra = sorted(set(source) - audio_ids)[:5]
                raise ValueError(
                    f"{source_path}: word/audio/ARPABET IDs differ; "
                    f"missing source IDs={missing}, extra source IDs={extra}"
                )
            if output_path.exists():
                stats["existing_ipa_files"] += 1
                if output_path.read_text(encoding="utf-8") != content and not overwrite:
                    raise ValueError(f"Existing IPA file differs: {output_path}; use --overwrite to replace it")
            source_bytes = source_path.read_bytes()
            digest = hashlib.sha256(source_bytes).hexdigest()
            source_digest.update(str(source_path.relative_to(root)).encode("utf-8") + b"\0")
            source_digest.update(source_bytes)
            manifest.append((source_path, output_path, digest, subset))
            for utterance, text in source.items():
                source_inventory.update(text.upper().split())
                ipa_inventory.update(converted[utterance].split())
                stats["phones"] += len(text.split())
            stats["files"] += 1
            stats["utterances"] += len(source)
        report["subsets"][subset] = stats

    # Sources, unknown phones, existing outputs, and complete audio coverage
    # have all been checked before the first data write.
    if not check_only:
        for source_path, output_path, digest, subset in manifest:
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Source changed during preparation: {source_path}")
            _, _, content = convert_file(source_path, tokenizer, phonemizer)
            if not output_path.exists() or output_path.read_text(encoding="utf-8") != content:
                atomic_write(output_path, content)
                report["subsets"][subset]["written_ipa_files"] += 1

    report["source_sha256"] = source_digest.hexdigest()
    report["source_inventory"] = dict(sorted(source_inventory.items()))
    report["ipa_inventory"] = {phone: ipa_inventory[phone] for phone in tokenizer.tokens}
    report["unused_ipa_phones"] = [phone for phone in tokenizer.tokens if not ipa_inventory[phone]]
    report["unknown_phones"] = []
    if phonemizer is not None:
        report["phonemizer"] = phonemizer.audit()
    report["totals"] = {
        key: sum(stats[key] for stats in report["subsets"].values())
        for key in ("files", "utterances", "phones", "existing_ipa_files", "written_ipa_files")
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Dataset root containing LibriSpeech/")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=SUBSETS)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--source", choices=("words", "arpabet"), default="words",
                        help="Use original words consistently (default), or strictly convert CMU labels")
    parser.add_argument("--check-only", action="store_true", help="Audit without creating IPA labels")
    parser.add_argument("--overwrite", action="store_true", help="Replace differing IPA outputs only")
    parser.add_argument("--report", type=Path, help="Save a JSON audit report outside the dataset")
    args = parser.parse_args()
    try:
        report = prepare_dataset(args.dataset_root, args.subsets, args.vocab,
                                 args.check_only, args.overwrite, args.source)
    except (ValueError, OSError, RuntimeError, LookupError) as error:
        parser.exit(1, f"IPA preparation failed: {error}\n")
    if args.report:
        atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
