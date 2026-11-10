#!/usr/bin/env python3
"""Validate and distribute external Phone69 labels into LibriSpeech chapters.

The dataset root must contain ``LibriSpeech/<subset>/<speaker>/<chapter>``.
Sources are ``<labels-root>/<subset_with_underscores>/phone69.trans.txt`` or
``phone69``. All selected sources, audio IDs and existing outputs are checked
before any chapter transcript is created. Source annotations are never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence


DEFAULT_SUBSETS = ("train-clean-100", "train-clean-360", "dev-clean", "test-clean")
SUPPORTED_SUBSETS = (*DEFAULT_SUBSETS, "dev-other", "test-other")
DEFAULT_VOCAB = Path(__file__).resolve().parents[2] / "configs" / "phone69_vocab.json"
UTTERANCE_ID = re.compile(r"(?P<speaker>[0-9]+)-(?P<chapter>[0-9]+)-[0-9]{4}")
FIXED_PHONES = frozenset(
    "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
) | frozenset(
    f"{vowel}{stress}"
    for vowel in "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
    for stress in range(3)
)


def load_phones(vocab_path: Path) -> set[str]:
    """Require the fixed stressed ARPABET inventory, with only spn as noise."""
    spec = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"{vocab_path}: expected a Phone69 vocabulary object")
    tokens = spec.get("tokens")
    if (
        spec.get("version") != 1
        or spec.get("type") != "phone69"
        or spec.get("tokenization") != "whitespace"
        or not isinstance(tokens, list)
        or any(not isinstance(token, str) for token in tokens)
        or len(tokens) != 69
        or set(tokens) != FIXED_PHONES
        or spec.get("ignored_tokens") != ["spn"]
    ):
        raise ValueError(f"{vocab_path}: expected the fixed 69 phones and ignored token spn")
    return set(tokens) | {"spn"}


def read_labels(source: Path, allowed_phones: set[str]) -> tuple[dict[str, str], str]:
    """Read whole whitespace-delimited phones, preserving case and stress."""
    raw = source.read_bytes()
    labels = {}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if not fields:
            continue
        utterance_id = fields[0]
        location = f"{source}:{line_number}"
        if UTTERANCE_ID.fullmatch(utterance_id) is None:
            raise ValueError(f"{location}: invalid utterance ID {utterance_id!r}")
        if utterance_id in labels:
            raise ValueError(f"{location}: duplicate utterance ID {utterance_id}")
        if len(fields) != 2 or not fields[1].strip():
            raise ValueError(f"{location}: empty labels for {utterance_id}")
        transcript = fields[1].strip()
        unknown = sorted(set(transcript.split()) - allowed_phones)
        if unknown:
            raise ValueError(f"{location}: unknown Phone69 phone(s): {', '.join(unknown)}")
        labels[utterance_id] = transcript
    if not labels:
        raise ValueError(f"{source}: no utterance labels")
    return labels, hashlib.sha256(raw).hexdigest()


def read_audio_ids(subset_root: Path) -> set[str]:
    """Check that every FLAC uses its utterance's speaker/chapter directory."""
    if not subset_root.is_dir():
        raise ValueError(f"Missing audio subset directory: {subset_root}")
    audio_ids = set()
    for audio in sorted(subset_root.rglob("*.flac")):
        if not audio.is_file():
            continue
        utterance_id = audio.stem
        match = UTTERANCE_ID.fullmatch(utterance_id)
        if match is None:
            raise ValueError(f"{audio}: invalid audio utterance ID {utterance_id!r}")
        expected = (match["speaker"], match["chapter"], audio.name)
        if audio.relative_to(subset_root).parts != expected:
            raise ValueError(f"{audio}: FLAC is outside its speaker/chapter directory")
        if utterance_id in audio_ids:
            raise ValueError(f"{audio}: duplicate audio utterance ID {utterance_id}")
        audio_ids.add(utterance_id)
    if not audio_ids:
        raise ValueError(f"{subset_root}: no FLAC audio found")
    return audio_ids


def check_existing_output(path: Path, content: bytes) -> bool:
    """Return whether an identical output exists; refuse any other target."""
    if path.is_symlink():
        raise ValueError(f"Refusing to overwrite existing symlink: {path}")
    if not path.exists():
        return False
    if not path.is_file() or path.read_bytes() != content:
        raise ValueError(f"Refusing to overwrite different existing output: {path}")
    return True


def atomic_create(path: Path, content: bytes) -> bool:
    """Publish a complete chapter atomically without replacing another file."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A link publishes the complete file atomically and cannot clobber
            # an output created by another process after the preflight check.
            os.link(temporary_path, path)
        except FileExistsError:
            check_existing_output(path, content)
            return False
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def extract_phone69(
    labels_root: str | Path,
    dataset_root: str | Path,
    subsets: Sequence[str] = DEFAULT_SUBSETS,
    vocab_path: str | Path = DEFAULT_VOCAB,
    *,
    dry_run: bool = False,
    source_manifest: str | Path | None = None,
) -> dict:
    """Validate every requested subset, then create missing chapter labels."""
    subsets = tuple(subsets)
    if not subsets or len(subsets) != len(set(subsets)):
        raise ValueError("Select at least one subset, without duplicate subset names")
    unsupported = sorted(set(subsets) - set(SUPPORTED_SUBSETS))
    if unsupported:
        raise ValueError(f"Unsupported Phone69 subset(s): {', '.join(unsupported)}")

    labels_root = Path(labels_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    vocab_path = Path(vocab_path).resolve()
    allowed_phones = load_phones(vocab_path)
    expected_sources = None
    if source_manifest is not None:
        specification = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
        if specification.get("format_version") != 1 or not isinstance(specification.get("subsets"), dict):
            raise ValueError("Invalid source manifest: expected format_version 1 and subsets")
        expected_sources = specification["subsets"]
    outputs_to_create = []
    seen_ids = set()
    manifest = {
        "version": 1,
        "dry_run": dry_run,
        "labels_root": str(labels_root),
        "dataset_root": str(dataset_root),
        "vocab": str(vocab_path),
        "subsets": {},
    }

    for subset in subsets:
        source_root = labels_root / subset.replace("-", "_")
        sources = [
            candidate for name in ("phone69.trans.txt", "phone69")
            if (candidate := source_root / name).is_file()
        ]
        if len(sources) != 1:
            raise ValueError(
                f"{source_root}: expected exactly one source, phone69.trans.txt or phone69; "
                f"found {len(sources)}"
            )
        source = sources[0]
        labels, source_sha256 = read_labels(source, allowed_phones)
        if expected_sources is not None:
            expected = expected_sources.get(subset, {})
            if expected.get("sha256") != source_sha256 or expected.get("utterances") != len(labels):
                raise ValueError(f"{subset}: source hash or utterance count differs from the source manifest")
        duplicates = seen_ids.intersection(labels)
        if duplicates:
            raise ValueError(f"Duplicate utterance ID across subsets: {sorted(duplicates)[0]}")
        seen_ids.update(labels)
        subset_root = dataset_root / "LibriSpeech" / subset
        audio_ids = read_audio_ids(subset_root)
        missing_audio = labels.keys() - audio_ids
        missing_labels = audio_ids - labels.keys()
        if missing_audio or missing_labels:
            raise ValueError(
                f"{subset}: labels without audio={len(missing_audio)} "
                f"(sample: {', '.join(sorted(missing_audio)[:5])}); "
                f"audio without labels={len(missing_labels)} "
                f"(sample: {', '.join(sorted(missing_labels)[:5])})"
            )

        chapters = {}
        for utterance_id in sorted(labels):
            speaker, chapter, _ = utterance_id.split("-")
            output = subset_root / speaker / chapter / f"{speaker}-{chapter}-ph.trans.txt"
            chapters.setdefault(output, []).append(f"{utterance_id} {labels[utterance_id]}\n")
        unchanged = 0
        for output, lines in sorted(chapters.items()):
            content = "".join(lines).encode("utf-8")
            if check_existing_output(output, content):
                unchanged += 1
            else:
                outputs_to_create.append((output, content))
        manifest["subsets"][subset] = {
            "source": str(source),
            "source_sha256": source_sha256,
            "utterances": len(labels),
            "chapter_files": len(chapters),
            "files_to_create": len(chapters) - unchanged,
            "files_unchanged": unchanged,
        }

    manifest["created_files"] = 0
    if not dry_run:
        for output, content in outputs_to_create:
            manifest["created_files"] += int(atomic_create(output, content))
    manifest["utterances"] = len(seen_ids)
    manifest["chapter_files"] = sum(
        details["chapter_files"] for details in manifest["subsets"].values()
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument(
        "--dataset-root", required=True, type=Path,
        help="Parent directory containing LibriSpeech (not the LibriSpeech directory itself)",
    )
    parser.add_argument("--subsets", nargs="+", choices=SUPPORTED_SUBSETS, default=DEFAULT_SUBSETS)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--source-manifest", type=Path, help="Require audited source SHA256 hashes and counts")
    parser.add_argument(
        "--dry-run", "--validate-only", dest="dry_run", action="store_true",
        help="Complete validation and print the JSON manifest without writing chapter files",
    )
    args = parser.parse_args()
    try:
        manifest = extract_phone69(
            args.labels_root, args.dataset_root, args.subsets, args.vocab,
            dry_run=args.dry_run, source_manifest=args.source_manifest,
        )
    except (OSError, ValueError, UnicodeError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
