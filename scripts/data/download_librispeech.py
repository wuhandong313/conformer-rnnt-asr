#!/usr/bin/env python3
"""Download verified LibriSpeech archives using only the Python standard library.

The dataset root contains ``LibriSpeech/<subset>``. Downloads are cached in
``.downloads``; interrupted transfers retain a resumable ``.part`` file.
Phone69 annotations are supplied separately by ``extract_phone69.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Sequence


DEFAULT_SUBSETS = ("train-clean-100", "train-clean-360", "dev-clean", "test-clean")
DEFAULT_BASE_URL = "https://www.openslr.org/resources/12/"
# Alternative: https://openslr.magicdatatech.com/resources/12/
# Copied from conformer_rnnt/lightning_module.py; do not import its torch stack.
CHECKSUMS = {
    "dev-clean": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",
    "dev-other": "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",
    "test-clean": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
    "test-other": "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",
    "train-clean-100": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
    "train-clean-360": "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",
    "train-other-500": "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2",
}
AUDIO_COUNTS = {
    "train-clean-100": 28539,
    "train-clean-360": 104014,
    "train-other-500": 148688,
    "dev-clean": 2703,
    "dev-other": 2864,
    "test-clean": 2620,
    "test-other": 2939,
}
SUPPORTED_SUBSETS = tuple(CHECKSUMS)
MARKER_NAME = ".download_complete.json"
CHUNK_SIZE = 1024 * 1024
UTTERANCE_ID = re.compile(r"(?P<speaker>[0-9]+)-(?P<chapter>[0-9]+)-[0-9]{4}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, archive: Path, expected_hash: str) -> Path:
    """Resume a transfer, then publish only an archive with the expected hash."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".part")
    for path in (archive, partial):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Refusing download target that is not a regular file: {path}")
    if archive.exists():
        if sha256(archive) == expected_hash:
            return archive
        raise ValueError(f"SHA256 mismatch in cached archive {archive}; remove it and retry")

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "LibriSpeech-verified-downloader/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(f"Downloading {url}" + (f" (resume at {offset} bytes)" if offset else ""), file=sys.stderr)
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        # A complete .part can remain after interruption immediately before its rename.
        if error.code == 416 and offset and sha256(partial) == expected_hash:
            partial.replace(archive)
            return archive
        raise
    with response:
        status = response.status
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)", content_range)
            if match is None or int(match[1]) != offset or int(match[2]) < offset:
                raise ValueError(f"Invalid resumed Content-Range from {url}: {content_range!r}")
            mode = "ab"
        elif status == 200:
            # Some mirrors ignore Range. Never append a whole response to a partial.
            mode = "wb"
        else:
            raise ValueError(f"Unexpected HTTP status {status} from {url}")
        first_chunk = response.read(CHUNK_SIZE)
        if (
            "text/html" in response.headers.get("Content-Type", "").lower()
            or (status == 200 and not first_chunk.startswith(b"\x1f\x8b"))
        ):
            raise ValueError(
                f"Rejected non-gzip response from {response.url}; check campus/network login. "
                "Any existing partial download was preserved."
            )
        received = 0
        next_progress = time.monotonic() + 30
        with partial.open(mode) as stream:
            chunk = first_chunk
            while chunk:
                stream.write(chunk)
                received += len(chunk)
                if time.monotonic() >= next_progress:
                    completed = received + (offset if mode == "ab" else 0)
                    print(f"Downloaded {completed} bytes: {archive.name}", file=sys.stderr, flush=True)
                    next_progress = time.monotonic() + 30
                chunk = response.read(CHUNK_SIZE)
        content_length = response.headers.get("Content-Length")
        if content_length is not None and received != int(content_length):
            raise ValueError(f"Interrupted download from {url}; retained {partial} for resume")
    actual_hash = sha256(partial)
    if actual_hash != expected_hash:
        partial.unlink()
        raise ValueError(
            f"SHA256 mismatch for {url}: expected {expected_hash}, got {actual_hash}. "
            "Rejected the download; check campus/network login or select another --base-url."
        )
    partial.replace(archive)
    return archive


def validate_subset(subset_root: Path, subset: str) -> int:
    """Require the official FLAC count and the standard speaker/chapter layout."""
    if subset_root.is_symlink() or not subset_root.is_dir():
        raise ValueError(f"Missing or unsafe subset directory: {subset_root}")
    count = 0
    for path in subset_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink in subset: {path}")
        if path.suffix != ".flac":
            continue
        match = UTTERANCE_ID.fullmatch(path.stem)
        if (
            not path.is_file() or path.stat().st_size == 0 or match is None
            or path.relative_to(subset_root).parts != (match["speaker"], match["chapter"], path.name)
        ):
            raise ValueError(f"Invalid FLAC file or speaker/chapter layout: {path}")
        count += 1
    expected = AUDIO_COUNTS[subset]
    if count != expected:
        raise ValueError(
            f"Incomplete subset {subset_root}: found {count} FLAC files, expected {expected}. "
            "Move or remove the incomplete subset before retrying; existing data was preserved."
        )
    return count


def write_marker(subset_root: Path, details: dict) -> None:
    marker = subset_root / MARKER_NAME
    if marker.is_symlink():
        raise ValueError(f"Refusing marker symlink: {marker}")
    descriptor, name = tempfile.mkstemp(prefix=".download-marker-", dir=subset_root)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(details, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)


def extract_archive(archive: Path, staging: Path, subset: str) -> Path:
    """Extract regular files/directories inside staging, rejecting links and escapes."""
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute() or ".." in path.parts or "\\" in member.name
                or not path.parts or path.parts[0] != "LibriSpeech"
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError(f"Unsafe archive member: {member.name!r}")
            destination = staging.joinpath(*path.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as input_stream, destination.open("xb") as output:
                    shutil.copyfileobj(input_stream, output, CHUNK_SIZE)
    subset_root = staging / "LibriSpeech" / subset
    validate_subset(subset_root, subset)
    return subset_root


def download_librispeech(
    dataset_root: str | Path,
    subsets: Sequence[str] = DEFAULT_SUBSETS,
    base_url: str = DEFAULT_BASE_URL,
) -> dict:
    """Download missing subsets and validate existing subsets before marking success."""
    subsets = tuple(subsets)
    if not subsets or len(set(subsets)) != len(subsets):
        raise ValueError("Select at least one subset, without duplicate subset names")
    if set(subsets) - set(SUPPORTED_SUBSETS):
        raise ValueError(f"Unsupported LibriSpeech subset(s): {sorted(set(subsets) - set(SUPPORTED_SUBSETS))}")
    if urllib.parse.urlsplit(base_url).scheme not in ("http", "https"):
        raise ValueError("--base-url must use http:// or https://")
    dataset_root = Path(dataset_root).expanduser().resolve()
    librispeech = dataset_root / "LibriSpeech"
    if librispeech.is_symlink():
        raise ValueError(f"Refusing LibriSpeech directory symlink: {librispeech}")
    librispeech.mkdir(parents=True, exist_ok=True)
    result = {"version": 1, "dataset_root": str(dataset_root), "subsets": {}}
    for subset in subsets:
        target = librispeech / subset
        url = base_url.rstrip("/") + f"/{subset}.tar.gz"
        details = {
            "version": 1, "subset": subset, "url": url,
            "expected_archive_sha256": CHECKSUMS[subset],
            "audio_files": AUDIO_COUNTS[subset],
            "validation": "official_audio_count_and_layout",
            "archive_sha256_verified": False,
        }
        if target.exists() or target.is_symlink():
            validate_subset(target, subset)
            # Preserve provenance when a previous verified extraction was marked.
            marker = target / MARKER_NAME
            if marker.is_file() and not marker.is_symlink():
                try:
                    previous = json.loads(marker.read_text(encoding="utf-8"))
                    if (
                        isinstance(previous, dict) and previous.get("subset") == subset
                        and previous.get("expected_archive_sha256") == CHECKSUMS[subset]
                    ):
                        details["archive_sha256_verified"] = previous.get("archive_sha256_verified") is True
                except (ValueError, OSError):
                    pass
            write_marker(target, details)
            print(f"Validated existing {subset}: {AUDIO_COUNTS[subset]} audio files", file=sys.stderr)
        else:
            archive = download_archive(url, dataset_root / ".downloads" / f"{subset}.tar.gz", CHECKSUMS[subset])
            with tempfile.TemporaryDirectory(prefix=f".{subset}-extract-", dir=dataset_root) as directory:
                extracted = extract_archive(archive, Path(directory), subset)
                # Preserve the official license, speaker list and README files.
                for metadata in (Path(directory) / "LibriSpeech").iterdir():
                    if not metadata.is_file():
                        continue
                    destination = librispeech / metadata.name
                    if destination.is_symlink():
                        raise ValueError(f"Refusing metadata symlink: {destination}")
                    if destination.exists():
                        if not destination.is_file() or destination.read_bytes() != metadata.read_bytes():
                            raise ValueError(f"Conflicting existing corpus metadata: {destination}")
                    else:
                        os.link(metadata, destination)
                details["archive_sha256_verified"] = True
                write_marker(extracted, details)
                if target.exists() or target.is_symlink():
                    raise ValueError(f"Subset appeared during download; refusing to replace it: {target}")
                extracted.rename(target)
            print(f"Installed {subset}: {AUDIO_COUNTS[subset]} audio files", file=sys.stderr)
        result["subsets"][subset] = details
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", required=True, type=Path,
        help="Parent directory containing LibriSpeech (not the LibriSpeech directory itself)",
    )
    parser.add_argument("--subsets", nargs="+", choices=SUPPORTED_SUBSETS, default=DEFAULT_SUBSETS)
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help="Archive directory URL; CN mirror: https://openslr.magicdatatech.com/resources/12/",
    )
    args = parser.parse_args()
    try:
        result = download_librispeech(args.dataset_root, args.subsets, args.base_url)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
