#!/usr/bin/env python3
"""Read-only audit of external Epitran IPA labels and matching LibriSpeech audio."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from ipa_features import DEFAULT_FEATURES, IPAFeatureTable
from ipa_labels import ExternalIPALabels
from ipa_tokenizer import DEFAULT_VOCAB, FixedIPATokenizer


SUBSETS = (
    "train-clean-100", "train-clean-360", "train-other-500",
    "dev-clean", "dev-other", "test-clean", "test-other",
)


def audit_dataset(dataset_root, labels_root, subsets=SUBSETS,
                  vocab_path=DEFAULT_VOCAB, features_path=DEFAULT_FEATURES):
    tokenizer = FixedIPATokenizer(vocab_path)
    features = IPAFeatureTable(tokenizer, features_path)
    report = {"dataset_root": str(Path(dataset_root).resolve()),
              "labels_root": str(Path(labels_root).resolve()),
              "vocab_sha256": tokenizer.fingerprint,
              "features_sha256": features.fingerprint,
              "num_symbols": len(tokenizer.tokens), "blank_id": tokenizer.blank_id,
              "feature_dim": len(features.names), "subsets": {}}
    inventory = Counter()
    subsets = tuple(subsets)
    if not subsets or len(set(subsets)) != len(subsets) or any(s not in SUBSETS for s in subsets):
        raise ValueError("Select distinct, supported LibriSpeech subsets")
    for subset in subsets:
        labels = ExternalIPALabels(labels_root, subset, tokenizer=tokenizer)
        directory = Path(dataset_root) / "LibriSpeech" / subset
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing audio subset: {directory}")
        audio_ids = []
        for _, _, files in os.walk(directory):
            audio_ids.extend(name[:-5] for name in files if name.endswith(".flac"))
        if len(set(audio_ids)) != len(audio_ids):
            raise ValueError(f"Duplicate audio IDs in {directory}")
        labels.validate_audio_ids(audio_ids)
        for text in labels.transcripts.values():
            inventory.update(text.split())
        report["subsets"][subset] = {
            "label_file": str(labels.path), "label_sha256": labels.sha256,
            "utterances": len(labels.transcripts),
            "phones": sum(labels.phone_counts.values()),
        }
    report["inventory"] = {phone: inventory[phone] for phone in tokenizer.tokens}
    report["unused_phones"] = [phone for phone in tokenizer.tokens if not inventory[phone]]
    report["unknown_phones"] = []
    report["totals"] = {
        key: sum(stats[key] for stats in report["subsets"].values())
        for key in ("utterances", "phones")
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=SUBSETS)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report:
        report_path = args.report.resolve()
        for root in (args.dataset_root.resolve(), args.labels_root.resolve()):
            if report_path == root or root in report_path.parents:
                parser.error("Save the audit report outside the external data directories")
    try:
        report = audit_dataset(args.dataset_root, args.labels_root, args.subsets,
                               args.vocab, args.features)
    except (ValueError, OSError) as error:
        parser.exit(1, f"IPA data audit failed: {error}\n")
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
