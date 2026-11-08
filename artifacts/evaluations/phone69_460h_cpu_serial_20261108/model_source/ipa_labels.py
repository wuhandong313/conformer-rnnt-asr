"""Read external Epitran label files without copying or rewriting datasets."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ipa_tokenizer import FixedIPATokenizer


class ExternalIPALabels:
    """One file per subset: ``<labels_root>/train_clean_100/ipa``."""

    def __init__(self, labels_root, subset, filename="ipa", tokenizer=None):
        self.tokenizer = tokenizer or FixedIPATokenizer()
        if filename != self.tokenizer.spec["label_filename"]:
            raise ValueError("Label filename does not match the configured IPA inventory variant")
        self.path = Path(labels_root) / subset.replace("-", "_") / filename
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Missing external Epitran IPA labels: {self.path}. "
                "Set paths.ipa_labels_path to the existing label root; no ARPABET fallback is used."
            )
        content = self.path.read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        self.transcripts = {}
        self.phone_counts = {}
        for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not re.fullmatch(r"\d+-\d+-\d+", parts[0]):
                raise ValueError(f"{self.path}:{number}: expected utterance ID and IPA phones")
            utterance, text = parts
            if utterance in self.transcripts:
                raise ValueError(f"{self.path}:{number}: duplicate utterance ID {utterance}")
            try:
                phones = self.tokenizer.tokenize(text)
                if not phones:
                    raise ValueError("Empty IPA target")
            except ValueError as error:
                raise ValueError(f"{self.path}:{number} [{utterance}]: {error}") from error
            self.transcripts[utterance] = " ".join(phones)
            self.phone_counts[utterance] = len(phones)
        if not self.transcripts:
            raise ValueError(f"Empty IPA label file: {self.path}")

    def validate_audio_ids(self, audio_ids):
        audio_ids = set(audio_ids)
        label_ids = set(self.transcripts)
        if label_ids != audio_ids:
            raise ValueError(
                f"{self.path}: audio/IPA IDs differ; "
                f"missing labels={sorted(audio_ids - label_ids)[:5]}, "
                f"labels without audio={sorted(label_ids - audio_ids)[:5]}"
            )
