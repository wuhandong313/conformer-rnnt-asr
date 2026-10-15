"""Fixed IPA → 24-dimensional PanPhon features; runtime lookup is stdlib-only."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from ipa_tokenizer import DEFAULT_VOCAB, FixedIPATokenizer


DEFAULT_FEATURES = Path(__file__).resolve().parent / "configs" / "ipa_articulatory_features.json"


class IPAFeatureTable:
    def __init__(self, tokenizer=None, path=DEFAULT_FEATURES):
        tokenizer = tokenizer or FixedIPATokenizer()
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        if spec.get("version") != 1 or spec.get("vocab_sha256") != tokenizer.fingerprint:
            raise ValueError("PanPhon table does not match the configured IPA vocabulary")
        if spec.get("tokens") != list(tokenizer.tokens):
            raise ValueError("PanPhon rows must use exactly the tokenizer's token IDs")
        self.names = tuple(spec["feature_names"])
        self.vectors = tuple(tuple(row) for row in spec["vectors"])
        if len(self.names) != 24 or len(set(self.names)) != 24:
            raise ValueError("Expected 24 distinct PanPhon features")
        if len(self.vectors) != len(tokenizer.tokens) or any(
            len(row) != 24 or any(value not in (-1, 0, 1) for value in row)
            for row in self.vectors
        ):
            raise ValueError("Expected one 24-dimensional {-1, 0, 1} row per IPA token")
        self.blank_id = tokenizer.blank_id
        self.fingerprint = hashlib.sha256(
            json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def lookup(self, ids):
        """Return vectors and a speech mask; blank has zeros and mask=False."""
        vectors, mask = [], []
        for value in ids:
            index = int(value)
            if index == self.blank_id:
                vectors.append([0] * 24)
                mask.append(False)
            elif 0 <= index < len(self.vectors):
                vectors.append(list(self.vectors[index]))
                mask.append(True)
            else:
                raise ValueError(f"Invalid IPA token ID: {index}")
        return vectors, mask


def build_panphon_table(vocab_path=DEFAULT_VOCAB):
    """Rebuild offline; reject partial parsing or multi-segment approximations."""
    import panphon

    tokenizer = FixedIPATokenizer(vocab_path)
    ft = panphon.FeatureTable()
    if len(ft.names) != 24:
        raise ValueError(f"PanPhon exposes {len(ft.names)} features instead of 24")
    vectors = []
    for phone in tokenizer.tokens:
        if ft.ipa_segs(phone) != [phone]:
            raise ValueError(f"PanPhon does not recognize the complete IPA segment {phone!r}")
        segments = ft.word_fts(phone)
        if len(segments) != 1:
            raise ValueError(f"Expected one PanPhon segment for token {phone!r}")
        vectors.append([int(value) for value in segments[0].numeric()])
    return {
        "version": 1, "panphon_version": version("panphon"),
        "vocab_sha256": tokenizer.fingerprint,
        "tokens": list(tokenizer.tokens), "feature_names": ft.names,
        "vectors": vectors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--out", type=Path, default=DEFAULT_FEATURES)
    args = parser.parse_args()
    table = build_panphon_table(args.vocab)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    IPAFeatureTable(FixedIPATokenizer(args.vocab), args.out)
    print(f"Verified {len(table['tokens'])} IPA tokens × {len(table['feature_names'])} PanPhon features")
