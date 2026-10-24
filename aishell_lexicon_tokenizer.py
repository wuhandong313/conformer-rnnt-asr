"""One unchanged AISHELL dictionary symbol per token, with a separate RNNT blank."""
from __future__ import annotations

import json
from pathlib import Path
import re

from conformer_rnnt.aishell.protocol import LEXICON_SHA256, fingerprint

DEFAULT_AISHELL_VOCAB = Path(__file__).resolve().parent / "configs/aishell_lexicon_vocab.json"


class AISHELLLexiconTokenizer:
    is_phone_tokenizer = True

    def __init__(self, vocab_path=DEFAULT_AISHELL_VOCAB):
        self.spec = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        tokens = self.spec.get("tokens", [])
        if (self.spec.get("version") != 1 or self.spec.get("type") != "aishell_lexicon"
                or self.spec.get("tokenization") != "whitespace"
                or self.spec.get("lexicon_sha256") != LEXICON_SHA256
                or not isinstance(tokens, list) or len(tokens) != 217
                or any(not isinstance(t, str) or re.fullmatch(r"[a-z]+[1-5]?", t) is None for t in tokens)
                or tokens != sorted(set(tokens)) or "sil" not in tokens):
            raise ValueError("Expected the pinned 217-symbol raw AISHELL dictionary inventory")
        self.tokens = tuple(tokens)
        self.phone_to_id = {p: i for i, p in enumerate(tokens)}
        self.blank_id = len(tokens)
        self.non_emitting_ids = (self.blank_id,)
        self.fingerprint = fingerprint(self.spec)

    def get_piece_size(self):
        return len(self.tokens)

    def tokenize(self, text):
        tokens = text.split()
        for token in tokens:
            if token not in self.phone_to_id:
                raise ValueError(f"Unknown AISHELL dictionary symbol: {token!r}")
        return tokens

    def encode(self, text):
        return [self.phone_to_id[t] for t in self.tokenize(text)]

    def decode(self, ids):
        output = []
        for value in ids:
            index = int(value)
            if index == self.blank_id:
                continue
            if not 0 <= index < self.blank_id:
                raise ValueError(f"Invalid AISHELL token ID: {index}")
            output.append(self.tokens[index])
        return " ".join(output)
