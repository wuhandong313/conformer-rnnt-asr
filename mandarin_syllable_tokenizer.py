"""One original tone-marked Pinyin syllable per token; no G2P or dictionary."""
import json
from pathlib import Path
import re

from conformer_rnnt.aishell.protocol import fingerprint

SYLLABLE = re.compile(r"[a-zü:]+[0-5]")


def validate_syllables(tokens):
    if not tokens or any(not isinstance(t, str) or SYLLABLE.fullmatch(t) is None for t in tokens):
        raise ValueError("Expected original whitespace-delimited tone-marked Pinyin syllables")
    return list(tokens)


class MandarinSyllableTokenizer:
    is_phone_tokenizer = True  # Preserve complete whitespace-delimited label units.
    target_type = "tone_marked_syllables"

    def __init__(self, vocab_path):
        self.spec = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        tokens = validate_syllables(self.spec.get("tokens", []))
        if (self.spec.get("version") != 1 or self.spec.get("type") != "mandarin_syllable"
                or self.spec.get("tokenization") != "whitespace"
                or self.spec.get("normalization") != "none_preserve_original"
                or tokens != sorted(set(tokens))):
            raise ValueError("Invalid frozen Mandarin syllable inventory")
        self.tokens = tuple(tokens)
        self.token_to_id = {token: i for i, token in enumerate(tokens)}
        self.blank_id = len(tokens)
        self.non_emitting_ids = (self.blank_id,)
        self.fingerprint = fingerprint(self.spec)

    def get_piece_size(self):
        return len(self.tokens)

    def tokenize(self, text):
        tokens = validate_syllables(text.split()) if text.strip() else []
        if any(t not in self.token_to_id for t in tokens):
            raise ValueError("Unknown syllable: inventory must be explicitly reviewed; no UNK/blank substitution")
        return tokens

    def encode(self, text):
        return [self.token_to_id[t] for t in self.tokenize(text)]

    def decode(self, ids):
        tokens = []
        for value in ids:
            index = int(value)
            if index == self.blank_id:
                continue
            if not 0 <= index < len(self.tokens):
                raise ValueError("Invalid syllable token ID")
            tokens.append(self.tokens[index])
        return " ".join(tokens)
