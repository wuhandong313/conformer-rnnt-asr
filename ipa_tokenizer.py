"""Fixed, whitespace-delimited IPA phones; no SentencePiece dependency."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path


DEFAULT_VOCAB = Path(__file__).resolve().parent / "configs" / "ipa_vocab.json"


class FixedIPATokenizer:
    """One complete IPA phone per token, with an RNNT blank appended at K.

    Transcripts must contain phone boundaries: ``tʃ aɪ l d``. Compact IPA
    strings are rejected rather than guessed or split into Unicode characters.
    """

    def __init__(self, vocab_path: str | Path = DEFAULT_VOCAB):
        self.spec = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        if (self.spec.get("version") != 1 or self.spec.get("normalization") != "NFC"
                or self.spec.get("tokenization") != "whitespace"):
            raise ValueError("Expected version 1 IPA vocabulary with NFC and whitespace tokenization")
        tokens = self.spec.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("IPA vocabulary must contain a nonempty token list")
        for token in tokens:
            if (not isinstance(token, str) or not token or any(c.isspace() for c in token)
                    or unicodedata.normalize("NFC", token) != token
                    or token.startswith("<")):
                raise ValueError(f"Invalid IPA phone token: {token!r}")
        if len(set(tokens)) != len(tokens):
            raise ValueError("IPA vocabulary contains duplicate phones")
        self.tokens = tuple(tokens)
        self.phone_to_id = {phone: i for i, phone in enumerate(tokens)}
        self.aliases = self.spec.get("aliases", {})
        if not isinstance(self.aliases, dict):
            raise ValueError("IPA aliases must be a dictionary")
        if any(key in self.phone_to_id and value != key for key, value in self.aliases.items()):
            raise ValueError("IPA aliases cannot change canonical vocabulary phones")
        if any(value not in self.phone_to_id for value in self.aliases.values()):
            raise ValueError("Every IPA alias must resolve to a vocabulary phone")
        mapping = self.spec.get("arpabet_to_ipa", {})
        overrides = self.spec.get("unstressed_overrides", {})
        if not isinstance(mapping, dict) or not isinstance(overrides, dict):
            raise ValueError("ARPABET mappings must be dictionaries")
        if any(value not in self.phone_to_id for value in [*mapping.values(), *overrides.values()]):
            raise ValueError("ARPABET mappings must resolve to vocabulary phones")
        vowels = self.spec.get("arpabet_vowels", [])
        if not isinstance(vowels, list) or not set(vowels).issubset(mapping):
            raise ValueError("ARPABET vowel list must reference mapped source phones")
        self.blank_id = len(tokens)
        self.non_emitting_ids = (self.blank_id,)
        self.fingerprint = hashlib.sha256(
            json.dumps(self.spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def get_piece_size(self) -> int:
        """Vocabulary size excluding blank; shared with the legacy adapter."""
        return len(self.tokens)

    def tokenize(self, text: str) -> list[str]:
        phones = []
        for index, raw in enumerate(unicodedata.normalize("NFC", text).split()):
            phone = self.aliases.get(raw, raw)
            if phone not in self.phone_to_id:
                raise ValueError(
                    f"Unknown IPA phone {raw!r} at token {index}. "
                    "Use whitespace-delimited IPA phones; convert ARPABET first."
                )
            phones.append(phone)
        return phones

    def encode(self, text: str) -> list[int]:
        return [self.phone_to_id[phone] for phone in self.tokenize(text)]

    def decode(self, ids) -> str:
        phones = []
        for value in ids:
            index = int(value)
            if index == self.blank_id:
                continue
            if not 0 <= index < len(self.tokens):
                raise ValueError(f"Invalid IPA token ID: {index}")
            phones.append(self.tokens[index])
        return " ".join(phones)

    def convert_arpabet(self, text: str) -> str:
        """Remove lexical stress, preserving AH0→ə and ER0→ɚ reductions."""
        mapping = self.spec["arpabet_to_ipa"]
        overrides = self.spec["unstressed_overrides"]
        vowels = set(self.spec["arpabet_vowels"])
        phones = []
        for raw in text.split():
            token = raw.upper()
            match = re.fullmatch(r"([A-Z]+)([012]?)", token)
            if not match or match[1] not in mapping or (match[2] and match[1] not in vowels):
                raise ValueError(f"Unknown or invalid ARPABET phone: {raw!r}")
            phones.append(overrides.get(token, mapping[match[1]]))
        return " ".join(phones)


def encode_transcript(tokenizer, text: str, strip_spaces: bool = True) -> list[int]:
    """Share exactly the same encoding between collation and batch guarding."""
    if not isinstance(tokenizer, FixedIPATokenizer):
        text = text.lower()
        if strip_spaces:
            text = text.replace(" ", "")
    return tokenizer.encode(text)
