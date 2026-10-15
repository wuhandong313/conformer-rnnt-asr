"""Fixed CMU/ARPABET phones with stress, without SentencePiece."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEFAULT_PHONE69_VOCAB = Path(__file__).resolve().parent / "configs" / "phone69_vocab.json"
CONSONANTS = tuple("B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split())
VOWELS = tuple("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
PHONE69 = tuple(sorted((*CONSONANTS, *(f"{v}{s}" for v in VOWELS for s in range(3)))))


class FixedPhone69Tokenizer:
    """Encode each whitespace-delimited phone as one stable ID.

    Stress digits are part of the phone: AH0, AH1 and AH2 are distinct.
    Source annotation ``spn`` is explicitly filtered, leaving exactly 69 phones.
    Blank follows all label tokens and also pads targets outside their lengths.
    """

    is_phone_tokenizer = True

    def __init__(self, vocab_path: str | Path = DEFAULT_PHONE69_VOCAB):
        self.spec = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        if (self.spec.get("version") != 1 or self.spec.get("type") != "phone69"
                or self.spec.get("tokenization") != "whitespace"):
            raise ValueError("Expected version 1 whitespace phone69 vocabulary")
        tokens = self.spec.get("tokens")
        if not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens):
            raise ValueError("phone69 vocabulary must contain a token list")
        if len(tokens) != len(set(tokens)):
            raise ValueError("phone69 vocabulary contains duplicate phones")
        if set(tokens) != set(PHONE69):
            raise ValueError("phone69 vocabulary must contain exactly the 69 stressed phones")
        if self.spec.get("ignored_tokens") != ["spn"]:
            raise ValueError("phone69 vocabulary must explicitly filter only spn")
        self.tokens = tuple(tokens)
        self.phone_to_id = {phone: i for i, phone in enumerate(tokens)}
        self.blank_id = len(tokens)
        self.non_emitting_ids = (self.blank_id,)
        self.fingerprint = hashlib.sha256(
            json.dumps(self.spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def get_piece_size(self) -> int:
        return len(self.tokens)

    def tokenize(self, text: str) -> list[str]:
        phones = []
        for index, phone in enumerate(text.split()):
            if phone == "spn":
                continue
            if phone not in self.phone_to_id:
                raise ValueError(
                    f"Unknown phone69 phone {phone!r} at token {index}. "
                    "Expected uppercase ARPABET with vowel stress 0/1/2; "
                    "IPA labels and concatenated phones are not accepted."
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
                raise ValueError(f"Invalid phone69 token ID: {index}")
            phones.append(self.tokens[index])
        return " ".join(phones)
