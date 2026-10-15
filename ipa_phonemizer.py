"""Consistent English word → CMU phones for offline IPA label preparation."""

from __future__ import annotations

import hashlib
import json
import string
from importlib.metadata import PackageNotFoundError, version

from ipa_tokenizer import FixedIPATokenizer


class WordPhonemizer:
    """Use CMUdict's first pronunciation, with cached g2p-en fallback for OOV.

    This prepares supervision offline. It is separate from the fixed IPA
    tokenizer used in training, which never imports g2p-en or SentencePiece.
    """

    def __init__(self, tokenizer: FixedIPATokenizer, g2p=None):
        if g2p is None:
            try:
                from g2p_en import G2p
            except ImportError as error:
                raise RuntimeError("Word label preparation requires g2p-en: pip install g2p-en==2.1.0") from error
            g2p = G2p()
        self.g2p = g2p
        self.tokenizer = tokenizer
        self.cache = {}
        self.oov_words = set()
        try:
            self.g2p_version = version("g2p-en")
        except PackageNotFoundError:
            self.g2p_version = "custom-backend"
        self.dictionary_sha256 = hashlib.sha256(
            json.dumps(g2p.cmu, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def arpabet(self, text: str) -> str:
        phones = []
        for raw_word in text.split():
            word = raw_word.lower()
            if word not in self.cache:
                if word in self.g2p.cmu:
                    pronunciation = self.g2p.cmu[word][0]
                else:
                    self.oov_words.add(word)
                    pronunciation = [
                        phone for phone in self.g2p(word)
                        if phone.strip() and not all(c in string.punctuation for c in phone)
                    ]
                if not pronunciation:
                    raise ValueError(f"No pronunciation for word {raw_word!r}")
                # Validate the backend's phones before caching; never emit UNK.
                self.tokenizer.convert_arpabet(" ".join(pronunciation))
                self.cache[word] = tuple(pronunciation)
            phones.extend(self.cache[word])
        return " ".join(phones)

    def audit(self) -> dict:
        return {
            "backend": "CMUdict first pronunciation + g2p-en OOV fallback",
            "g2p_en_version": self.g2p_version,
            "cmudict_sha256": self.dictionary_sha256,
            "unique_words": len(self.cache),
            "oov_word_count": len(self.oov_words),
            "oov_words": sorted(self.oov_words),
            "pronunciations_sha256": hashlib.sha256(
                json.dumps(self.cache, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
