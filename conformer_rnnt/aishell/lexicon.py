"""Dictionary-constrained context selection. Original phone strings never change."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import importlib.metadata
from pathlib import Path
import re

from .protocol import G2PM_VERSION, file_sha, fingerprint

INITIALS = frozenset("aa b c ch d ee f g h ii j k l m n oo p q r s sh t uu vv x z zh".split())
SPECIAL_WORDS = {"SIL": ("sil",), "<SPOKEN_NOISE>": ("sil",)}


@dataclass(frozen=True)
class Pronunciation:
    phones: tuple[str, ...]
    lines: tuple[int, ...]


def canonical_pinyin(value):
    if not isinstance(value, str):
        return None
    value = value.lower().replace("ü", "v").replace("u:", "v")
    # g2pM's numbered output is required; no guessed tone or text fallback.
    return value if re.fullmatch(r"[a-zv]+[1-5]", value) else None


def phone_syllable(initial, final):
    """Comparison spelling only. Raw aa/ii/vv/ix/iy symbols stay in targets."""
    if re.fullmatch(r"[a-z]+[1-5]", final) is None:
        return None
    stem, tone = final[:-1], final[-1]
    if stem in {"ix", "iy", "iz"}:
        stem = "i"
    if initial in {"", "aa", "ee", "oo"}:
        # Standalone finals occur in official entries, especially er2/er4.
        if initial == "":
            if stem.startswith("i"):
                return phone_syllable("ii", final)
            if stem.startswith("u"):
                return phone_syllable("uu", final)
            if stem.startswith("v"):
                return phone_syllable("vv", final)
        return stem + tone
    if initial == "ii":
        base = {"i": "yi", "in": "yin", "ing": "ying", "ia": "ya", "ian": "yan",
                "iang": "yang", "iao": "yao", "ie": "ye", "iong": "yong", "iu": "you"}.get(stem)
    elif initial == "uu":
        base = {"u": "wu", "ua": "wa", "uai": "wai", "uan": "wan", "uang": "wang",
                "ui": "wei", "un": "wen", "uo": "wo", "ueng": "weng"}.get(stem)
    elif initial == "vv":
        base = {"v": "yu", "ve": "yue", "van": "yuan", "vn": "yun", "iong": "yong"}.get(stem)
    else:
        if initial not in INITIALS:
            return None
        if initial in {"j", "q", "x"}:
            stem = {"v": "u", "ve": "ue", "van": "uan", "vn": "un"}.get(stem, stem)
        base = initial + stem
    return None if base is None else base + tone


def matches_pinyin(phones, syllables):
    """Match variable-length raw phone chunks to a supplied pinyin sequence.

    This is constrained equality, not edit-distance matching. A syllable may be
    an initial+final pair or a standalone final, as observed in the source.
    """
    phones, syllables = tuple(phones), tuple(syllables)
    if any(canonical_pinyin(s) is None for s in syllables):
        return False
    wanted = tuple(canonical_pinyin(s) for s in syllables)

    @lru_cache(None)
    def visit(i, j):
        if j == len(wanted):
            return i == len(phones)
        if i == len(phones):
            return False
        if phone_syllable("", phones[i]) == wanted[j] and visit(i + 1, j + 1):
            return True
        return (i + 1 < len(phones) and phones[i] in INITIALS
                and phone_syllable(phones[i], phones[i + 1]) == wanted[j]
                and visit(i + 2, j + 1))
    return visit(0, 0)


class Lexicon:
    def __init__(self, path):
        self.sha256 = file_sha(path)
        entries = defaultdict(lambda: defaultdict(list))
        irregular = []
        rows = 0
        for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"Dictionary line {number} has no pronunciation")
            word, phones = parts[0], tuple(parts[1:])
            if any(re.fullmatch(r"[a-z]+[1-5]?", p) is None for p in phones):
                raise ValueError(f"Invalid raw dictionary phone at line {number}")
            entries[word][phones].append(number)
            rows += 1
            if len(phones) != 2 * len(word):
                irregular.append({"line": number, "word": word, "phones": list(phones)})
        self.entries = {w: tuple(Pronunciation(p, tuple(lines)) for p, lines in values.items())
                        for w, values in entries.items()}
        self.symbols = sorted({p for values in self.entries.values() for value in values for p in value.phones})
        self.audit = {"lexicon_sha256": self.sha256, "rows": rows, "unique_words": len(entries),
                      "multiple_pronunciation_words": sum(len(v) > 1 for v in self.entries.values()),
                      "raw_symbol_count": len(self.symbols), "symbols": self.symbols,
                      "non_two_symbols_per_character": irregular}

    def select_sentence(self, utterance, words, predictions, *, split, audio_path, resolutions=None):
        words = list(words)
        text = "".join(words)
        resolutions = resolutions or {}
        predictions = list(predictions)
        aligned = len(predictions) == len(text)
        selected, provenance, issues = [], [], []
        offset = 0
        for word in words:
            end = offset + len(word)
            predicted = predictions[offset:end] if aligned else []
            candidates = self.entries.get(word, ())
            matching = [p for p in candidates if aligned and matches_pinyin(p.phones, predicted)]
            if word in SPECIAL_WORDS:
                matching = [p for p in candidates if p.phones == SPECIAL_WORDS[word]]
            chosen = matching[0] if len(matching) == 1 else None
            reason = "dictionary_unique_verified" if len(candidates) == 1 else "context_unique_dictionary_candidate"
            if chosen is None:
                kind = ("missing_word" if not candidates else "prediction_alignment" if not aligned
                        else "ambiguous_equivalent_candidates" if len(matching) > 1 else "prediction_dictionary_conflict")
                issue = {"utterance_id": utterance, "split": split, "audio_path": audio_path,
                         "text": text, "words": words, "word": word, "character_span": [offset, end],
                         "prediction": predicted, "kind": kind, "lexicon_sha256": self.sha256,
                         "candidates": [{"phones": list(p.phones), "source_lines": list(p.lines)} for p in candidates]}
                issue["issue_id"] = fingerprint(issue)
                resolution = resolutions.get(issue["issue_id"])
                if resolution:
                    if not resolution.get("reason") or not resolution.get("version") or resolution.get("issue_id") != issue["issue_id"]:
                        raise ValueError("Resolution requires the exact issue ID, a reason and a decision version")
                    chosen = next((p for p in candidates if list(p.phones) == resolution.get("candidate_phones")), None)
                    if chosen is None:
                        raise ValueError("Manual resolution must select an existing raw dictionary candidate")
                    issue["resolution"] = resolution
                    reason = "manual_existing_dictionary_candidate"
                issue["resolved"] = chosen is not None
                issues.append(issue)
            if chosen is not None:
                selected.extend(chosen.phones)
                provenance.append({"word": word, "character_span": [offset, end], "phones": list(chosen.phones),
                                   "source_lines": list(chosen.lines), "method": reason})
            offset = end
        unresolved = any(not issue["resolved"] for issue in issues)
        return {"phones": None if unresolved else selected, "label_provenance": provenance, "issues": issues}


class PinnedG2PM:
    """Lazy local-only adapter. Installing dependencies is an explicit operator step."""
    def __init__(self):
        if importlib.metadata.version("g2pM") != G2PM_VERSION:
            raise ValueError(f"Preparation requires g2pM=={G2PM_VERSION}")
        import g2pM
        root = Path(g2pM.__file__).resolve().parent
        files = {str(p.relative_to(root)): file_sha(p) for p in root.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        import numpy
        self.metadata = {"package": "g2pM", "version": G2PM_VERSION, "numpy": numpy.__version__,
                         "package_and_model_files": files, "tone": True, "char_split": True}
        self.model = g2pM.G2pM()

    def __call__(self, text):
        return self.model(text, tone=True, char_split=True)
