import copy
import json
import tempfile
import unittest
from pathlib import Path

from ipa_tokenizer import FixedIPATokenizer, encode_transcript


class IPATokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FixedIPATokenizer()

    def test_complete_phones_roundtrip_and_blank(self):
        text = "tʃ dʒ aɪ aʊ eɪ oʊ ɔɪ ɝ ɚ ə ɹ ɡ"
        ids = self.tokenizer.encode(text)
        self.assertEqual(len(ids), 12)
        self.assertEqual(self.tokenizer.decode(ids), text)
        self.assertEqual(self.tokenizer.decode([41, *ids, 41]), text)
        self.assertEqual(self.tokenizer.get_piece_size(), 41)
        self.assertEqual(self.tokenizer.blank_id, 41)
        self.assertEqual(self.tokenizer.encode("ɑ æ ʌ"), [0, 1, 2])

    def test_only_explicit_aliases_are_normalized(self):
        self.assertEqual(self.tokenizer.tokenize("t͡ʃ d͜ʒ g"), ["tʃ", "dʒ", "ɡ"])
        for unknown in ("r", "ɻ", "kʰ", "iː", "T", "HH", "tʃaɪld", "ˈæ", "<unk>"):
            with self.subTest(unknown=unknown), self.assertRaises(ValueError):
                self.tokenizer.encode(unknown)

    def test_stress_reductions(self):
        self.assertEqual(
            self.tokenizer.convert_arpabet("AH0 AH1 AH2 ER0 ER1 ER2 CH AY1 L D"),
            "ə ʌ ʌ ɚ ɝ ɝ tʃ aɪ l d",
        )
        self.assertEqual(self.tokenizer.convert_arpabet("hh iy1"), "h i")
        for invalid in ("T1", "SIL", "AX", "AH3", "A", "tʃ"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.tokenizer.convert_arpabet(invalid)

    def test_shared_encoding_preserves_phone_boundaries(self):
        self.assertEqual(encode_transcript(self.tokenizer, "tʃ aɪ", False), [7, 5])

        class LegacyTokenizer:
            def encode(self, text):
                return text

        self.assertEqual(encode_transcript(LegacyTokenizer(), "HH IY1", True), "hhiy1")
        self.assertEqual(encode_transcript(LegacyTokenizer(), "HELLO WORLD", False), "hello world")

    def test_fingerprint_detects_reordered_ids_and_duplicate_inventory(self):
        spec = copy.deepcopy(self.tokenizer.spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            spec["tokens"][0], spec["tokens"][1] = spec["tokens"][1], spec["tokens"][0]
            path.write_text(json.dumps(spec))
            self.assertNotEqual(FixedIPATokenizer(path).fingerprint, self.tokenizer.fingerprint)
            spec["tokens"].append("ɑ")
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                FixedIPATokenizer(path)


if __name__ == "__main__":
    unittest.main()
