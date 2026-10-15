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
        text = "t͡ʃ d͡ʒ a j o w ə ɹ ɹ̩ ɡ"
        ids = self.tokenizer.encode(text)
        self.assertEqual(len(ids), 10)
        self.assertEqual(self.tokenizer.decode(ids), text)
        self.assertEqual(self.tokenizer.decode([38, *ids, 38]), text)
        self.assertEqual(self.tokenizer.get_piece_size(), 38)
        self.assertEqual(self.tokenizer.blank_id, 38)
        self.assertEqual(self.tokenizer.encode("a b d"), [0, 1, 2])

    def test_preserves_tie_bars_syllabic_r_and_rejects_other_conventions(self):
        self.assertEqual(self.tokenizer.tokenize("t͡ʃ   ɹ̩"), ["t͡ʃ", "ɹ̩"])
        for unknown in ("r", "ɻ", "kʰ", "iː", "T", "HH", "t͡ʃajld", "ˈæ", "<unk>", "aɪ", "ɝ", "ɚ", "tʃ"):
            with self.subTest(unknown=unknown), self.assertRaises(ValueError):
                self.tokenizer.encode(unknown)

    def test_shared_encoding_preserves_phone_boundaries(self):
        self.assertEqual(encode_transcript(self.tokenizer, "t͡ʃ a j", False), [17, 0, 8])

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
