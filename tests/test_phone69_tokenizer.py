import copy
import json
import tempfile
import unittest
from pathlib import Path

from ipa_tokenizer import encode_transcript
from phone69_tokenizer import CONSONANTS, PHONE69, VOWELS, FixedPhone69Tokenizer


class Phone69TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FixedPhone69Tokenizer()

    def test_inventory_and_stable_ids(self):
        self.assertEqual((len(CONSONANTS), len(VOWELS), len(PHONE69)), (24, 15, 69))
        self.assertEqual(self.tokenizer.tokens, PHONE69)
        self.assertEqual(self.tokenizer.get_piece_size(), 69)
        self.assertEqual(self.tokenizer.blank_id, 69)
        self.assertEqual(self.tokenizer.encode("AA0 AA1 AA2"), [0, 1, 2])
        self.assertEqual(self.tokenizer.encode("spn"), [])

    def test_whole_phones_preserve_stress_and_filter_only_spn(self):
        text = "CH AE1 P T ER0 AH0 AH1 AH2 spn"
        ids = encode_transcript(self.tokenizer, text, False)
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids[-3:])), 3)
        self.assertEqual(self.tokenizer.decode([69, *ids, 69]), text.removesuffix(" spn"))
        self.assertEqual(self.tokenizer.tokenize("CH\t AE1\n"), ["CH", "AE1"])
        for unknown in ("ch", "AH", "AH3", "SIL", "SPN", "<unk>", "t͡ʃ", "CHAE1"):
            with self.subTest(unknown=unknown), self.assertRaisesRegex(ValueError, "Unknown phone69"):
                self.tokenizer.encode(unknown)
        with self.assertRaises(ValueError):
            self.tokenizer.decode([71])

    def test_checkpoint_fingerprint_detects_id_changes(self):
        spec = copy.deepcopy(self.tokenizer.spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            spec["tokens"][0], spec["tokens"][1] = spec["tokens"][1], spec["tokens"][0]
            path.write_text(json.dumps(spec))
            self.assertNotEqual(FixedPhone69Tokenizer(path).fingerprint, self.tokenizer.fingerprint)
            spec["tokens"].append("AA0")
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                FixedPhone69Tokenizer(path)


if __name__ == "__main__":
    unittest.main()
