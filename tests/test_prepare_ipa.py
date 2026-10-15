import tempfile
import unittest
from pathlib import Path

from prepare_ipa import prepare_dataset
from ipa_phonemizer import WordPhonemizer
from ipa_tokenizer import FixedIPATokenizer


class PrepareIPATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.chapter = self.root / "LibriSpeech/train-clean-100/1/2"
        self.chapter.mkdir(parents=True)
        self.source = self.chapter / "1-2-ph.trans.txt"
        self.word = self.chapter / "1-2.trans.txt"
        self.output = self.chapter / "1-2-ipa.trans.txt"
        self.source.write_text("1-2-0000 CH AY1 L D\n1-2-0001 AH0 ER0\n")
        self.word.write_text("1-2-0000 CHILD\n1-2-0001 A HER\n")
        for uid in ("1-2-0000", "1-2-0001"):
            (self.chapter / f"{uid}.flac").touch()

    def prepare(self, **kwargs):
        return prepare_dataset(self.root, ["train-clean-100"], source_type="arpabet", **kwargs)

    def test_audit_conversion_and_idempotence_preserve_originals(self):
        before = (self.source.read_bytes(), self.word.read_bytes())
        report = self.prepare(check_only=True)
        self.assertEqual(report["totals"]["utterances"], 2)
        self.assertEqual(report["totals"]["phones"], 6)
        self.assertFalse(self.output.exists())
        report = self.prepare()
        self.assertEqual(report["totals"]["written_ipa_files"], 1)
        self.assertEqual(self.output.read_text(), "1-2-0000 tʃ aɪ l d\n1-2-0001 ə ɚ\n")
        self.assertEqual(self.prepare()["totals"]["written_ipa_files"], 0)
        self.assertEqual(before, (self.source.read_bytes(), self.word.read_bytes()))

    def test_unknown_phone_fails_without_writes(self):
        self.source.write_text("1-2-0000 SIL\n1-2-0001 AH0\n")
        with self.assertRaisesRegex(ValueError, "SIL"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_missing_audio_label_fails_without_writes(self):
        self.source.write_text("1-2-0000 CH AY1 L D\n")
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_conflicting_output_requires_explicit_overwrite(self):
        self.output.write_text("do not silently replace\n")
        with self.assertRaisesRegex(ValueError, "Existing IPA file differs"):
            self.prepare()
        self.prepare(overwrite=True)
        self.assertIn("tʃ aɪ", self.output.read_text())

    def test_duplicate_ids_are_rejected(self):
        self.source.write_text("1-2-0000 CH\n1-2-0000 D\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare()

    def test_words_source_ignores_missing_or_mixed_legacy_labels(self):
        class FakeG2P:
            cmu = {"child": [["CH", "AY1", "L", "D"]],
                   "a": [["AH0"]], "her": [["HH", "ER0"]]}

        self.source.unlink()
        phonemizer = WordPhonemizer(FixedIPATokenizer(), FakeG2P())
        report = prepare_dataset(self.root, ["train-clean-100"], phonemizer=phonemizer)
        self.assertEqual(report["source_type"], "words")
        self.assertEqual(report["totals"]["phones"], 7)
        self.assertEqual(self.output.read_text(), "1-2-0000 tʃ aɪ l d\n1-2-0001 ə h ɚ\n")
        self.assertFalse(self.source.exists())


if __name__ == "__main__":
    unittest.main()
