import tempfile
import unittest
from pathlib import Path

from audit_ipa_data import audit_dataset
from ipa_labels import ExternalIPALabels


class ExternalIPALabelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio_root = self.root / "audio"
        self.labels_root = self.root / "labels"
        chapter = self.audio_root / "LibriSpeech/train-clean-100/1/2"
        chapter.mkdir(parents=True)
        for uid in ("1-2-0000", "1-2-0001"):
            (chapter / f"{uid}.flac").touch()
        folder = self.labels_root / "train_clean_100"
        folder.mkdir(parents=True)
        self.path = folder / "ipa"
        self.path.write_text("1-2-0000 t͡ʃ a j l d\n1-2-0001 ɹ̩ ə\n", encoding="utf-8")

    def test_reads_external_labels_and_audits_without_writes(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        labels = ExternalIPALabels(self.labels_root, "train-clean-100")
        self.assertEqual(labels.phone_counts["1-2-0000"], 5)
        result = audit_dataset(self.audio_root, self.labels_root, ["train-clean-100"])
        self.assertEqual(result["totals"], {"utterances": 2, "phones": 7})
        self.assertEqual((result["num_symbols"], result["feature_dim"]), (38, 24))
        after = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_file_never_falls_back_to_phone69(self):
        self.path.rename(self.path.with_name("phone69.trans.txt"))
        with self.assertRaisesRegex(FileNotFoundError, "Missing external"):
            ExternalIPALabels(self.labels_root, "train-clean-100")

    def test_mismatched_ids_fail(self):
        self.path.write_text("1-2-0000 t͡ʃ\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            audit_dataset(self.audio_root, self.labels_root, ["train-clean-100"])

    def test_duplicate_and_unknown_labels_fail(self):
        for text, message in (("1-2-0000 a\n1-2-0000 j\n", "duplicate"),
                              ("1-2-0000 AH0\n", "Unknown IPA"),
                              ("1-2-0000 \n", "expected utterance")):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    ExternalIPALabels(self.labels_root, "train-clean-100")


if __name__ == "__main__":
    unittest.main()
