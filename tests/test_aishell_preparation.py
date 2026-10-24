import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from conformer_rnnt.aishell.preparation import freeze_bundle, prepare_labels
from conformer_rnnt.aishell.protocol import file_sha, read_jsonl, write_json


class AISHELLPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audit = self.root / "audited"; self.audit.mkdir()
        row = {"utterance_id": "test1", "split": "test", "audio_path": "wav/test/s/test1.wav",
               "text": "银行", "words": ["银行"]}
        (self.audit / "audio_index.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(self.audit / "audio_audit.json", {"valid": True,
                   "audio_index_sha256": file_sha(self.audit / "audio_index.jsonl")})
        self.lexicon = self.root / "lexicon.txt"
        self.lexicon.write_text("银行 ii in2 h ang2\n银行 ii in2 x ing2\n银行 ii in2 h ang2\n")
        self.tokenizer = SimpleNamespace(tokens=tuple(sorted("ii in2 h ang2 x ing2".split())),
                                         fingerprint="test_inventory", tokenize=lambda s: s.split())

    def predictor(self, values):
        class Predictor:
            metadata = {"package": "g2pM", "version": "0.1.2.5", "package_and_model_files": {"fixture": "sha256"}}
            def __call__(self, sentence):
                if sentence != "银行":
                    raise AssertionError("Predict the full sentence")
                return values
        return Predictor()

    def test_constrained_label_pipeline_traces_duplicates_and_audits_conflicts(self):
        with patch("conformer_rnnt.aishell.preparation.LEXICON_SHA256", file_sha(self.lexicon)):
            report = prepare_labels(self.audit, self.lexicon, self.predictor(["yin2", "hang2"]),
                                    self.root / "valid", tokenizer=self.tokenizer)
            self.assertTrue(report["valid"])
            row = next(read_jsonl(self.root / "valid/test.jsonl"))
            self.assertEqual(row["phones"], "ii in2 h ang2".split())
            self.assertEqual(row["label_provenance"][0]["source_lines"], [1, 3])
            bad = prepare_labels(self.audit, self.lexicon, self.predictor(["yin1", "hang2"]),
                                 self.root / "conflict", tokenizer=self.tokenizer)
        self.assertFalse(bad["valid"])
        issue = next(read_jsonl(self.root / "conflict/issues.jsonl"))
        self.assertFalse(issue["resolved"])
        self.assertIsNone(next(read_jsonl(self.root / "conflict/test.jsonl"))["phones"])
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            freeze_bundle(self.root / "missing", self.root / "conflict", self.root / "frozen")
        self.assertFalse((self.root / "frozen").exists())

    def test_missing_local_inputs_and_unpinned_predictor_fail_without_fetching(self):
        from conformer_rnnt.aishell.preparation import audit_audio
        with self.assertRaisesRegex(FileNotFoundError, "never downloads"):
            audit_audio(self.root / "absent", self.root / "should_not_exist")
        self.assertFalse((self.root / "should_not_exist").exists())
        predictor = self.predictor(["yin2", "hang2"])
        predictor.metadata = {**predictor.metadata, "version": "other"}
        with patch("conformer_rnnt.aishell.preparation.LEXICON_SHA256", file_sha(self.lexicon)):
            with self.assertRaisesRegex(ValueError, "pinned"):
                prepare_labels(self.audit, self.lexicon, predictor, self.root / "rejected", tokenizer=self.tokenizer)
        self.assertFalse((self.root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
