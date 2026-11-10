import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.data.extract_phone69 import DEFAULT_SUBSETS, extract_phone69


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "extract_phone69.py"


class ExtractPhone69Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.labels_root = self.root / "labels"
        self.dataset_root = self.root / "datasets"

    def make_subset(self, subset, records, *, source_name="phone69.trans.txt", audio_ids=None):
        source = self.labels_root / subset.replace("-", "_") / source_name
        source.parent.mkdir(parents=True)
        source.write_text("".join(f"{uid} {phones}\n" for uid, phones in records), encoding="utf-8")
        if audio_ids is None:
            audio_ids = [uid for uid, _ in records]
        for uid in audio_ids:
            speaker, chapter, _ = uid.split("-")
            audio = self.dataset_root / "LibriSpeech" / subset / speaker / chapter / f"{uid}.flac"
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"fixture")
        return source

    def outputs(self):
        return list(self.dataset_root.rglob("*-ph.trans.txt"))

    def extract(self, subsets=("train-clean-100",), **kwargs):
        return extract_phone69(self.labels_root, self.dataset_root, subsets, **kwargs)

    def test_distributes_whole_phones_preserves_spn_and_is_idempotent(self):
        source = self.make_subset("train-clean-100", [
            ("19-20-0002", "CH  AE1\tspn ER0 AH2"),
            ("19-21-0001", "SH IY0"),
            ("19-20-0001", "B AA0 AA1 AA2"),
        ])
        source_before = source.read_bytes()
        manifest = self.extract()
        chapter = self.dataset_root / "LibriSpeech/train-clean-100/19/20/19-20-ph.trans.txt"
        expected = "19-20-0001 B AA0 AA1 AA2\n19-20-0002 CH  AE1\tspn ER0 AH2\n"
        self.assertEqual(chapter.read_text(encoding="utf-8"), expected)
        self.assertEqual(len(self.outputs()), 2)
        details = manifest["subsets"]["train-clean-100"]
        self.assertEqual(details["utterances"], 3)
        self.assertEqual(details["source_sha256"], hashlib.sha256(source_before).hexdigest())
        self.assertEqual(manifest["created_files"], 2)
        original_times = {path: path.stat().st_mtime_ns for path in self.outputs()}
        repeated = self.extract()
        self.assertEqual(repeated["created_files"], 0)
        self.assertEqual(repeated["subsets"]["train-clean-100"]["files_unchanged"], 2)
        self.assertEqual({path: path.stat().st_mtime_ns for path in self.outputs()}, original_times)
        self.assertEqual(source.read_bytes(), source_before)

    def test_default_subsets_and_both_source_names(self):
        for index, subset in enumerate(DEFAULT_SUBSETS):
            self.make_subset(subset, [(f"{index + 1}-10-0000", "CH AE1")],
                             source_name="phone69" if index % 2 else "phone69.trans.txt")
        manifest = extract_phone69(self.labels_root, self.dataset_root)
        self.assertEqual(tuple(manifest["subsets"]), DEFAULT_SUBSETS)
        self.assertEqual(manifest["utterances"], 4)
        self.assertEqual(len(self.outputs()), 4)

    def test_dry_run_and_cli_manifest(self):
        self.make_subset("dev-other", [("20-30-0000", "spn AH0")], source_name="phone69")
        manifest = self.extract(("dev-other",), dry_run=True)
        self.assertEqual(manifest["created_files"], 0)
        self.assertEqual(manifest["subsets"]["dev-other"]["files_to_create"], 1)
        self.assertFalse(self.outputs())
        completed = subprocess.run([
            sys.executable, str(SCRIPT), "--labels-root", str(self.labels_root),
            "--dataset-root", str(self.dataset_root), "--subsets", "dev-other", "--validate-only",
        ], check=True, capture_output=True, text=True, cwd=self.root)
        self.assertTrue(json.loads(completed.stdout)["dry_run"])
        self.assertFalse(self.outputs())

    def test_missing_audio_or_labels_prevents_all_writes(self):
        self.make_subset("train-clean-100", [("1-2-0000", "CH AE1")])
        self.make_subset("dev-clean", [("3-4-0000", "B AA0")], audio_ids=["3-4-0001"])
        with self.assertRaisesRegex(ValueError, "labels without audio=1.*audio without labels=1"):
            self.extract(("train-clean-100", "dev-clean"))
        self.assertFalse(self.outputs())

    def test_duplicate_or_unknown_phone_prevents_all_writes(self):
        self.make_subset("train-clean-100", [("1-2-0000", "CH AE1")])
        source = self.make_subset("dev-clean", [("3-4-0000", "B AA0")])
        invalid_contents = (
            ("3-4-0000 B AA0\n3-4-0000 B AA0\n", "duplicate utterance ID"),
            ("3-4-0000 CHAE1\n", "unknown Phone69"),
            ("3-4-0000 ch AE1\n", "unknown Phone69"),
            ("3-4-0000 SPN\n", "unknown Phone69"),
            ("3-4-0000 AH\n", "unknown Phone69"),
            ("3-4-0000\n", "empty labels"),
            ("3-4-bad B AA0\n", "invalid utterance ID"),
        )
        for contents, error in invalid_contents:
            with self.subTest(contents=contents):
                source.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    self.extract(("train-clean-100", "dev-clean"))
                self.assertFalse(self.outputs())

    def test_existing_conflict_is_found_before_any_new_output(self):
        self.make_subset("train-clean-100", [("1-2-0000", "CH AE1")])
        self.make_subset("dev-clean", [("3-4-0000", "B AA0")])
        conflict = self.dataset_root / "LibriSpeech/dev-clean/3/4/3-4-ph.trans.txt"
        conflict.write_text("previous annotations\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different existing output"):
            self.extract(("train-clean-100", "dev-clean"))
        self.assertEqual(self.outputs(), [conflict])
        self.assertEqual(conflict.read_text(encoding="utf-8"), "previous annotations\n")

    def test_source_manifest_mismatch_prevents_all_writes(self):
        first = self.make_subset("train-clean-100", [("1-2-0000", "CH AE1")])
        second = self.make_subset("dev-clean", [("3-4-0000", "B AA0")])
        manifest = self.root / "sources.json"
        manifest.write_text(json.dumps({"format_version": 1, "subsets": {
            "train-clean-100": {"sha256": hashlib.sha256(first.read_bytes()).hexdigest(), "utterances": 1},
            "dev-clean": {"sha256": "0" * 64, "utterances": 1},
        }}))
        with self.assertRaisesRegex(ValueError, "source hash or utterance count"):
            self.extract(("train-clean-100", "dev-clean"), source_manifest=manifest)
        self.assertFalse(self.outputs())
        data = json.loads(manifest.read_text())
        data["subsets"]["dev-clean"]["sha256"] = hashlib.sha256(second.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(data))
        result = self.extract(("train-clean-100", "dev-clean"), source_manifest=manifest)
        self.assertEqual(result["created_files"], 2)

    def test_ambiguous_sources_and_train_other_are_rejected(self):
        source = self.make_subset("train-clean-100", [("1-2-0000", "CH AE1")])
        (source.parent / "phone69").write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "exactly one source"):
            self.extract()
        with self.assertRaisesRegex(ValueError, "Unsupported Phone69 subset.*train-other-500"):
            self.extract(("train-other-500",))
        self.assertFalse(self.outputs())


if __name__ == "__main__":
    unittest.main()
