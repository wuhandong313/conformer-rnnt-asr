import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from aishell_lexicon_tokenizer import AISHELLLexiconTokenizer
from conformer_rnnt.aishell.manifest import (EpochBatchSampler, ManifestBatches, ManifestDataset,
                                           augmentation_seed, make_batches, validate_batches)
from conformer_rnnt.aishell.preparation import audit_audio
from conformer_rnnt.aishell.protocol import (FRONTEND, LEXICON_SHA256, feature_frames, file_sha,
                                           fingerprint, validate_bundle, write_json)


def tiny_bundle(root):
    """Generated PCM fixtures only: never loads or downloads corpus audio."""
    data, bundle = root / "data", root / "frozen"
    bundle.mkdir()
    counts = {"train": 2, "dev": 1, "test": 1}
    tokenizer = AISHELLLexiconTokenizer()
    for split, number in counts.items():
        rows = []
        for i in range(number):
            uid = split + str(i)
            samples = 4000 + i * 320
            relative = f"wav/{split}/{split}speaker/{uid}.wav"
            path = data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pcm = (np.sin(np.arange(samples) * (0.03 + i * .01 + len(split) * .001)) * 4000).astype("<i2").tobytes()
            with wave.open(str(path), "wb") as out:
                out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                out.writeframes(pcm)
            rows.append({"utterance_id": uid, "split": split, "speaker_id": split + "speaker", "audio_path": relative,
                         "sample_rate": 16000, "num_samples": samples, "num_frames": feature_frames(samples),
                         "audio_pcm_sha256": hashlib.sha256(pcm).hexdigest(), "words": ["中国"], "text": "中国",
                         "phones": ["zh", "ong1", "g", "uo2"], "label_provenance": []})
        with (bundle / (split + ".jsonl")).open("w") as out:
            out.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_json(bundle / "statistics.json", {"mean": [0.] * 80, "invstddev": [1.] * 80})
    write_json(bundle / "label_protocol.json", {"version": 1, "lexicon_sha256": LEXICON_SHA256, "tokenizer_fingerprint": tokenizer.fingerprint,
        "selection": "exact_context_match_to_original_dictionary_candidates", "comparison_mapping_version": 1,
        "word_boundaries": "official_transcript_unchanged", "stale_resolution_ids": [],
        "preprocessing_source_sha256": {"fixture": "synthetic_test_only"},
        "predictor": {"package": "g2pM", "version": "0.1.2.5", "package_and_model_files": {"fixture": "test"}}})
    write_json(bundle / "audio_audit.json", {"valid": True, "counts": counts})
    write_json(bundle / "label_audit.json", {"valid": True, "counts": counts, "unresolved_issues": 0})
    write_json(bundle / "lexicon_audit.json", {"synthetic_fixture": True})
    freeze = {"version": 1, "status": "frozen", "counts": counts, "unresolved_issues": 0,
              "lexicon_sha256": LEXICON_SHA256, "tokenizer_fingerprint": tokenizer.fingerprint,
              "frontend": FRONTEND, "files": {p.name: file_sha(p) for p in bundle.iterdir()}}
    freeze["fingerprint"] = fingerprint(freeze)
    write_json(bundle / "freeze.json", freeze)
    plan = {"version": 1, "bundle_fingerprint": freeze["fingerprint"], "seed": 42, "token_budget": 8,
            "vocab_including_blank": 218, "numel_limit": 1900000000,
            "batches": {s: make_batches([json.loads(line) for line in (bundle / (s + ".jsonl")).read_text().splitlines()], 8)
                        for s in ("train", "dev")}}
    plan["fingerprint"] = fingerprint(plan)
    write_json(root / "batches.json", plan)
    return data, bundle, freeze, plan


class AISHELLManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.bundle, self.freeze, self.plan = tiny_bundle(self.root)

    def test_freeze_checks_payload_hash_and_official_counts(self):
        self.assertEqual(validate_bundle(self.bundle, expected_counts=self.freeze["counts"]), self.freeze)
        with self.assertRaisesRegex(ValueError, "official"):
            validate_bundle(self.bundle)
        with (self.bundle / "train.jsonl").open("a") as out:
            out.write("{}\n")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            validate_bundle(self.bundle, expected_counts=self.freeze["counts"])

    def test_audio_loading_and_tamper_detection(self):
        dataset = ManifestDataset(self.data, self.bundle, "train", AISHELLLexiconTokenizer(), freeze=self.freeze)
        waveform, rate, labels, speaker, split, uid = dataset[0]
        self.assertEqual((waveform.shape[1], rate, labels, split), (4000, 16000, "zh ong1 g uo2", "train"))
        path = self.data / dataset.rows[0]["audio_path"]
        contents = bytearray(path.read_bytes()); contents[-1] ^= 1; path.write_bytes(contents)
        with self.assertRaisesRegex(ValueError, "Audio differs"):
            dataset[0]

    def test_common_batches_cover_all_ids_and_real_frames_guard(self):
        rows = [{"utterance_id": str(i), "phones": ["a1"] * u, "num_frames": t}
                for i, (u, t) in enumerate([(4, 50), (4, 1000), (2, 75)])]
        plan = make_batches(rows, 8, vocab=218, numel_limit=2000000)
        self.assertEqual(sorted(uid for b in plan for uid in b), ["0", "1", "2"])
        for bad in ([["0", "1", "2"]], [["0"], ["0", "1", "2"]], [["0"], ["1"]]):
            with self.assertRaises(ValueError):
                validate_batches(rows, bad, 8, 218, 2000000)
        with self.assertRaisesRegex(ValueError, "never dropped"):
            make_batches([{"utterance_id": "long", "phones": ["a1"], "num_frames": 100000}], 8, 218, 10)

    def test_sampler_carries_epoch_into_worker_context(self):
        sampler = EpochBatchSampler(20)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)
        self.assertTrue(all(epoch == 1 for epoch, _ in second))
        sampler.set_epoch(0)
        self.assertEqual(list(sampler), first)
        self.assertNotEqual(augmentation_seed(42, {"epoch": 0}), augmentation_seed(42, {"epoch": 1}))

    def test_manifest_batches_verify_every_frozen_sample(self):
        base = ManifestDataset(self.data, self.bundle, "train", AISHELLLexiconTokenizer(), freeze=self.freeze)
        batches = ManifestBatches(base, self.plan)
        samples = batches[(3, 0)]
        self.assertEqual(samples[0][6]["epoch"], 3)
        self.assertTrue(all(sample[6] == samples[0][6] for sample in samples))

    def test_feature_formula_matches_the_centered_frontend(self):
        import torch
        import torchaudio
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
        for length in (4000, 4001, 4160, 4320):
            self.assertEqual(mel(torch.zeros(length)).shape[-1], feature_frames(length))

    def test_audio_audit_preserves_official_word_boundaries(self):
        transcript = self.data / "transcript/aishell_transcript_v0.8.txt"
        transcript.parent.mkdir()
        transcript.write_text("train0 中国\ntrain1 中国\ndev0 中国\ntest0 中国\n", encoding="utf-8")
        report = audit_audio(self.data, self.root / "audio_audit", expected_counts=self.freeze["counts"])
        # Generated fixture waveforms can coincide across splits: detect this, never silently deduplicate.
        duplicate = any(error["kind"] == "identical_audio_across_splits" for error in report["errors"])
        self.assertEqual(report["valid"], not duplicate)
        self.assertEqual(report["counts"], self.freeze["counts"])


if __name__ == "__main__":
    unittest.main()
