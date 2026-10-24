"""Generated audio only: no corpus downloads or formal training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np
import torch
import yaml

from mandarin_syllable_tokenizer import MandarinSyllableTokenizer
from conformer_rnnt.aishell.manifest import (ManifestBatches, ManifestDataset, create_batch_plan, row_audio)
from conformer_rnnt.aishell.protocol import read_jsonl, validate_bundle
from conformer_rnnt.mandarin.preparation import prepare_mix, freeze_mix, parse_aishell3, parse_thchs30

REPO = Path(__file__).resolve().parents[1]
CONFIGS = [REPO / f"configs/server_mandarin_syllable_mix_{s}_s42_e50.yaml" for s in ("stride4_none", "stride1_cosd_learned")]


def make_wav(path, rate, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.random.default_rng(seed).integers(-2000, 2000, 6000 + seed * 37, dtype=np.int16).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed")); stream.writeframes(pcm)


def fixture(root):
    a3, th = root / "aishell3", root / "thchs30"
    for split, names in (("train", ["SSB00010001", "SSB00020001"]), ("test", ["SSB00030001"])):
        (a3 / split).mkdir(parents=True)
        (a3 / split / "content.txt").write_text("".join(name + ".wav\t中 zhong1 国 guo2 。\n" for name in names))
        for i, name in enumerate(names):
            make_wav(a3 / split / "wav" / name[:7] / (name + ".wav"), 44100, i + (1 if split == "train" else 3))
    for i, (split, uid) in enumerate((("train", "A1_0"), ("dev", "A1_1"), ("test", "A2_0")), 4):
        make_wav(th / "data" / (uid + ".wav"), 16000, i)
        (th / "data" / (uid + ".wav.trn")).write_text("中 国\nzhong1 guo2\nTHIS THIRD LINE MUST NEVER BECOME THE TARGET\n")
        (th / split).mkdir(parents=True)
        (th / split / (uid + ".wav")).symlink_to(Path("../data") / (uid + ".wav"))
    return a3, th


class MandarinSyllableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a3, self.th = fixture(self.root)

    def prepare(self, name="draft"):
        out = self.root / name
        report = prepare_mix(self.root, self.a3, self.th, out, expected_aishell3_total=3,
                             expected_thchs_counts={s: 1 for s in ("train", "dev", "test")})
        return out, report

    def bundle(self):
        draft, report = self.prepare()
        self.assertTrue(report["valid"], report["errors"])
        bundle = self.root / "frozen"
        freeze = freeze_mix(self.root, draft, bundle)
        plan = create_batch_plan(bundle, 8)
        (self.root / "plan.json").write_text(json.dumps(plan))
        return bundle, freeze, plan

    def test_original_full_syllables_and_exact_thchs_second_line(self):
        parsed = parse_aishell3("id.wav\t中 zhong1 国 guo2 。")
        self.assertEqual(parsed["syllables"], ["zhong1", "guo2"])
        self.assertEqual(parsed["excluded_punctuation"], ["。"])
        self.assertEqual(parse_thchs30(self.th / "data/A1_0.wav.trn")["syllables"], ["zhong1", "guo2"])
        with self.assertRaises(ValueError):
            parse_aishell3("id.wav\t中 zh ong1")

    def test_freeze_joint_corpora_resampling_speakers_and_dynamic_vocabulary(self):
        bundle, freeze, plan = self.bundle()
        self.assertEqual(freeze["counts"], {"train": 2, "dev": 2, "test": 2})
        self.assertEqual(plan["vocab_including_blank"], 3)
        self.assertEqual(validate_bundle(bundle), freeze)
        tokenizer = MandarinSyllableTokenizer(bundle / "syllable_vocab.json")
        self.assertEqual(tokenizer.decode(tokenizer.encode("zhong1 guo2")), "zhong1 guo2")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            tokenizer.encode("xin1")
        for split in ("train", "dev", "test"):
            dataset = ManifestDataset(self.root, bundle, split, tokenizer)
            self.assertEqual({r["corpus"] for r in dataset.rows}, {"aishell3", "thchs30"})
            for i, row in enumerate(dataset.rows):
                audio = dataset[i][0]
                self.assertEqual(audio.shape[1], row["num_samples"])
                self.assertEqual(row["duration_seconds"], row["original_num_samples"] / row["original_sample_rate"])
                self.assertEqual(row["num_frames"], row["num_samples"] // 160 + 1)

    def test_oov_missing_labels_and_public_count_mismatches_block_freeze(self):
        (self.a3 / "test/content.txt").write_text("SSB00030001.wav\t新 xin1\n")
        draft, report = self.prepare()
        self.assertFalse(report["valid"])
        self.assertTrue(any(e["kind"] == "unseen_syllables" for e in report["errors"]))
        with self.assertRaisesRegex(ValueError, "audit"):
            freeze_mix(self.root, draft, self.root / "blocked")
        report = prepare_mix(self.root, self.a3, self.th, self.root / "wrong_release")
        self.assertTrue(any(e["kind"].endswith("count_mismatch") for e in report["errors"]))

    def test_both_actual_models_use_joint_batches_and_finite_syllable_loss(self):
        from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
        bundle, freeze, plan = self.bundle()
        features = []
        for path in CONFIGS:
            config = yaml.safe_load(path.read_text())
            config["paths"].update(dataset_path=str(self.root), manifest_root=str(bundle), global_stats_path=str(bundle / "statistics.json"), batch_plan_path=str(self.root / "plan.json"))
            config["tokenizer"]["vocab_path"] = str(bundle / "syllable_vocab.json")
            config["data"].update(max_token_limit=8, num_workers=0)
            module = LibriSpeechRNNTModule.from_config(config)
            module.log = lambda *a, **kw: None
            data = ManifestBatches(ManifestDataset(self.root, bundle, "train", module.tokenizer), plan)
            batch = module._train_collate_fn(data[(0, 0)])
            features.append(batch.features)
            self.assertEqual(module.blank_idx, 2)
            self.assertEqual(module._extract_labels(data[(0, 0)])[1].tolist(), [2, 2])
            loss = module._step(batch, 0, "train")
            self.assertTrue(torch.isfinite(loss)); loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))
            checkpoint = {}; module.on_save_checkpoint(checkpoint)
            module.on_load_checkpoint(checkpoint)
            bad = copy.deepcopy(checkpoint); bad["mandarin_syllable_protocol"]["tokenizer"] = "changed"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                module.on_load_checkpoint(bad)
        torch.testing.assert_close(features[0], features[1], rtol=0, atol=0)

    def test_per_corpus_and_pooled_metrics_deciles_are_reference_based(self):
        from conformer_rnnt.mandarin.evaluation import decile_assignment, sample_metrics, validate_predictions
        bundle, _, _ = self.bundle()
        rows = list(read_jsonl(bundle / "test.jsonl"))
        bins = decile_assignment(rows)
        records = []
        for i, row in enumerate(rows):
            hyp = row["syllables"] if i else ["zhong1"]
            records.append({"utterance_id": row["utterance_id"], "corpus": row["corpus"],
                "ref": " ".join(row["syllables"]), "hyp": " ".join(hyp), "decile": bins[row["utterance_id"]],
                "duration_seconds": row["duration_seconds"], "mel_frames": row["num_frames"],
                "reference_symbols_per_second": len(row["syllables"]) / row["duration_seconds"],
                "pre_compression_frames": row["num_frames"], "post_compression_frames": 1,
                **sample_metrics(row["syllables"], hyp)})
        result = validate_predictions(rows, records)
        self.assertEqual(result["syllable_error_rate"], .25)
        self.assertEqual(set(result["by_corpus"]), {"aishell3", "thchs30"})
        self.assertEqual(result["D"], 1)
        records[0]["S"] = 2
        with self.assertRaises(ValueError):
            validate_predictions(rows, records)

    def test_no_local_data_never_creates_or_downloads_a_corpus(self):
        with self.assertRaisesRegex(FileNotFoundError, "nothing is downloaded"):
            prepare_mix(self.root, self.root / "missing", self.th, self.root / "not_created")
        self.assertFalse((self.root / "not_created").exists())

    def test_syllable_epoch_recovery_preserves_full_state_and_next_epoch(self):
        from pytorch_lightning import Trainer
        from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
        from conformer_rnnt.aishell.runtime import ImmutableCheckpoints, checkpoint_metadata
        bundle, _, _ = self.bundle()
        config = yaml.safe_load(CONFIGS[1].read_text())
        config["paths"].update(dataset_path=str(self.root), manifest_root=str(bundle),
            global_stats_path=str(bundle / "statistics.json"), batch_plan_path=str(self.root / "plan.json"))
        config["tokenizer"]["vocab_path"] = str(bundle / "syllable_vocab.json")
        config["data"].update(max_token_limit=8, num_workers=0)
        first = LibriSpeechRNNTModule.from_config(config)
        callback = ImmutableCheckpoints(self.root / "checkpoints")
        options = dict(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, val_check_interval=1)
        a = Trainer(max_epochs=1, callbacks=[callback], **options); a.fit(first)
        state = torch.load(callback.latest_boundary, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint_metadata(state, first)["completed_epochs"], 1)
        self.assertIn("mandarin_syllable_protocol", state)
        self.assertTrue(state["optimizer_states"][0]["state"])
        second = LibriSpeechRNNTModule.from_config(config)
        b = Trainer(max_epochs=2, callbacks=[ImmutableCheckpoints(self.root / "checkpoints", save_initial=False)], **options)
        b.fit(second, ckpt_path=callback.latest_boundary)
        self.assertEqual((b.current_epoch, b.global_step), (2, 2))

    def test_templates_inspect_dynamic_inventory_and_refuse_unapproved_controls(self):
        from conformer_rnnt.aishell.launch import read_configs, readiness, validate_pair
        configs = read_configs(CONFIGS)
        self.assertTrue(all(c["model"]["num_symbols"] is None for c in configs))
        self.assertFalse(readiness(configs)["training_allowed"])
        bad = copy.deepcopy(configs)
        for c in bad:
            c["model"]["compression"]["keep_rate"] = .27
        with self.assertRaises(ValueError):
            validate_pair(bad)
        bundle, freeze, plan = self.bundle()
        data = ManifestDataset(self.root, bundle, "train", MandarinSyllableTokenizer(bundle / "syllable_vocab.json"))
        bad_plan = copy.deepcopy(plan); bad_plan["vocab_including_blank"] = 218
        from conformer_rnnt.aishell.protocol import fingerprint
        bad_plan["fingerprint"] = fingerprint({k: v for k, v in bad_plan.items() if k != "fingerprint"})
        with self.assertRaises(ValueError):
            ManifestBatches(data, bad_plan)


if __name__ == "__main__":
    unittest.main()
