"""Generated WAV fixtures exercise the real model and complete Lightning state."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from pytorch_lightning import Trainer

from conformer_rnnt.aishell.launch import read_configs, readiness, validate_pair
from conformer_rnnt.aishell.manifest import ManifestBatches, ManifestDataset
from conformer_rnnt.aishell.runtime import ImmutableCheckpoints, checkpoint_metadata
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
from scripts.server.probe_aishell1 import checkpoint_roundtrip
from test_aishell_manifest import tiny_bundle

REPO = Path(__file__).resolve().parents[1]
CONFIGS = [REPO / f"configs/作废_aishell1_lexicon_20261024/server_aishell1_lexicon_{s}_s42_e50.yaml"
           for s in ("stride4_none", "stride1_cosd_learned")]


class AISHELLRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.bundle, self.freeze, self.plan = tiny_bundle(self.root)
        self.counts = patch("conformer_rnnt.aishell.protocol.OFFICIAL_COUNTS", self.freeze["counts"])
        self.counts.start(); self.addCleanup(self.counts.stop)
        self.configs = [yaml.safe_load(p.read_text()) for p in CONFIGS]
        for config in self.configs:
            config["paths"].update(dataset_path=str(self.data), manifest_root=str(self.bundle),
                                   global_stats_path=str(self.bundle / "statistics.json"),
                                   batch_plan_path=str(self.root / "batches.json"))
            config["data"].update(max_token_limit=8, num_workers=0)
        self.config = self.configs[1]

    def module(self, config=None):
        result = LibriSpeechRNNTModule.from_config(config or self.config)
        result.log = lambda *a, **kw: None
        return result

    def samples(self, module):
        data = ManifestDataset(self.data, self.bundle, "train", module.tokenizer, freeze=self.freeze)
        return ManifestBatches(data, self.plan)[(0, 0)]

    def test_actual_218_class_model_finite_loss_gradient_and_blank_padding(self):
        for config in self.configs:
            module = self.module(config)
            samples = self.samples(module)
            batch = module._train_collate_fn(samples)
            self.assertEqual(module.blank_idx, 217)
            self.assertEqual(module.tokenizer.get_piece_size(), 217)
            module.train()
            loss = module._step(batch, 0, "train")
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
            if config is self.config:
                self.assertIsNotNone(module.model.transcriber.compression.mixer.weight_predictor[0].weight.grad)

    def test_pair_augmentation_independent_of_model_rng_and_worker_epoch(self):
        first = self.module(self.configs[0])
        samples = self.samples(first)
        state = torch.random.get_rng_state().clone()
        a = first._train_collate_fn(samples)
        torch.testing.assert_close(state, torch.random.get_rng_state())
        torch.rand(1234)
        second = self.module()
        b = second._train_collate_fn(samples)
        torch.testing.assert_close(a.features, b.features, rtol=0, atol=0)
        # The sampler reads current_epoch at iterator creation, before worker prefetch.
        loader = second.train_dataloader()
        self.assertEqual(next(iter(loader.sampler))[0], 0)
        class Epoch:
            current_epoch = 7
        loader.sampler.epoch_getter = lambda: Epoch.current_epoch
        self.assertEqual(next(iter(loader.sampler))[0], 7)
        samples[0][6]["epoch"] = 1
        c = second._train_collate_fn(samples)
        self.assertFalse(torch.equal(b.features, c.features))

    def test_beam20_decoder_uses_blank217_and_exposes_real_compression_lengths(self):
        for config in self.configs:
            module = self.module(config).eval()
            batch = module._valid_collate_fn(self.samples(module)[:1])
            with torch.no_grad():
                module.model.joiner.linear.weight.zero_()
                module.model.joiner.linear.bias.fill_(-100.)
                module.model.joiner.linear.bias[217] = 100.
            seen = []
            hook = module.model.transcriber.compression.register_forward_hook(
                lambda _m, args, output: seen.append((args[1].tolist(), output[1].tolist())))
            with torch.inference_mode():
                self.assertEqual(module(batch), "")
            hook.remove()
            self.assertEqual(len(seen), 1)
            self.assertLessEqual(seen[0][1][0], seen[0][0][0])

    def test_checkpoint_rejects_label_vocab_statistics_model_and_budget_mismatch(self):
        module = self.module()
        checkpoint = {}
        module.on_save_checkpoint(checkpoint)
        module.on_load_checkpoint(checkpoint)
        for key in ("bundle", "tokenizer", "statistics", "batch_plan"):
            corrupt = copy.deepcopy(checkpoint)
            corrupt["aishell_lexicon_protocol"][key] = "changed"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                module.on_load_checkpoint(corrupt)
        bad = copy.deepcopy(checkpoint); bad["offline_native_definition"]["stride"] = 4
        with self.assertRaisesRegex(ValueError, "definition"):
            module.on_load_checkpoint(bad)

    def test_full_epoch_checkpoint_restores_optimizer_scheduler_and_next_epoch(self):
        # Real RNNT model on tiny generated audio; two epochs only inside this unit test.
        module = LibriSpeechRNNTModule.from_config(self.config)
        snapshots = ImmutableCheckpoints(self.root / "checkpoints", interval=1)
        options = dict(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
        first = Trainer(max_epochs=1, callbacks=[snapshots], val_check_interval=1, **options)
        first.fit(module)
        checkpoint = torch.load(snapshots.latest_boundary, map_location="cpu", weights_only=False)
        info = checkpoint_metadata(checkpoint, module)
        self.assertEqual((info["completed_epochs"], info["global_step"]), (1, 1))
        self.assertTrue(checkpoint["optimizer_states"][0]["state"])
        self.assertEqual(checkpoint["lr_schedulers"][0]["last_epoch"], 1)
        old_files = {p: p.read_bytes() for p in (self.root / "checkpoints").glob("*.ckpt")}
        restored = LibriSpeechRNNTModule.from_config(self.config)
        second_callback = ImmutableCheckpoints(self.root / "checkpoints", save_initial=False, interval=1)
        second = Trainer(max_epochs=2, callbacks=[second_callback], val_check_interval=1, **options)
        second.fit(restored, ckpt_path=snapshots.latest_boundary)
        self.assertEqual((second.current_epoch, second.global_step), (2, 2))
        self.assertTrue(all(p.read_bytes() == content for p, content in old_files.items()))
        self.assertEqual(int(next(iter(second.optimizers[0].state.values()))["step"]), 2)

    def test_initial_roundtrip_has_zero_updates_and_full_states(self):
        from types import SimpleNamespace
        # A CPU fixture must not initialize any GPU even on an occupied CUDA server.
        saved = {}
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.get_rng_state", side_effect=AssertionError("CPU check touched GPU")):
            ImmutableCheckpoints(self.root / "unused").on_save_checkpoint(
                None, SimpleNamespace(device=torch.device("cpu")), saved)
        self.assertIsNone(saved["aishell_rng_state"]["torch_cuda"])
        directory = self.root / "preflight"
        directory.mkdir()
        records = checkpoint_roundtrip(self.config, directory, accelerator="cpu")
        self.assertEqual([r["global_step"] for r in records], [0, 0])
        self.assertEqual([r["completed_epochs"] for r in records], [0, 0])

    def test_best_checkpoint_uses_utterance_weighted_validation_loss(self):
        from types import MethodType
        from torch.utils.data import DataLoader
        module = LibriSpeechRNNTModule.from_config(self.config)

        def validation_step(model, batch, batch_idx):
            model.log("Losses/val_loss", batch.mean(), on_step=True, on_epoch=True, batch_size=batch.numel())

        module.validation_step = MethodType(validation_step, module)
        callback = ImmutableCheckpoints(self.root / "weighted_checkpoints")
        trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                          max_epochs=1, callbacks=[callback])
        trainer.fit(module, val_dataloaders=DataLoader([torch.tensor([8.]), torch.zeros(3)], batch_size=None))
        self.assertEqual(callback.best_score, 2.)

    def test_pair_protocol_rejects_shared_unapproved_settings_and_missing_data(self):
        configs = read_configs(CONFIGS)
        self.assertFalse(readiness(configs)["training_allowed"])
        for group, key, value in (("training", "adam_eps", .01), ("model", "keep_rate", .27),
                                  ("data", "hop_length", 80), ("trainer", "precision", "16-mixed")):
            modified = copy.deepcopy(configs)
            for config in modified:
                config[group][key] = value
            with self.assertRaises(ValueError):
                validate_pair(modified)
        result = subprocess.run([sys.executable, REPO / "scripts/server/train_aishell1_one.py", CONFIGS[0],
                                 "--gate", "/not/present", "--execute"], cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Training disabled", result.stderr)
        from conformer_rnnt.model import conformer_rnnt_model_from_config
        mismatch = copy.deepcopy(configs[0]); mismatch["model"]["num_symbols"] = 69
        with self.assertRaisesRegex(ValueError, "inventory"):
            conformer_rnnt_model_from_config(mismatch)


if __name__ == "__main__":
    unittest.main()
