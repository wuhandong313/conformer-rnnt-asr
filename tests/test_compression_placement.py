"""Verify global stage positions, neighbourhood semantics and model integration."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import torch
import yaml

from conformer_rnnt.compression import NaturalPeakCompression
from conformer_rnnt.compression.stages import normalize_compression_stages, placement_options_from_config
from conformer_rnnt.model import conformer_rnnt_model_from_config
from conformer_rnnt.staged_encoder import StagedConformerEncoder
from conformer_rnnt.checkpoint_utils import phone69_resume_metadata, select_phone69_resume_checkpoint

REPO = Path(__file__).resolve().parents[1]


def learned(layer):
    return {"after_layer": layer, "mixer": {"type": "learned"}, "weight_predictor_mode": "five_frames"}


def neighbourhood(layer=2):
    return {"after_layer": layer, "mixer": {"type": "hard"}, "neighborhood_radius": 1}


def small_config(stages):
    return {
        "tokenizer": {"type": "phone69"},
        "model": {
            "sequence_mode": "offline", "input_dim": 8, "encoding_dim": 16,
            "time_reduction_input_dim": 8, "time_reduction_stride": 2,
            "num_heads": 2, "ffn_dim": 32, "num_layers": 10,
            "dropout": 0.0, "depthwise_conv_kernel_size": 3,
            "compression": {"stages": stages},
            "predictor": {"symbol_embedding_dim": 8, "num_lstm_layers": 1,
                          "lstm_hidden_dim": 8, "lstm_dropout": 0.0},
        },
    }


class PlacementSchemaTests(unittest.TestCase):
    def test_factories_reject_ambiguous_or_obsolete_stage_settings(self):
        wrong = (
            [learned(5), learned(2)], [learned(5), learned(5)], [learned(10)],
            [learned(-1)], [{**learned(2), "keep_rate": None}],
            [{**learned(2), "right_context_length": 0}],
            [{**learned(2), "neighborhood_radius": 1}],
            [{**learned(2), "weight_predictor_mode": "diff5"}],
            [{**learned(2), "window_size": 3}], [],
            [{**neighbourhood(), "window_size": 5}],
        )
        for stages in wrong:
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                normalize_compression_stages(stages, 10)
        for extra in ({"num_layers_block1": 2}, {"right_context_length": 0},
                      {"left_context_length": 0}, {"sequence_mode": "legacy"}):
            config = {**small_config([learned(5)])["model"], **extra}
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                placement_options_from_config(config)
        config = small_config([learned(5)])["model"]
        config["compression"]["filter"] = "hanning"
        with self.assertRaises(ValueError):
            placement_options_from_config(config)

    def test_prepared_recipes_preserve_the_comparison_protocol(self):
        base = yaml.safe_load((REPO / "configs/server_phone69_460h_offline_native_stride2_cosd_learned_s42_e50.yaml").read_text())
        outputs = set()
        for name, layout in (("mel", [(0, "learned", 0)]),
                             ("layer5", [(5, "learned", 0)]),
                             ("layers2_5_n3", [(2, "hard", 1), (5, "learned", 0)])):
            path = REPO / f"configs/server_phone69_460h_placement_{name}_stride2_s42_e50.yaml"
            config = yaml.safe_load(path.read_text())
            for key in ("training", "data", "trainer", "tokenizer", "decoding", "seed"):
                self.assertEqual(config[key], base[key])
            self.assertIsNone(config["paths"]["from_checkpoint"])
            self.assertNotEqual(config["paths"]["exp_dir"], base["paths"]["exp_dir"])
            outputs.add(config["paths"]["exp_dir"])
            stages = placement_options_from_config(config["model"])["compression_stages"]
            self.assertEqual([(x["after_layer"], x["mixer"]["type"], x["neighborhood_radius"]) for x in stages], layout)
        self.assertEqual(len(outputs), 3)


class NeighbourhoodRetentionTests(unittest.TestCase):
    def test_original_frames_are_sorted_deduplicated_and_differentiable(self):
        module = NaturalPeakCompression(mixer={"type": "hard"}, neighborhood_radius=1)
        module.record_indices = True
        module.detector = Mock()
        module.detector.detect.return_value = (torch.tensor([10, 5, 3, 3]), torch.ones(4))
        frames = torch.arange(48.0).reshape(1, 12, 4).requires_grad_()
        result, lengths = module(frames, torch.tensor([12]))
        expected = [2, 3, 4, 5, 6, 9, 10, 11]
        torch.testing.assert_close(result[0], frames[0, expected])
        self.assertEqual(lengths.tolist(), [8])
        self.assertEqual(module.last_candidate_lengths, [3])
        self.assertEqual(module.last_indices, {"centres": [[3, 5, 10]], "retained": [expected]})
        self.assertEqual(list(module.parameters()), [])
        result.sum().backward()
        self.assertEqual(frames.grad[0, 4].tolist(), [1., 1., 1., 1.])
        self.assertEqual(frames.grad[0, 0].tolist(), [0., 0., 0., 0.])

    def test_endpoints_padding_and_short_fallback(self):
        module = NaturalPeakCompression(mixer={"type": "hard"}, neighborhood_radius=1)
        module.record_indices = True
        frames = torch.ones(4, 7, 4)
        frames[1, 2:] = 10000
        result, lengths = module(frames, torch.tensor([7, 2, 1, 0]))
        self.assertEqual(lengths.tolist(), [2, 2, 1, 0])
        self.assertEqual(module.last_indices["retained"], [[0, 1], [0, 1], [0], []])
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(module.last_candidate_lengths, [1, 1, 1, 0])


class EncoderPlacementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_exact_global_layer_order_and_total_depth(self):
        cases = ([learned(0)], [learned(5)], [neighbourhood(), learned(5)])
        for stages in cases:
            with self.subTest(stages=stages):
                model = conformer_rnnt_model_from_config(small_config(stages)).eval()
                encoder = model.transcriber
                events, handles = [], []
                global_layer = 0
                handles.append(encoder.input_linear.register_forward_pre_hook(lambda *args: events.append("projection")))
                for block in encoder.conformer_blocks:
                    for layer in block.conformer_layers:
                        global_layer += 1
                        handles.append(layer.register_forward_pre_hook(
                            lambda *args, layer_id=global_layer: events.append(layer_id)))
                for stage, module in zip(encoder.compression.definitions, encoder.compression.stages):
                    handles.append(module.register_forward_pre_hook(
                        lambda *args, position=stage["after_layer"]: events.append(f"C{position}")))
                result, lengths = model.transcribe(torch.randn(2, 33, 8), torch.tensor([33, 25]))
                for handle in handles:
                    handle.remove()
                expected = (["C0"] if stages[0]["after_layer"] == 0 else []) + ["projection"]
                for layer in range(1, 11):
                    expected.append(layer)
                    if any(s["after_layer"] == layer for s in stages):
                        expected.append(f"C{layer}")
                self.assertEqual(events, expected)
                self.assertEqual(global_layer, 10)
                self.assertEqual(result.shape[0::2], (2, 16))
                self.assertEqual(result.size(1), int(lengths.max()))
                self.assertTrue(torch.isfinite(result).all())

    def test_mel_dimension_and_all_learned_stages_receive_gradients(self):
        for stages in ([learned(0)], [learned(5)], [neighbourhood(), learned(5)]):
            model = conformer_rnnt_model_from_config(small_config(stages)).eval()
            frames = torch.randn(2, 35, 8, requires_grad=True)
            output, _ = model.transcribe(frames, torch.tensor([35, 29]))
            (output * torch.randn_like(output)).sum().backward()
            self.assertTrue(torch.isfinite(frames.grad).all())
            for stage, module in zip(model.transcriber.compression.definitions, model.transcriber.compression.stages):
                if stage["mixer"]["type"] == "learned":
                    network = module.mixer.weight_predictor
                    self.assertEqual(network[0].in_features, 40 if stage["after_layer"] == 0 else 80)
                    self.assertIsNotNone(network[0].weight.grad)
                    self.assertTrue(torch.isfinite(network[0].weight.grad).all())
                    self.assertGreater(float(network[0].weight.grad.abs().sum()), 0)

    def test_second_stage_redetects_the_shortened_sequence(self):
        encoder = StagedConformerEncoder(
            input_dim=8, output_dim=16, time_reduction_input_dim=8,
            time_reduction_stride=2, num_heads=2, ffn_dim=32,
            num_layers=10, compression_stages=[neighbourhood(), learned(5)],
            dropout=0, depthwise_conv_kernel_size=3,
        ).eval()
        first, second = encoder.compression.stages
        first.detector = Mock()
        first.detector.detect.return_value = (torch.tensor([3, 5, 10]), torch.ones(3))
        second.detector = Mock()
        second.detector.detect.side_effect = lambda frames, diff, raw, n: (torch.tensor([n-1]), torch.ones(1))
        encoder.compression.record_indices = True
        _, lengths = encoder(torch.randn(1, 24, 8), torch.tensor([24]))
        self.assertEqual(first.detector.detect.call_args.args[3], 12)
        self.assertEqual(second.detector.detect.call_args.args[3], 8)
        self.assertEqual(lengths.tolist(), [1])
        trace = encoder.compression.last_lengths["stages"]
        self.assertEqual([(row["after_layer"], row["input"], row["output"]) for row in trace],
                         [(2, [12], [8]), (5, [8], [1])])
        self.assertEqual(encoder.compression.last_indices[1]["centres"], [[7]])

    def test_input_padding_is_not_seen_by_any_stage(self):
        for stages in ([learned(0)], [learned(5)], [neighbourhood(), learned(5)]):
            model = conformer_rnnt_model_from_config(small_config(stages)).eval()
            frames = torch.randn(2, 33, 8)
            lengths = torch.tensor([33, 25])
            before, before_lengths = model.transcribe(frames, lengths)
            changed = frames.clone()
            changed[1, 25:] = 1e6
            after, after_lengths = model.transcribe(changed, lengths)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            torch.testing.assert_close(before_lengths, after_lengths)


class PlacementCheckpointTests(unittest.TestCase):
    def test_lightning_full_checkpoint_resumes_optimizer_scheduler_and_loops(self):
        try:
            from pytorch_lightning import Trainer
            from torch.utils.data import DataLoader
            from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
            from common import Batch
        except ModuleNotFoundError:
            self.skipTest("Lightning training runtime required")
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory)/"stats.json"
            stats.write_text(json.dumps({"mean": [0.]*8, "invstddev": [1.]*8}))
            config = small_config([neighbourhood(), learned(5)])
            config.update(paths={"dataset_path": directory, "global_stats_path": str(stats)},
                          data={"n_mels": 8, "num_workers": 0, "hop_length": 160})
            module = LibriSpeechRNNTModule.from_config(config)
            batch = Batch(torch.randn(2, 35, 8), torch.tensor([35, 29], dtype=torch.int32),
                          torch.tensor([[1, 2, 3], [4, 5, 69]], dtype=torch.int32),
                          torch.tensor([3, 2], dtype=torch.int32))
            data = DataLoader([batch], batch_size=None)
            options = dict(accelerator="cpu", devices=1, logger=False,
                           enable_checkpointing=False, enable_progress_bar=False,
                           enable_model_summary=False, limit_val_batches=0,
                           num_sanity_val_steps=0, default_root_dir=directory)
            trainer = Trainer(max_steps=1, max_epochs=1, **options)
            trainer.fit(module, train_dataloaders=data)
            path = Path(directory)/"placement.ckpt"
            trainer.save_checkpoint(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["global_step"], 1)
            self.assertTrue(checkpoint["optimizer_states"][0]["state"])
            self.assertTrue(checkpoint["lr_schedulers"][0])
            self.assertEqual(checkpoint["offline_native_definition"]["version"], 2)
            restored = LibriSpeechRNNTModule.load_from_checkpoint(path)
            resumed = Trainer(max_steps=2, max_epochs=2, **options)
            resumed.fit(restored, train_dataloaders=data, ckpt_path=path)
            self.assertEqual(resumed.global_step, 2)
            self.assertEqual(restored.warmup_lr_scheduler.last_epoch, 2)
            self.assertTrue(all(int(state["step"]) == 2 for state in restored.optimizer.state.values()))

    def test_resume_selection_checks_stage_semantics_not_just_tensor_shapes(self):
        definition = {"version": 2, "compression_stages": [{"after_layer": 2, "neighborhood_radius": 1}]}
        checkpoint = {
            "optimizer_states": [{}], "lr_schedulers": [{}],
            "hyper_parameters": {"max_token_limit": 16000}, "global_step": 5,
            "loops": {"fit_loop": {"epoch_progress": {"current": {"ready": 1, "completed": 1}}}},
            "offline_native_definition": definition,
        }
        self.assertTrue(phone69_resume_metadata(checkpoint, 8000, definition)["resume_at_epoch_boundary"])
        with tempfile.TemporaryDirectory() as directory:
            wrong = deepcopy(checkpoint)
            wrong["offline_native_definition"]["compression_stages"][0]["neighborhood_radius"] = 2
            good_path, wrong_path = Path(directory)/"good.ckpt", Path(directory)/"wrong.ckpt"
            torch.save(checkpoint, good_path)
            torch.save(wrong, wrong_path)
            path, _ = select_phone69_resume_checkpoint([wrong_path, good_path], 16000, definition)
            self.assertEqual(path, str(good_path))
            with self.assertRaisesRegex(ValueError, "No safe resume"):
                select_phone69_resume_checkpoint([wrong_path], 16000, definition)

    def test_lightning_factories_save_and_validate_version_two_definitions(self):
        try:
            from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
            from common import Batch
        except ModuleNotFoundError:
            self.skipTest("Lightning training runtime required")
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory)/"stats.json"
            stats.write_text(json.dumps({"mean": [0.]*8, "invstddev": [1.]*8}))
            for stages in ([learned(0)], [learned(5)], [neighbourhood(), learned(5)]):
                config = small_config(stages)
                config.update(paths={"dataset_path": directory, "global_stats_path": str(stats)},
                              data={"n_mels": 8, "num_workers": 0, "hop_length": 160})
                module = LibriSpeechRNNTModule.from_config(config)
                self.assertEqual(module.offline_native_definition["version"], 2)
                self.assertEqual(module.offline_native_definition["layers"], module.model.transcriber.block_sizes)
                batch = Batch(torch.randn(2, 35, 8), torch.tensor([35, 29], dtype=torch.int32),
                              torch.tensor([[1, 2, 3], [4, 5, 69]], dtype=torch.int32),
                              torch.tensor([3, 2], dtype=torch.int32))
                module.log = lambda *args, **kwargs: None
                loss = module._step(batch, 0, "train")
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                module.optimizer.step()
                module.warmup_lr_scheduler.step()
                checkpoint = {"state_dict": module.state_dict(),
                              "optimizer_states": [module.optimizer.state_dict()],
                              "lr_schedulers": [module.warmup_lr_scheduler.state_dict()]}
                module.on_save_checkpoint(checkpoint)
                restored = LibriSpeechRNNTModule.from_config(config)
                restored.on_load_checkpoint(checkpoint)
                restored.load_state_dict(checkpoint["state_dict"], strict=True)
                restored.optimizer.load_state_dict(checkpoint["optimizer_states"][0])
                restored.warmup_lr_scheduler.load_state_dict(checkpoint["lr_schedulers"][0])
                self.assertTrue(restored.optimizer.state_dict()["state"])
                wrong = deepcopy(checkpoint)
                wrong["offline_native_definition"]["compression_stages"][0]["after_layer"] += 1
                with self.assertRaisesRegex(ValueError, "offline natural"):
                    restored.on_load_checkpoint(wrong)
                del checkpoint["offline_native_definition"]
                with self.assertRaisesRegex(ValueError, "offline natural"):
                    restored.on_load_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
