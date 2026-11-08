"""Verify batched execution preserves multi-stage compression semantics."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from conformer_rnnt.compression.stages import CompressionSchedule, placement_options_from_config
from conformer_rnnt.model import conformer_rnnt_model_from_config


def staged_config(layers, backend=None):
    compression = {
        "stages": [
            {"after_layer": layer, "detector": {"type": "cosine_diff"},
             "mixer": {"type": "learned"}, "window_size": 5,
             "weight_predictor_mode": "five_frames"}
            for layer in layers
        ]
    }
    if backend is not None:
        compression["execution_backend"] = backend
    return {
        "tokenizer": {"type": "phone69"},
        "model": {
            "sequence_mode": "offline", "input_dim": 8, "encoding_dim": 16,
            "time_reduction_input_dim": 8, "time_reduction_stride": 2,
            "num_heads": 2, "ffn_dim": 32, "num_layers": 4,
            "dropout": 0.0, "depthwise_conv_kernel_size": 3,
            "compression": compression,
            "predictor": {"symbol_embedding_dim": 8, "num_lstm_layers": 1,
                          "lstm_hidden_dim": 8, "lstm_dropout": 0.0},
        },
    }


def local_devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class StagedBatchedCosineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def assert_stage_traces_equal(self, reference, batched):
        self.assertEqual(reference.compression.last_indices,
                         batched.compression.last_indices)
        self.assertEqual(reference.compression.last_lengths,
                         batched.compression.last_lengths)

    def assert_gradients_close(self, reference, batched, atol=1e-5, max_relative_l2=None):
        before = dict(reference.named_parameters())
        after = dict(batched.named_parameters())
        self.assertEqual(list(before), list(after))
        for name, parameter in before.items():
            self.assertEqual(parameter.grad is None, after[name].grad is None, name)
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertTrue(torch.isfinite(after[name].grad).all(), name)
                torch.testing.assert_close(
                    after[name].grad, parameter.grad, rtol=5e-4, atol=atol,
                    msg=lambda message, name=name: name + ": " + message,
                )
                if max_relative_l2 is not None:
                    difference = after[name].grad.double() - parameter.grad.double()
                    norm = torch.linalg.vector_norm(parameter.grad.double())
                    if float(norm) == 0:
                        self.assertLessEqual(float(difference.abs().max()), atol, name)
                    else:
                        relative_l2 = float(torch.linalg.vector_norm(difference) / norm)
                        self.assertLessEqual(relative_l2, max_relative_l2, name)

    def test_default_and_explicit_backends_preserve_model_state_and_definition(self):
        config = staged_config([0, 2, 3])
        torch.manual_seed(614)
        implicit = conformer_rnnt_model_from_config(config)
        torch.manual_seed(614)
        explicit = conformer_rnnt_model_from_config(staged_config([0, 2, 3], "reference"))
        fast = conformer_rnnt_model_from_config(staged_config([0, 2, 3], "batched"))
        for model, backend in ((implicit, "reference"), (explicit, "reference"),
                               (fast, "batched")):
            schedule = model.transcriber.compression
            self.assertEqual(schedule.execution_backend, backend)
            self.assertEqual([stage.execution_backend for stage in schedule.stages],
                             [backend] * 3)
            self.assertEqual([stage["feature_dim"] for stage in schedule.definitions],
                             [8, 16, 16])
            self.assertEqual(model.transcriber.placement_definition,
                             implicit.transcriber.placement_definition)
        self.assertEqual(list(implicit.state_dict()), list(fast.state_dict()))
        for name, value in implicit.state_dict().items():
            torch.testing.assert_close(explicit.state_dict()[name], value, rtol=0, atol=0)
        fast.load_state_dict(implicit.state_dict(), strict=True)
        with self.assertRaises(AttributeError):
            fast.transcriber.compression.execution_backend = "reference"
        self.assertEqual(fast.transcriber.compression.execution_backend, "batched")
        self.assertEqual([stage.execution_backend for stage in fast.transcriber.compression.stages],
                         ["batched"] * 3)

    def test_backend_schema_rejects_ambiguous_and_unsupported_combinations(self):
        config = staged_config([0, 2], "batched")
        options = placement_options_from_config(config["model"])
        self.assertEqual([stage["after_layer"] for stage in options["compression_stages"]],
                         [0, 2])
        for backend in ("unknown", None, [], {}):
            wrong = deepcopy(config)
            wrong["model"]["compression"]["execution_backend"] = backend
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                conformer_rnnt_model_from_config(wrong)
        for extra in ({"filter": "none"}, {"keep_rate": 0.5},
                      {"detector": {"type": "cosine_diff"}}):
            wrong = deepcopy(config)
            wrong["model"]["compression"].update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                conformer_rnnt_model_from_config(wrong)
        wrong = deepcopy(config)
        wrong["model"]["compression"]["stages"][0]["execution_backend"] = "batched"
        with self.assertRaises(ValueError):
            conformer_rnnt_model_from_config(wrong)
        for radius in (0, 1):
            wrong = deepcopy(config)
            wrong["model"]["compression"]["stages"][0] = {
                "after_layer": 0, "mixer": {"type": "hard"},
                "neighborhood_radius": radius,
            }
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                conformer_rnnt_model_from_config(wrong)
            wrong["model"]["compression"]["execution_backend"] = "reference"
            conformer_rnnt_model_from_config(wrong)

    def test_two_and_three_stage_indices_outputs_and_gradients_match(self):
        for layers in ([0, 2], [0, 2, 3]):
            for device in local_devices():
                with self.subTest(layers=layers, device=device):
                    torch.manual_seed(718)
                    reference = conformer_rnnt_model_from_config(
                        staged_config(layers, "reference")).to(device).eval()
                    batched = conformer_rnnt_model_from_config(
                        staged_config(layers, "batched")).to(device).eval()
                    batched.load_state_dict(reference.state_dict(), strict=True)
                    reference.transcriber.compression.record_indices = True
                    batched.transcriber.compression.record_indices = True
                    frames = torch.randn(3, 129, 8, device=device, requires_grad=True)
                    copy = frames.detach().clone().requires_grad_(True)
                    lengths = torch.tensor([129, 113, 77], device=device, dtype=torch.int32)
                    expected, expected_lengths = reference.transcribe(frames, lengths)
                    actual, actual_lengths = batched.transcribe(copy, lengths)
                    self.assert_stage_traces_equal(reference.transcriber, batched.transcriber)
                    torch.testing.assert_close(actual_lengths, expected_lengths, rtol=0, atol=0)
                    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
                    probe = torch.randn_like(expected)
                    (expected * probe).sum().backward()
                    (actual * probe).sum().backward()
                    torch.testing.assert_close(copy.grad, frames.grad, rtol=5e-4, atol=1e-5)
                    self.assert_gradients_close(reference.transcriber, batched.transcriber)
                    for stage in batched.transcriber.compression.stages:
                        weight_gradient = stage.mixer.weight_predictor[0].weight.grad
                        self.assertIsNotNone(weight_gradient)
                        self.assertGreater(float(weight_gradient.abs().sum()), 0)

    def test_production_triple_recipe_preserves_indices_outputs_and_gradients(self):
        repo = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((repo / (
            "configs/server_phone69_460h_placement_mel_layer5_layer8_"
            "stride1_s42_b11050_e50.yaml")).read_text())
        fast_config = deepcopy(config)
        fast_config["model"]["compression"]["execution_backend"] = "batched"
        for device in local_devices():
            with self.subTest(device=device):
                torch.manual_seed(718)
                reference = conformer_rnnt_model_from_config(config).to(device).eval()
                batched = conformer_rnnt_model_from_config(fast_config).to(device).eval()
                batched.load_state_dict(reference.state_dict(), strict=True)
                self.assertEqual(sum(parameter.numel() for parameter in batched.parameters()),
                                 4269973)
                self.assertEqual(batched.transcriber.block_sizes, [5, 3, 2])
                self.assertEqual([stage["feature_dim"]
                                  for stage in batched.transcriber.compression.definitions],
                                 [80, 128, 128])
                for model in (reference, batched):
                    model.transcriber.compression.record_indices = True
                frames = torch.randn(2, 257, 80, device=device, requires_grad=True)
                copy = frames.detach().clone().requires_grad_(True)
                lengths = torch.tensor([257, 223], device=device, dtype=torch.int32)
                expected, expected_lengths = reference.transcribe(frames, lengths)
                actual, actual_lengths = batched.transcribe(copy, lengths)
                self.assert_stage_traces_equal(reference.transcriber, batched.transcriber)
                torch.testing.assert_close(actual_lengths, expected_lengths, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
                probe = torch.randn_like(expected)
                (expected * probe).sum().backward()
                (actual * probe).sum().backward()
                torch.testing.assert_close(copy.grad, frames.grad, rtol=5e-4, atol=1e-5)
                # Deep fp32 GEMM regrouping can change cancellation near zero.
                # Retain the rtol and exact discrete checks, with a small
                # production-only absolute floor plus a tensor norm safeguard.
                self.assert_gradients_close(reference.transcriber, batched.transcriber,
                                            atol=2e-5, max_relative_l2=1e-5)

    def test_padding_cannot_change_any_batched_stage(self):
        for device in local_devices():
            with self.subTest(device=device):
                torch.manual_seed(91)
                model = conformer_rnnt_model_from_config(
                    staged_config([0, 2, 3], "batched")).to(device).eval()
                model.transcriber.compression.record_indices = True
                frames = torch.randn(3, 129, 8, device=device)
                lengths = torch.tensor([129, 113, 77], device=device, dtype=torch.int32)
                expected, expected_lengths = model.transcribe(frames, lengths)
                indices = deepcopy(model.transcriber.compression.last_indices)
                trace = deepcopy(model.transcriber.compression.last_lengths)
                changed = frames.clone()
                changed[1, 113:] = 1e6
                changed[2, 77:] = -1e6
                actual, actual_lengths = model.transcribe(changed, lengths)
                self.assertEqual(model.transcriber.compression.last_indices, indices)
                self.assertEqual(model.transcriber.compression.last_lengths, trace)
                torch.testing.assert_close(actual_lengths, expected_lengths, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_short_empty_and_plateau_sequences_survive_a_cascade(self):
        stages = staged_config([0, 1, 2])["model"]["compression"]["stages"]
        for device in local_devices():
            with self.subTest(device=device):
                torch.manual_seed(141)
                reference = CompressionSchedule(stages, 4, 8, 8).to(device)
                batched = CompressionSchedule(
                    stages, 4, 8, 8, execution_backend="batched").to(device)
                batched.load_state_dict(reference.state_dict(), strict=True)
                frames = torch.ones(5, 12, 8, device=device)
                frames[0] = 1e6
                # Exact cosine-distance plateaus, avoiding a near-tie oracle.
                frames[4] = 0
                frames[4, :10, :2] = frames.new_tensor([
                    [1, 0], [1, 0], [0, 1], [-1, 0], [-1, 0],
                    [0, 1], [1, 0], [1, 0], [0, -1], [-1, 0],
                ])
                lengths = torch.tensor([0, 1, 2, 12, 10], device=device, dtype=torch.int32)
                expected, actual = frames, frames.clone()
                expected_lengths, actual_lengths = lengths, lengths.clone()
                for schedule in (reference, batched):
                    schedule.record_indices = True
                    schedule.start_batch(lengths)
                for layer in (0, 1, 2):
                    expected, expected_lengths = reference.at_layer(
                        layer, expected, expected_lengths)
                    actual, actual_lengths = batched.at_layer(layer, actual, actual_lengths)
                    self.assertEqual(reference.last_indices, batched.last_indices)
                    self.assertEqual(reference.last_lengths, batched.last_lengths)
                    torch.testing.assert_close(actual_lengths, expected_lengths, rtol=0, atol=0)
                    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
                    self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(actual_lengths[:4].tolist(), [0, 1, 1, 1])
                torch.testing.assert_close(actual[1, 0], frames[1, 0], rtol=0, atol=0)

    def test_late_singleton_stage_does_not_advance_warmed_adam_state(self):
        stages = staged_config([0, 1])["model"]["compression"]["stages"]
        for device in local_devices():
            with self.subTest(device=device):
                torch.manual_seed(813)
                reference = CompressionSchedule(stages, 4, 8, 8).to(device)
                batched = CompressionSchedule(
                    stages, 4, 8, 8, execution_backend="batched").to(device)
                reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
                batched_optimizer = torch.optim.Adam(batched.parameters(), lr=0.01)
                # Warm both stage mixers through the reference, then copy the
                # identical parameters and nonzero Adam moments to the backend.
                warmup = 0
                for stage in reference.stages:
                    frames = torch.randn(2, 25, 8, device=device)
                    output, _ = stage(frames, torch.tensor([25, 21], device=device))
                    warmup = warmup + (output * torch.randn_like(output)).sum()
                warmup.backward()
                reference_optimizer.step()
                reference_optimizer.zero_grad(set_to_none=True)
                batched.load_state_dict(reference.state_dict(), strict=True)
                batched_optimizer.load_state_dict(deepcopy(reference_optimizer.state_dict()))
                before_parameters = {
                    name: parameter.detach().clone()
                    for name, parameter in reference.stages[1].named_parameters()
                }
                before_states = {
                    name: deepcopy(reference_optimizer.state[parameter])
                    for name, parameter in reference.stages[1].named_parameters()
                }
                self.assertTrue(any(float(state["exp_avg"].abs().sum()) > 0
                                    for state in before_states.values()))
                for schedule, optimizer in ((reference, reference_optimizer),
                                            (batched, batched_optimizer)):
                    schedule.record_indices = True
                    lengths = torch.tensor([12, 2, 1, 0], device=device, dtype=torch.int32)
                    frames = torch.ones(4, 12, 8, device=device, requires_grad=True)
                    schedule.start_batch(lengths)
                    output, lengths = schedule.at_layer(0, frames, lengths)
                    self.assertEqual(lengths.tolist(), [1, 1, 1, 0])
                    output, _ = schedule.at_layer(1, output, lengths)
                    output.sum().backward()
                    for parameter in schedule.stages[1].parameters():
                        self.assertIsNone(parameter.grad)
                    optimizer.step()
                    for name, parameter in schedule.stages[1].named_parameters():
                        torch.testing.assert_close(parameter, before_parameters[name], rtol=0, atol=0)
                        for key, value in before_states[name].items():
                            torch.testing.assert_close(optimizer.state[parameter][key], value,
                                                       rtol=0, atol=0)
                self.assertEqual(reference.last_indices, batched.last_indices)
                self.assertEqual(reference.last_lengths, batched.last_lengths)

    def test_lightning_rnnt_and_cross_backend_checkpoint_compatibility(self):
        try:
            from common import Batch
            from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
        except ModuleNotFoundError:
            self.skipTest("Lightning training runtime required")
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory) / "stats.json"
            stats.write_text(json.dumps({"mean": [0.] * 8, "invstddev": [1.] * 8}))
            config = staged_config([0, 2, 3], "reference")
            config.update(paths={"dataset_path": directory, "global_stats_path": str(stats)},
                          data={"n_mels": 8, "num_workers": 0, "hop_length": 160})
            for device in local_devices():
                with self.subTest(device=device):
                    torch.manual_seed(718)
                    reference = LibriSpeechRNNTModule.from_config(config).to(device).eval()
                    fast_config = deepcopy(config)
                    fast_config["model"]["compression"]["execution_backend"] = "batched"
                    batched = LibriSpeechRNNTModule.from_config(fast_config).to(device).eval()
                    batched.load_state_dict(reference.state_dict(), strict=True)
                    self.assertEqual(reference.offline_native_definition,
                                     batched.offline_native_definition)
                    self.assertEqual(batched.hparams.compression_execution_backend, "batched")
                    for model in (reference, batched):
                        model.log = lambda *args, **kwargs: None
                        model.model.transcriber.compression.record_indices = True
                    frames = torch.randn(2, 129, 8, device=device, requires_grad=True)
                    copy = frames.detach().clone().requires_grad_(True)
                    lengths = torch.tensor([129, 113], device=device, dtype=torch.int32)
                    targets = torch.tensor([[1, 2], [3, 69]], device=device, dtype=torch.int32)
                    target_lengths = torch.tensor([2, 1], device=device, dtype=torch.int32)
                    first = Batch(frames, lengths, targets, target_lengths)
                    second = Batch(copy, lengths, targets, target_lengths)
                    expected = reference._step(first, 0, "train")
                    actual = batched._step(second, 0, "train")
                    self.assert_stage_traces_equal(reference.model.transcriber,
                                                  batched.model.transcriber)
                    self.assertTrue(torch.isfinite(expected))
                    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
                    expected.backward()
                    actual.backward()
                    torch.testing.assert_close(copy.grad, frames.grad, rtol=5e-4, atol=1e-5)
                    self.assert_gradients_close(reference.model, batched.model)
                    batched.optimizer.step()
                    batched.warmup_lr_scheduler.step()
                    checkpoint = {
                        "state_dict": batched.state_dict(),
                        "optimizer_states": [batched.optimizer.state_dict()],
                        "lr_schedulers": [batched.warmup_lr_scheduler.state_dict()],
                    }
                    batched.on_save_checkpoint(checkpoint)
                    for restore_config in (config, fast_config):
                        restored = LibriSpeechRNNTModule.from_config(restore_config).to(device)
                        restored.on_load_checkpoint(checkpoint)
                        restored.load_state_dict(checkpoint["state_dict"], strict=True)
                        restored.optimizer.load_state_dict(checkpoint["optimizer_states"][0])
                        restored.warmup_lr_scheduler.load_state_dict(checkpoint["lr_schedulers"][0])
                        self.assertTrue(restored.optimizer.state_dict()["state"])
                        self.assertEqual(restored.warmup_lr_scheduler.last_epoch,
                                         batched.warmup_lr_scheduler.last_epoch)
                    wrong = deepcopy(checkpoint)
                    wrong["offline_native_definition"]["compression_stages"][0]["after_layer"] = 1
                    with self.assertRaisesRegex(ValueError, "offline natural"):
                        batched.on_load_checkpoint(wrong)


if __name__ == "__main__":
    unittest.main()
