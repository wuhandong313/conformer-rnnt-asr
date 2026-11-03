"""Feature filtering must precede cosine computation and preserve fusion."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import find_peaks
import torch
import yaml

from conformer_rnnt.compression import DynamicCompression, NaturalPeakCompression
from conformer_rnnt.compression.detectors import CosineDiffPeaksDetector, build_detector
from conformer_rnnt.compression.feature_smoothing import smooth_feature_frames
from conformer_rnnt.model import conformer_rnnt_model_from_config


class FeatureSmoothingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    @staticmethod
    def detector_config(alpha):
        return {"type": "cosine_diff", "params": {"feature_smoothing_alpha": alpha}}

    @staticmethod
    def numpy_smooth(frames, alpha):
        output = []
        for t in range(len(frames)):
            neighbours = [(t, 1 - alpha / 2)]
            neighbours += [(j, alpha / 4) for j in (t - 1, t + 1) if 0 <= j < len(frames)]
            output.append(sum(weight * frames[j] for j, weight in neighbours)
                          / sum(weight for _, weight in neighbours))
        return np.asarray(output)

    def test_kernel_boundaries_and_feature_channels(self):
        x = torch.tensor([[0., 10.], [2., 10.], [6., 10.]], dtype=torch.float64)
        result = smooth_feature_frames(x, 1.0)
        expected = torch.tensor([[2 / 3, 10.], [2.5, 10.], [14 / 3, 10.]], dtype=x.dtype)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(smooth_feature_frames(x, 0.0), x, rtol=0, atol=0)
        # Rotating the feature basis must commute with a shared temporal kernel.
        q = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=x.dtype)
        torch.testing.assert_close(smooth_feature_frames(x @ q, 0.25),
                                   smooth_feature_frames(x, 0.25) @ q)

    def test_padding_empty_and_single_frames_are_isolated(self):
        torch.manual_seed(8)
        x = torch.randn(5, 13, 4, dtype=torch.float64, requires_grad=True)
        lengths = torch.tensor([0, 1, 2, 7, 13])
        changed = x.detach().clone()
        for row, n in enumerate(lengths.tolist()):
            changed[row, n:] = float("nan")
        result = smooth_feature_frames(x, 0.25, lengths)
        actual = smooth_feature_frames(changed, 0.25, lengths)
        torch.testing.assert_close(actual, result)
        for row, n in enumerate(lengths.tolist()):
            torch.testing.assert_close(result[row, :n], smooth_feature_frames(x[row, :n], 0.25))
            self.assertTrue((result[row, n:] == 0).all())
        torch.testing.assert_close(result[1, 0], x[1, 0])
        self.assertEqual(tuple(smooth_feature_frames(x[:, :0], 0.5).shape), (5, 0, 4))
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for row, n in enumerate(lengths.tolist()):
            self.assertTrue((x.grad[row, n:] == 0).all())

    def test_filter_runs_on_vectors_before_normalization_and_cosine(self):
        frames = torch.tensor([[1., 0.], [0., 2.], [3., 1.], [-1., 2.], [1., 3.]],
                              dtype=torch.float64)
        expected = self.numpy_smooth(frames.numpy(), 0.25)
        normalized = expected / np.linalg.norm(expected, axis=1, keepdims=True)
        distances = 1 - np.sum(normalized[:-1] * normalized[1:], axis=1)
        peaks, _ = find_peaks(distances)
        valleys, _ = find_peaks(-distances)
        centres = np.unique(np.concatenate((peaks + 1, valleys)))
        if not len(centres):
            centres = np.array([0])
        detector = build_detector(self.detector_config(0.25))
        normalize = torch.nn.functional.normalize
        with patch("conformer_rnnt.compression.detectors.torch.nn.functional.normalize",
                   wraps=normalize) as observed:
            indices, _ = detector.detect(frames, torch.zeros(4), np.zeros(4), len(frames))
        np.testing.assert_allclose(observed.call_args.args[0].numpy(), expected)
        np.testing.assert_array_equal(indices.numpy(), centres)

    def test_original_features_and_gradients_are_used_after_selection(self):
        torch.manual_seed(24)
        compression = NaturalPeakCompression(detector=self.detector_config(0.5),
                                              mixer={"type": "hard"})
        compression.record_indices = True
        x = torch.randn(1, 23, 4, requires_grad=True)
        original = x.detach().clone()
        output, lengths = compression(x, torch.tensor([23]))
        selected = compression.last_indices["centres"][0]
        self.assertEqual(lengths.tolist(), [len(selected)])
        torch.testing.assert_close(output[0], original[0, selected], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in compression.parameters()), 0)
        output.sum().backward()
        expected_grad = torch.zeros_like(x)
        expected_grad[0, selected] = 1
        torch.testing.assert_close(x.grad, expected_grad, rtol=0, atol=0)
        torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)

    def test_legacy_right_context_is_not_filtered_or_selected(self):
        torch.manual_seed(42)
        compression = DynamicCompression(
            right_context_length=3, detector=self.detector_config(0.5),
            mixer={"type": "hard"}, feature_dim=4,
        )
        x = torch.randn(1, 15, 4)
        first, lengths = compression(x, torch.tensor([15]))
        changed = x.clone()
        changed[:, 12:] += 1000
        second, lengths2 = compression(changed, torch.tensor([15]))
        self.assertEqual(lengths.tolist(), lengths2.tolist())
        torch.testing.assert_close(first[:, :-3], second[:, :-3], rtol=0, atol=0)
        torch.testing.assert_close(first[:, -3:], x[:, -3:], rtol=0, atol=0)
        torch.testing.assert_close(second[:, -3:], changed[:, -3:], rtol=0, atol=0)

    def test_reference_and_batched_indices_outputs_and_gradients(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for mode, window in (("five_frames", 5), ("nine_frames", 9)):
                for alpha in (0.0, 0.25, 0.5):
                    with self.subTest(device=device, mode=mode, alpha=alpha):
                        torch.manual_seed(314)
                        module = NaturalPeakCompression(
                            detector=self.detector_config(alpha), mixer={"type": "learned"},
                            feature_dim=4, window_size=window, weight_predictor_mode=mode,
                        ).to(device)
                        module.record_indices = True
                        lengths = torch.tensor([0, 1, 2, 3, 4, 11, 23], device=device)
                        x = torch.randn(7, 23, 4, device=device)
                        for row, n in enumerate(lengths.tolist()):
                            x[row, n:] = 9999
                        original = x.clone().requires_grad_(True)
                        expected, expected_lengths = module(original, lengths)
                        indices = copy.deepcopy(module.last_indices)
                        expected.square().sum().backward()
                        input_grad = original.grad.clone()
                        parameter_grads = [p.grad.clone() for p in module.mixer.parameters()]
                        module.zero_grad(set_to_none=True)
                        module.execution_backend = "batched"
                        actual_input = x.clone().requires_grad_(True)
                        actual, actual_lengths = module(actual_input, lengths)
                        actual.square().sum().backward()
                        self.assertEqual(module.last_indices, indices)
                        torch.testing.assert_close(actual_lengths, expected_lengths, rtol=0, atol=0)
                        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
                        torch.testing.assert_close(actual_input.grad, input_grad, rtol=5e-4, atol=1e-6)
                        for p, grad in zip(module.mixer.parameters(), parameter_grads):
                            torch.testing.assert_close(p.grad, grad, rtol=5e-4, atol=1e-6)

    def test_default_detector_keeps_legacy_selection(self):
        torch.manual_seed(17)
        x = torch.randn(41, 8)
        f = torch.nn.functional.normalize(x, dim=-1)
        distances = (1 - (f[1:] * f[:-1]).sum(-1)).numpy()
        peaks, _ = find_peaks(distances)
        valleys, _ = find_peaks(-distances)
        expected = np.unique(np.concatenate((peaks + 1, valleys)))
        actual, _ = CosineDiffPeaksDetector().detect(x, torch.zeros(40), np.zeros(40), 41)
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_config_preserves_parameter_and_state_dict_shapes(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "configs/server_phone69_460h_offline_native_stride1_cosd_learned_feature_smooth_a025_s42_e50.yaml").read_text())
        self.assertEqual(config["model"]["num_layers_block1"], 2)
        filtered = conformer_rnnt_model_from_config(config)
        self.assertEqual(filtered.transcriber.compression.detector.feature_smoothing_alpha, 0.25)
        original_config = copy.deepcopy(config)
        original_config["model"]["compression"]["detector"] = {"type": "cosine_diff"}
        original = conformer_rnnt_model_from_config(original_config)
        self.assertEqual(sum(p.numel() for p in filtered.parameters()),
                         sum(p.numel() for p in original.parameters()))
        filtered.load_state_dict(original.state_dict(), strict=True)
        filtered.eval()
        features = torch.randn(2, 31, config["model"]["input_dim"])
        lengths = torch.tensor([31, 19])
        output, output_lengths = filtered.transcriber(features, lengths)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[-1], config["model"]["encoding_dim"])
        self.assertTrue(((output_lengths > 0) & (output_lengths <= lengths)).all())
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(filtered.transcriber.input_linear.weight.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in filtered.transcriber.compression.mixer.parameters()))
        self.assertEqual(config["paths"]["from_checkpoint"], None)
        self.assertNotEqual(config["paths"]["exp_dir"],
                            yaml.safe_load((root / "configs/server_phone69_460h_offline_native_stride1_cosd_learned_s42_e50.yaml").read_text())["paths"]["exp_dir"])

    def test_invalid_strength_and_double_filtering_are_rejected(self):
        for alpha in (-0.1, 1.1, float("nan"), float("inf"), True, "0.25"):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                CosineDiffPeaksDetector(feature_smoothing_alpha=alpha)
        with self.assertRaisesRegex(ValueError, "distance-curve smoothing"):
            CosineDiffPeaksDetector(preprocess="savgol", feature_smoothing_alpha=0.25)

    def test_checkpoint_records_filter_and_rejects_a_different_definition(self):
        from conformer_rnnt.lightning_module import LibriSpeechRNNTModule

        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory) / "stats.json"
            stats.write_text(json.dumps({"mean": [0.] * 8, "invstddev": [1.] * 8}))
            config = {
                "paths": {"dataset_path": directory, "global_stats_path": str(stats)},
                "tokenizer": {"type": "phone69"},
                "model": {
                    "sequence_mode": "offline", "input_dim": 8, "encoding_dim": 16,
                    "num_symbols": 69, "time_reduction_input_dim": 8,
                    "time_reduction_stride": 1, "num_heads": 2, "ffn_dim": 32,
                    "num_layers_block1": 2, "num_layers_block2": 1,
                    "depthwise_conv_kernel_size": 3, "dropout": 0.,
                    "compression": {
                        "detector": self.detector_config(0.25),
                        "mixer": {"type": "learned"}, "weight_predictor_mode": "five_frames",
                    },
                    "predictor": {"symbol_embedding_dim": 8, "num_lstm_layers": 1,
                                  "lstm_hidden_dim": 8},
                },
                "data": {"n_mels": 8},
            }
            filtered = LibriSpeechRNNTModule.from_config(config)
            checkpoint = {}
            filtered.on_save_checkpoint(checkpoint)
            saved = checkpoint["offline_native_definition"]["detector"]["params"]
            self.assertEqual(saved["feature_smoothing_alpha"], 0.25)
            batched_config = copy.deepcopy(config)
            batched_config["model"]["compression"]["execution_backend"] = "batched"
            batched = LibriSpeechRNNTModule.from_config(batched_config)
            batched.on_load_checkpoint(checkpoint)
            for detector in ({"type": "cosine_diff"}, self.detector_config(0.5)):
                other_config = copy.deepcopy(config)
                other_config["model"]["compression"]["detector"] = detector
                other = LibriSpeechRNNTModule.from_config(other_config)
                with self.assertRaisesRegex(ValueError, "offline natural-compression definition"):
                    other.on_load_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
