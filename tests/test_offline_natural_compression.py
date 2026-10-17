"""Natural cardinality, whole-utterance processing, and offline frame lengths."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import torch

from conformer_rnnt.compression import NaturalPeakCompression
from conformer_rnnt.encoder import ConformerEncoder, TimeReduction
from conformer_rnnt.model import sequence_options_from_config


class NaturalCompressionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    @staticmethod
    def frames(increments):
        angles = torch.tensor([0.0] + increments).cumsum(0)
        return torch.stack([angles.cos(), angles.sin()], -1)

    def test_all_peaks_and_valleys_without_quota(self):
        frames = self.frames([0.1, 1.0, 0.2, 0.8, 0.1, 1.1, 0.1])
        compression = NaturalPeakCompression(mixer={"type": "hard"})
        output, lengths = compression(frames[None], torch.tensor([8]))
        torch.testing.assert_close(output[0], frames[[2, 4, 6]])
        self.assertEqual(lengths.tolist(), [3])
        other = self.frames([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        _, lengths = compression(other[None], torch.tensor([8]))
        self.assertEqual(lengths.tolist(), [1])
        self.assertFalse(hasattr(compression, "keep_rate"))
        self.assertFalse(hasattr(compression, "right_context_length"))

    def test_tail_is_given_to_detector_and_not_appended_separately(self):
        compression = NaturalPeakCompression(mixer={"type": "hard"})
        compression.detector = Mock()
        compression.detector.detect.side_effect = lambda frames, diff, raw, n: (
            torch.tensor([n - 1]), torch.ones(1)
        )
        frames = torch.arange(40.0).reshape(2, 10, 2)
        output, lengths = compression(frames, torch.tensor([10, 7]))
        self.assertEqual(lengths.tolist(), [1, 1])
        torch.testing.assert_close(output[:, 0], torch.stack([frames[0, 9], frames[1, 6]]))
        self.assertEqual([c.args[3] for c in compression.detector.detect.call_args_list], [10, 7])

    def test_short_constant_and_padded_sequences(self):
        compression = NaturalPeakCompression(mixer={"type": "learned"}, feature_dim=8)
        frames = torch.ones(4, 8, 8, requires_grad=True)
        output, lengths = compression(frames, torch.tensor([8, 2, 1, 0]))
        self.assertEqual(lengths.tolist(), [1, 1, 1, 0])
        output.sum().backward()
        self.assertTrue(torch.isfinite(frames.grad).all())
        grad = compression.mixer.weight_predictor[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_partial_stride_preserves_final_real_frame(self):
        reduction = TimeReduction(4, pad_to_stride=True)
        frames = torch.arange(5.0).reshape(1, 5, 1)
        output, lengths = reduction(frames, torch.tensor([5]))
        self.assertEqual(lengths.tolist(), [2])
        torch.testing.assert_close(output, torch.tensor([[[0., 1., 2., 3.], [4., 0., 0., 0.]]]))
        _, old_lengths = TimeReduction(4)(frames, torch.tensor([5]))
        self.assertEqual(old_lengths.tolist(), [1])

    def test_encoder_odd_lengths_and_no_padding_signal(self):
        for stride in (1, 2, 4):
            encoder = ConformerEncoder(
                input_dim=8, output_dim=16, right_context_length=0,
                time_reduction_input_dim=16 // stride, time_reduction_stride=stride,
                num_heads=2, ffn_dim=32, num_layers_block1=1, num_layers_block2=1,
                dropout=0, depthwise_conv_kernel_size=3, compression_filter="none",
                sequence_mode="offline",
            ).eval()
            x = torch.randn(2, 13, 8)
            lengths = torch.tensor([13, 9])
            output, result = encoder(x, lengths)
            self.assertEqual(result.tolist(), [(13 + stride - 1) // stride, (9 + stride - 1) // stride])
            changed = x.clone(); changed[1, 9:] = 10000
            torch.testing.assert_close(encoder(changed, lengths)[0], output)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(encoder.input_linear.weight.grad).all())

    def test_offline_rejects_obsolete_config_controls(self):
        for wrong in ({"right_context_length": 0}, {"compression": {"keep_rate": None}},
                      {"compression": {"right_context_length": 0}}):
            with self.assertRaises(ValueError):
                sequence_options_from_config({"sequence_mode": "offline", **wrong})

    def test_offline_feature_mask_loss_and_checkpoint_definition(self):
        try:
            from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
            from common import Batch
        except ModuleNotFoundError:
            self.skipTest("Lightning training runtime required")
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory) / "stats.json"
            stats.write_text(json.dumps({"mean": [1.] * 8, "invstddev": [1.] * 8}))
            module = LibriSpeechRNNTModule(
                librispeech_path=directory, global_stats_path=str(stats),
                input_dim=8, n_mels=8, encoding_dim=16, num_symbols=69,
                tokenizer_type="phone69", right_context_length=0, sequence_mode="offline",
                time_reduction_input_dim=8, time_reduction_stride=2,
                compression_detector={"type": "cosine_diff"}, compression_mixer={"type": "learned"},
                num_heads=2, ffn_dim=32, num_layers_block1=1, num_layers_block2=1,
                depthwise_conv_kernel_size=3, symbol_embedding_dim=8, lstm_hidden_dim=8,
                num_lstm_layers=1, dropout=0, num_workers=0,
            )
            samples = [(torch.randn(1, 4800), 16000, "CH AE1 P"),
                       (torch.randn(1, 3200), 16000, "AH0 T")]
            batch = module._valid_collate_fn(samples)
            self.assertEqual(batch.features.size(1), int(batch.feature_lengths.max()))
            self.assertTrue((batch.features[1, batch.feature_lengths[1]:] == 0).all())
            module.log = lambda *a, **kw: None
            loss = module._step(batch, 0, "train"); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            ckpt = {}; module.on_save_checkpoint(ckpt); module.on_load_checkpoint(ckpt)
            del ckpt["offline_native_definition"]
            with self.assertRaisesRegex(ValueError, "offline natural"):
                module.on_load_checkpoint(ckpt)


if __name__ == "__main__":
    unittest.main()
