import unittest
import torch
from conformer_rnnt.compression.mixers import LearnedWeightMixer

class NineFrameMixerTests(unittest.TestCase):
    def test_nine_frame_input_boundaries_and_uniform_mix(self):
        mixer = LearnedWeightMixer(2, 9, 'nine_frames')
        for p in mixer.parameters(): torch.nn.init.zeros_(p)
        frames = torch.arange(24.0).reshape(12, 2).requires_grad_()
        idx = torch.tensor([0, 5, 11])
        captured = []
        handle = mixer.weight_predictor[0].register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
        out = mixer.mix(frames, idx, torch.randn(12), 12)
        handle.remove()
        offsets = torch.arange(-4, 5)
        expected = frames.detach()[(idx[:, None]+offsets).clamp(0, 11)].reshape(3, 18)
        torch.testing.assert_close(captured[0], expected)
        torch.testing.assert_close(out[0], frames[:5].mean(0))
        torch.testing.assert_close(out[1], frames[1:10].mean(0))
        torch.testing.assert_close(out[2], frames[7:].mean(0))
        torch.testing.assert_close(out, mixer.mix(frames, idx, torch.full((12,), 999.), 12))
        out.sum().backward()
        self.assertTrue(torch.isfinite(frames.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in mixer.parameters()))
    def test_short_and_wrong_window(self):
        with self.assertRaises(ValueError): LearnedWeightMixer(2, 5, 'nine_frames')
        mixer = LearnedWeightMixer(2, 9, 'nine_frames')
        frames = torch.randn(2, 2)
        self.assertTrue(torch.isfinite(mixer.mix(frames, torch.tensor([0,1]), torch.zeros(2), 2)).all())
    def test_previous_five_frame_shape_unchanged(self):
        mixer = LearnedWeightMixer(128, 5, 'five_frames')
        self.assertEqual(tuple(mixer.weight_predictor[0].weight.shape), (32, 640))
        nine = LearnedWeightMixer(128, 9, 'nine_frames')
        self.assertEqual(tuple(nine.weight_predictor[0].weight.shape), (32, 1152))
        self.assertEqual(sum(p.numel() for p in nine.parameters()), 37193)
        with self.assertRaises(RuntimeError): nine.load_state_dict(mixer.state_dict())
if __name__ == '__main__': unittest.main()
