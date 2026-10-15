import unittest

from conformer_rnnt.encoder import ConformerEncoder


class EncoderCompressionContextTest(unittest.TestCase):
    def _build_encoder(self, stride, compression_right_context_length=None):
        return ConformerEncoder(
            input_dim=80,
            output_dim=256,
            right_context_length=4,
            time_reduction_input_dim=256 // stride,
            time_reduction_stride=stride,
            compression_right_context_length=compression_right_context_length,
            num_heads=8,
            ffn_dim=512,
            num_layers_block1=1,
            num_layers_block2=1,
            compression_detector={"type": "cosine_diff"},
            compression_mixer={"type": "learned"},
            compression_keep_rate=0.27,
        )

    def test_explicit_compression_context_uses_reduced_frames(self):
        encoder = self._build_encoder(stride=4, compression_right_context_length=1)
        self.assertEqual(encoder.compression.right_context_length, 1)

    def test_omitted_compression_context_keeps_legacy_value(self):
        encoder = self._build_encoder(stride=4)
        self.assertEqual(encoder.compression.right_context_length, 4)

    def test_compression_keeps_four_context_frames_without_time_stride(self):
        encoder = self._build_encoder(stride=1)
        self.assertEqual(encoder.compression.right_context_length, 4)


if __name__ == "__main__":
    unittest.main()
