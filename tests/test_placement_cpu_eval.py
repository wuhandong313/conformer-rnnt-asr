import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from eval_phone69_placement_checkpoint import COMPONENTS, placement_encoder_macs, placement_sites, summarize


class PlacementCPUCountsTest(unittest.TestCase):
    def test_legacy_and_multistage_manifest_positions(self):
        self.assertEqual(placement_sites({'position': 3}), [3])
        self.assertEqual(placement_sites({'positions': [0, 3]}), [0, 3])
        for sites in ([], [3, 0], [0, 0], [True], [11]):
            with self.subTest(sites=sites), self.assertRaises(ValueError):
                placement_sites({'positions': sites})

    def test_mel_layer3_counts_both_stage_widths(self):
        config = ROOT / 'configs/server_phone69_460h_placement_mel_layer3_stride1_s42_e50_supp20261112.yaml'
        model = yaml.safe_load(config.read_text())['model']
        lengths = [40] * 3 + [16] * 7
        result = placement_encoder_macs(model, 100, lengths, 16)
        self.assertEqual(result, dict(
            encoder_weights=380800 * sum(lengths),
            attention_products=256 * sum(t * t for t in lengths),
            projections=40 * 80 * 128 + 16 * 128 * 256,
            fusion_mlp=40 * (5 * 80 * 32 + 32 * 5) + 16 * (5 * 128 * 32 + 32 * 5),
            detector_dots=99 * 80 + 39 * 128,
            fusion_weighted_sum=40 * 5 * 80 + 16 * 5 * 128))

    def model(self, site):
        path = ROOT / ('configs/server_phone69_460h_placement_layer%d_stride1_s42_e50_supp20261112.yaml' % site)
        return yaml.safe_load(path.read_text())['model']

    def test_each_requested_placement_counts_pre_and_post_compression(self):
        for site in (1, 3, 9):
            with self.subTest(site=site):
                lengths = [100] * site + [40] * (10 - site)
                result = placement_encoder_macs(self.model(site), 100, lengths, 40)
                expected = dict(encoder_weights=380800 * sum(lengths),
                                attention_products=256 * sum(t * t for t in lengths),
                                projections=100 * 80 * 128 + 40 * 128 * 256,
                                fusion_mlp=40 * (5 * 128 * 32 + 32 * 5),
                                detector_dots=99 * 128, fusion_weighted_sum=40 * 5 * 128)
                self.assertEqual(result, expected)

    def test_trace_from_another_position_is_rejected(self):
        with self.assertRaises(ValueError):
            placement_encoder_macs(self.model(3), 100, [100] + [40] * 9, 40)

    def test_one_frame_uses_the_actual_compressor_bypass(self):
        counts = placement_encoder_macs(self.model(9), 1, [1] * 10, 1)
        for key in ('fusion_mlp', 'detector_dots', 'fusion_weighted_sum'):
            self.assertEqual(counts[key], 0)

    def test_corpus_per_and_macs_use_summed_numerators(self):
        rows = [dict(uid='a', reference_phones=2, phone_errors=1, audio_seconds=1,
                     **{key: 10 for key in COMPONENTS}),
                dict(uid='b', reference_phones=8, phone_errors=0, audio_seconds=3,
                     **{key: 20 for key in COMPONENTS})]
        result = summarize(rows, ['a', 'b'], expected_phones=10)
        self.assertEqual(result['corpus_PER'], 0.1)
        self.assertEqual(result['gmac_per_audio_second'], 240 / 4 / 1e9)
        with self.assertRaises(ValueError):
            summarize(rows, ['b', 'a'], expected_phones=10)
        with self.assertRaises(ValueError):
            summarize(rows[:1], ['a'], expected_phones=10)


if __name__ == '__main__':
    unittest.main()
