import importlib.util
import json
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('theory', ROOT / 'scripts/evaluation/theoretical_phone69_macs.py')
theory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(theory)


class TheoryTests(unittest.TestCase):
    def config(self, name):
        return yaml.safe_load((theory.ARCHIVE / name / 'config.yaml').read_text())['model']

    def test_all_archived_structural_counts(self):
        manifest = theory.read(theory.ARCHIVE / 'manifest.json')
        for item in manifest['models']:
            model = self.config(item['id'])
            rows = theory.read(theory.COUNTS / (item['id'] + '.json'))['rows']
            self.assertEqual(len(rows), 2620)
            for audio, row in zip(manifest['audio'], rows):
                trace = theory.observed_trace(model, audio['samples'] // 160 + 1, row)
                self.assertEqual(theory.encoder_macs(model, trace),
                                 {k: row.get(k, 0) for k in theory.ENCODER_KEYS})

    def test_pooled_ratio_not_utterance_mean(self):
        model = self.config('E02')
        rows = [dict(layer_frames=[n] * 2 + [k] * 8, output_steps=k)
                for n, k in [(10, 9), (90, 9)]]
        r = theory.calibrate(model, [10, 90], rows)[0]
        self.assertEqual(r['retention'], .18)
        self.assertEqual(r['utterance_ratio_mean'], .5)
        self.assertEqual(theory.projected_trace(model, 100, [r])['output_frames'], 18)

    def test_conditional_multi_stage_ratio(self):
        model = self.config('E10')
        row = dict(layer_frames=[40] * 5 + [10] * 5, output_steps=10)
        rates = theory.calibrate(model, [100], [row])
        self.assertEqual([r['retention'] for r in rates], [.4, .25])
        trace = theory.projected_trace(model, 1000, rates)
        self.assertEqual(trace['projected_frames'], 400)
        self.assertEqual(trace['layer_frames'], [400] * 5 + [100] * 5)

    def test_fixed_stride_and_duration(self):
        self.assertEqual(theory.mel_frames(10), 1001)
        self.assertEqual(theory.mel_frames(30), 3001)
        model = self.config('E09')
        trace = theory.projected_trace(model, 1001, [])
        self.assertEqual(trace['layer_frames'], [251] * 10)
        expected = 1001 * 80 * 128 + 10 * (251 * (4 * 512 * 512 + 7 * 512**2 + 31 * 512)
                                          + 2 * 512 * 251**2) + 251 * 512 * 256
        self.assertEqual(sum(theory.encoder_macs(model, trace).values()), expected)

    def test_single_frame_bypass(self):
        model = self.config('E10')
        trace = theory.frame_trace(model, 1, lambda site, n: 1)
        count = theory.encoder_macs(model, trace)
        self.assertEqual([count[k] for k in ('fusion_mlp', 'fusion_weighted_sum', 'detector_dots')], [0, 0, 0])

    def test_reference_decoder_is_prescribed_path(self):
        result = theory.reference_decoder_macs(self.config('E09'), 126, 98)
        self.assertEqual(result, dict(predictor=99 * 294912, joiner=224 * 17920))

    def test_invalid_inputs(self):
        for duration in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                theory.mel_frames(duration)
        with self.assertRaises(ValueError):
            theory.projected_trace(self.config('E10'), 1001, [])

    def test_new_three_stage_measurement_matches_archived_inputs(self):
        path = theory.DEFAULT_OUT / 'E11_stage_lengths.json'
        measured = theory.read(path)
        model = yaml.safe_load((theory.TRIPLE / 'config.yaml').read_text())['model']
        self.assertEqual(len(measured['rows']), 2620)
        for row in measured['rows']:
            trace = theory.observed_trace(model, row['input_mel_frames'], row)
            for inferred, observed in zip(trace['stages'], row['stages']):
                self.assertEqual(inferred['after_layer'], observed['after_layer'])
                self.assertEqual([inferred['input_frames']], observed['input'])
                self.assertEqual([inferred['output_frames']], observed['output'])

    def test_full_inventory_and_fixed_duration_coverage(self):
        calibration = (theory.DEFAULT_OUT / 'E11_stage_lengths.json').relative_to(ROOT)
        report, rows = theory.generate([30], calibration)
        self.assertEqual(len(report['models']), 20)
        self.assertEqual(len(rows), 14)
        self.assertTrue(all(r['input_frames'] == 3001 and r['target_phones'] == 292 for r in rows))
        self.assertEqual([m['id'] for m in report['models'] if not m['results']],
                         ['supp_layer1', 'supp_layer3', 'supp_layer9', 'supp_mel_layer3',
                          'supp_mel_layer7', 'supp_layers2_7_n3'])


if __name__ == '__main__':
    unittest.main()
