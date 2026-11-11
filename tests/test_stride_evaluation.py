"""Verify corpus coverage and component aggregation without model execution."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/evaluation/eval_phone69_stride_checkpoint.py'
spec = importlib.util.spec_from_file_location('stride_eval', path)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class StrideEvaluationTests(unittest.TestCase):
    def rows(self):
        return [dict(uid='a', reference_phones=2, phone_errors=1, audio_seconds=2,
                     **{k: 100 for k in evaluation.COMPONENTS}),
                dict(uid='b', reference_phones=3, phone_errors=0, audio_seconds=3,
                     **{k: 200 for k in evaluation.COMPONENTS})]

    def test_corpus_ratio_and_components_use_full_totals(self):
        result = evaluation.summarize(self.rows(), ['a', 'b'], expected_phones=5)
        self.assertEqual(result['corpus_PER'], 0.2)
        self.assertEqual(result['total_major_macs'], 2400)
        self.assertEqual(result['components_macs']['predictor'], 300)
        self.assertEqual(result['gmac_per_audio_second'], 2400 / 5 / 1e9)

    def test_reordered_or_incomplete_corpus_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'coverage/order'):
            evaluation.summarize(self.rows(), ['b', 'a'], expected_phones=5)
        with self.assertRaisesRegex(ValueError, 'phone coverage'):
            evaluation.summarize(self.rows(), ['a', 'b'], expected_phones=6)

    def test_negative_operator_count_is_rejected(self):
        rows = self.rows()
        rows[0]['predictor'] = -1000
        with self.assertRaisesRegex(ValueError, 'Invalid corpus totals'):
            evaluation.summarize(rows, ['a', 'b'], expected_phones=5)


if __name__ == '__main__':
    unittest.main()
