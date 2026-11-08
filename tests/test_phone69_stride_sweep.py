"""Batch-boundary search and the requested fixed-stride control conditions."""
import copy
from pathlib import Path
import unittest

import yaml

from scripts.server.phone69_stride_sweep import BASELINE, ROOT, maximize_budget, validate_recipe


class StrideSweepTests(unittest.TestCase):
    def search(self, limit):
        return maximize_budget(lambda b: dict(passed=b <= limit), initial=7000,
                               minimum=1000, maximum=50000, quantum=100, growth=1.4)

    def test_upward_search_reaches_one_quantum_of_failure(self):
        budget, trials, limiter = self.search(12600)
        self.assertEqual(budget, 12600)
        self.assertTrue(any(t['budget'] == 12700 and not t['passed'] for t in trials))
        self.assertEqual(limiter, 'memory_or_joint_boundary')

    def test_initial_oom_searches_downward_without_reusing_failed_budget(self):
        budget, trials, _ = self.search(3200)
        self.assertEqual(budget, 3200)
        self.assertFalse(trials[0]['passed'])
        self.assertTrue(any(t['budget'] == 3300 and not t['passed'] for t in trials))
        with self.assertRaises(RuntimeError):
            self.search(900)

    def test_numel_guard_plateau_stops_unbounded_budget_growth(self):
        budget, trials, limiter = maximize_budget(
            lambda b: dict(passed=True, batch_structure_sha256='same-shapes'),
            initial=7000, minimum=1000, maximum=100000, quantum=100, growth=1.4)
        self.assertEqual((budget, len(trials), limiter), (9800, 2, 'rnnt_guard_plateau'))

    def test_only_stride_and_output_path_differ_from_main_baseline(self):
        baseline = yaml.safe_load((ROOT / BASELINE).read_text())
        for stride in (1, 2, 8):
            recipe = ROOT / f'configs/server_phone69_460h_offline_native_stride{stride}_none_proj128_s42_e50.yaml'
            config = yaml.safe_load(recipe.read_text())
            validate_recipe(config, baseline, stride)
            changed = copy.deepcopy(config)
            changed['model']['num_layers_block2'] -= 1
            with self.assertRaises(ValueError):
                validate_recipe(changed, baseline, stride)
            self.assertNotEqual(config['paths']['exp_dir'], baseline['paths']['exp_dir'])


if __name__ == '__main__':
    unittest.main()
