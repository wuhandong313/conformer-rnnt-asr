"""Local safety checks for the authorized position-only serial supplement."""

import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import yaml

from scripts.server import phone69_placement_supplement as queue


ROOT = Path(__file__).resolve().parents[1]
BASELINE = yaml.safe_load((ROOT / 'configs/server_phone69_460h_placement_layer5_stride1_s42_e50.yaml').read_text())


class PlacementSupplementTests(unittest.TestCase):
    def recipe(self):
        cfg = copy.deepcopy(BASELINE)
        cfg['model']['compression']['stages'][0]['after_layer'] = 1
        cfg['paths']['exp_dir'] += '_new'
        return cfg

    def test_all_requested_recipes_only_change_the_authorized_positions(self):
        _, plan = queue.load_plan(ROOT / 'configs/phone69_placement_supplement_20261112.json')
        self.assertEqual([i['name'] for i in plan['experiments'] if i['action'] == 'train'],
                         ['layer1', 'layer3', 'layer9', 'mel_layer3', 'mel_layer7', 'layers2_7_n3'])
        self.assertEqual(sum(i['action'] == 'reuse' for i in plan['experiments']), 4)

    def test_backend_fusion_guard_and_seed_drift_are_rejected(self):
        for change in ('backend', 'fusion', 'guard', 'seed'):
            cfg = self.recipe()
            if change == 'backend':
                cfg['model']['compression']['execution_backend'] = 'batched'
            elif change == 'fusion':
                cfg['model']['compression']['stages'][0]['window_size'] = 3
            elif change == 'guard':
                cfg['data']['rnnt_frames_per_char'] = 3.0
            else:
                cfg['seed'] = 7
            with self.subTest(change=change), self.assertRaises(ValueError):
                queue.validate_recipe(cfg, BASELINE, [1], fresh=True)

    def test_fresh_training_cannot_silently_resume_old_weights(self):
        cfg = self.recipe()
        cfg['paths']['from_checkpoint'] = '/old/checkpoint.ckpt'
        with self.assertRaisesRegex(ValueError, 'start from scratch'):
            queue.validate_recipe(cfg, BASELINE, [1], fresh=True)

    def completed_fixture(self, root):
        cfg = self.recipe()
        exp = root / 'experiment'
        exp.mkdir()
        cfg['paths']['exp_dir'] = str(exp)
        config = root / 'recipe.yaml'
        config.write_text(yaml.safe_dump(cfg))
        best = exp / 'best.ckpt'
        best.write_bytes(b'original best model')
        boundary = exp / 'boundary.ckpt'
        boundary.write_bytes(b'full boundary')
        (exp / 'best_checkpoint.txt').write_text(str(best) + '\n')
        run = dict(status='trained', epoch=50, config=str(config), config_sha256=queue.sha(config),
                   compression=cfg['model']['compression'], max_token_limit=5000, internal_dim=128,
                   time_reduction_stride=1, best_checkpoint=str(best), epoch_boundary_checkpoint=str(boundary),
                   global_step=1000, code_commit='actual-startup',
                   offline_native_definition={'stages': [{'after_layer': 1}]})
        (exp / 'run.json').write_text(json.dumps(run))
        item = dict(name='test', action='reuse', config='recipe.yaml', positions=[1],
                    expected_best_sha256=queue.sha(best))
        return item, run, best

    def test_reuse_rejects_replaced_best_before_loading_the_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            item, _, best = self.completed_fixture(root)
            best.write_bytes(b'replaced best')
            torch = types.SimpleNamespace(load=Mock())
            helper = types.SimpleNamespace(phone69_resume_metadata=Mock())
            with patch.object(queue, 'ROOT', root), patch.dict('sys.modules', {
                    'torch': torch, 'conformer_rnnt.checkpoint_utils': helper}):
                with self.assertRaisesRegex(ValueError, 'frozen completed experiment'):
                    queue.verify_completed(item)
            torch.load.assert_not_called()

    def test_reuse_reads_full_epoch_fifty_and_preserves_startup_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            item, _, _ = self.completed_fixture(root)
            metadata = dict(resume_completed_epochs=50, resume_at_epoch_boundary=True,
                            resume_global_step=1000, previous_max_token_limit=5000)
            torch = types.SimpleNamespace(load=Mock(return_value={'state_dict': {'weight': 1}}))
            helper = types.SimpleNamespace(phone69_resume_metadata=Mock(return_value=metadata))
            with patch.object(queue, 'ROOT', root), patch.dict('sys.modules', {
                    'torch': torch, 'conformer_rnnt.checkpoint_utils': helper}):
                result = queue.verify_completed(item)
                self.assertEqual(result['training_commit'], 'actual-startup')
                self.assertEqual(result['epoch_boundary_metadata'], metadata)
                self.assertEqual(helper.phone69_resume_metadata.call_args.kwargs,
                                 {'expected_definition': {'stages': [{'after_layer': 1}]}})
                metadata['resume_completed_epochs'] = 49
                with self.assertRaisesRegex(ValueError, 'full 50-epoch'):
                    queue.verify_completed(item)
            self.assertEqual(torch.load.call_count, 2)

    def test_confirmation_requires_at_least_twenty_four_real_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            q = queue.PlacementQueue.__new__(queue.PlacementQueue)
            q.audit = Path(folder).resolve()
            q.plan = {'batch_search': {'max_reserved_fraction': 0.96}}
            q.sources = {'recipe.yaml': 'original-sha'}
            q.trial = 0
            q.unchanged = Mock()
            q.wait_idle = Mock()
            q.state = Mock()
            def launch(command, **kwargs):
                output = Path(command[command.index('--out') + 1])
                rounds = int(command[command.index('--rounds') + 1])
                output.write_text(json.dumps(dict(status='passed', budget=5000, config_sha256='original-sha',
                    selected_batches=[{}, {}, {}], results=[{'peak_reserved_GiB': 1}] * (3 * rounds))))
                return types.SimpleNamespace(wait=lambda: 0, pid=123)
            with patch.object(queue, 'ROOT', q.audit), patch.object(queue.subprocess, 'Popen', side_effect=launch):
                # Three selected real shapes over two rounds are insufficient.
                result = q.probe({'name': 'layer1', 'config': 'recipe.yaml', 'positions': [1]},
                                 5000, confirmation=True, rounds=2)
            self.assertEqual(result['optimizer_steps'], 24)
            self.assertEqual(q.trial, 2)
            self.assertEqual(len(list(q.audit.glob('*.json'))), 2)
            self.assertEqual(len(list(q.audit.glob('*.log'))), 2)

    def test_lock_holder_does_not_deadlock_on_another_waiting_controller(self):
        q = queue.PlacementQueue.__new__(queue.PlacementQueue)
        q.state = Mock()
        with patch.object(queue, 'live_jobs', return_value=[{'pid': 99, 'kind': 'controller'}]), \
                patch.object(queue.subprocess, 'check_output', return_value=''), \
                patch.object(queue.time, 'sleep', side_effect=AssertionError('lock-holder deadlocked')):
            q.wait_idle()
        q.state.assert_not_called()

    def test_real_worker_blocks_even_during_cpu_initialization(self):
        q = queue.PlacementQueue.__new__(queue.PlacementQueue)
        q.state = Mock()
        with patch.object(queue, 'live_jobs', side_effect=[[{'pid': 99, 'kind': 'calibration'}], []]), \
                patch.object(queue.subprocess, 'check_output', return_value=''), \
                patch.object(queue.time, 'sleep') as sleep:
            q.wait_idle()
        sleep.assert_called_once_with(10)
        self.assertEqual(q.state.call_args[0][0], 'waiting_for_gpu')


if __name__ == '__main__':
    unittest.main()
