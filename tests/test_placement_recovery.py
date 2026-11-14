"""Local recovery safety checks; no model execution or server access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.server import recover_phone69_placement_supplement as recovery


def full_checkpoint(epoch=1, step=1866, budget=8800):
    return dict(state_dict={'weight': 1}, optimizer_states=[{'state': 'preserved'}],
                lr_schedulers=[{'last_epoch': step}], global_step=step,
                offline_native_definition={'stages': [3]},
                hyper_parameters={'max_token_limit': budget},
                loops={'fit_loop': {'epoch_progress': {'current': {'ready': epoch, 'completed': epoch}}}})


class PlacementRecoveryTests(unittest.TestCase):
    def test_boundary_sort_is_numeric_and_reads_optimizer_and_scheduler_state(self):
        paths = [Path('epoch-boundary-epoch=09-step=9000-abcd.ckpt'),
                 Path('epoch-boundary-epoch=10-step=10000-abcd.ckpt')]
        checkpoint = full_checkpoint(10, 10000)
        chosen, metadata, errors = recovery.choose_boundary(paths, 8800, {'stages': [3]},
                                                           loader=lambda _: checkpoint)
        self.assertEqual(chosen, paths[1])
        self.assertEqual(metadata['resume_completed_epochs'], 10)
        self.assertEqual(metadata['resume_global_step'], 10000)
        self.assertEqual(errors, [])

    def test_weights_only_or_wrong_placement_cannot_replace_full_boundary(self):
        older = Path('epoch-boundary-epoch=01-step=1866-abcd.ckpt')
        newer = Path('epoch-boundary-epoch=02-step=3732-abcd.ckpt')
        for field in ['optimizer_states', 'lr_schedulers', 'offline_native_definition']:
            bad = full_checkpoint(2, 3732)
            bad[field] = [] if field != 'offline_native_definition' else {'stages': [9]}
            with self.subTest(field=field):
                chosen, _, errors = recovery.choose_boundary([newer, older], 8800, {'stages': [3]},
                    loader=lambda path: bad if path == newer else full_checkpoint())
                self.assertEqual(chosen, older)
                self.assertEqual(len(errors), 1)

    def test_incomplete_epoch_or_different_budget_is_rejected(self):
        path = Path('epoch-boundary-epoch=01-step=1866-abcd.ckpt')
        bad = full_checkpoint()
        bad['loops']['fit_loop']['epoch_progress']['current']['ready'] = 2
        for checkpoint in [bad, full_checkpoint(budget=8300)]:
            with self.assertRaisesRegex(ValueError, 'No intact full'):
                recovery.choose_boundary([path], 8800, {'stages': [3]}, loader=lambda _: checkpoint)

    def test_filename_cannot_claim_more_progress_than_the_checkpoint(self):
        path = Path('epoch-boundary-epoch=50-step=99000-abcd.ckpt')
        with self.assertRaisesRegex(ValueError, 'No intact full'):
            recovery.choose_boundary([path], 8800, {'stages': [3]}, loader=lambda _: full_checkpoint())

    def test_all_archived_checkpoints_survive_in_place_overwrite_and_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            experiment = root / 'experiment'
            checkpoints = experiment / 'checkpoints'
            checkpoints.mkdir(parents=True)
            last, best = checkpoints / 'last.ckpt', checkpoints / 'best.ckpt'
            last.write_bytes(b'original optimizer and loop state')
            best.write_bytes(b'original validation winner')
            archive = root / 'archive'
            records = recovery.preserve_checkpoints(experiment, archive)
            last.write_bytes(b'new rolling state')
            best.unlink()
            self.assertEqual((archive / 'last.ckpt').read_bytes(), b'original optimizer and loop state')
            self.assertEqual((archive / 'best.ckpt').read_bytes(), b'original validation winner')
            self.assertEqual(len(records), 2)
            self.assertTrue(all(row['independent_copy'] for row in records))

    def test_original_budget_mismatch_prevents_resume_before_any_file_mutation(self):
        queue = object.__new__(recovery.RecoveryQueue)
        selected = (Path('original.yaml'), {}, {'budget': 8800, 'effective_config_sha256': 'expected'})
        with patch.object(recovery, 'preserve_checkpoints') as preserve:
            with self.assertRaisesRegex(ValueError, 'differs from its selected'):
                queue.prepare_resume({'name': 'layer3'}, selected,
                    {'config_sha256': 'expected', 'max_token_limit': 8300})
            preserve.assert_not_called()

    def test_an_existing_recovery_deployment_cannot_be_relaunched_or_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            audit = Path(folder)
            deployment = audit / 'deployment.json'
            deployment.write_text(json.dumps({'original': 'must remain intact'}))
            with patch.object(recovery, 'OUT', audit), \
                    patch.object(recovery, 'load_plan', return_value=(Path('manifest.json'), {})), \
                    patch.object(recovery, 'source_identity', return_value={}):
                with self.assertRaisesRegex(FileExistsError, 'already launched'):
                    recovery.RecoveryQueue()
            self.assertEqual(json.loads(deployment.read_text()), {'original': 'must remain intact'})


if __name__ == '__main__':
    unittest.main()
