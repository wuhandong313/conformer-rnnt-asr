from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.server import append_phone69_layer3_rerun as rerun


class Layer3RerunTests(unittest.TestCase):
    def test_repeats_single_stage_with_same_seed_and_budget(self):
        _, plan, config = rerun.repeat_recipe()
        self.assertEqual(config['seed'], 42)
        self.assertEqual(config['training']['epochs'], 50)
        self.assertEqual(config['data']['max_token_limit'], 8800)
        self.assertEqual(plan['batch_search']['maximum'], 8800)
        self.assertIsNone(config['paths']['from_checkpoint'])
        self.assertEqual([s['after_layer'] for s in config['model']['compression']['stages']], [3])

    def test_waits_for_live_stride4_after_completion_state(self):
        queue = MagicMock()
        state = dict(status='complete', experiment='stride4_dim256', controller_pid=33, code_commit='same')
        with (patch.object(rerun, 'read', return_value=state),
              patch.object(rerun, 'matches', side_effect=[True, False]),
              patch.object(rerun, 'verify_stride4_completion', return_value=state) as verify,
              patch.object(rerun, 'save') as save,
              patch.object(rerun.time, 'sleep', side_effect=lambda _: verify.assert_not_called())):
            rerun.wait_for_stride4(queue, 8800)
        self.assertEqual(queue.state.call_args.args[0], 'waiting_for_predecessor')
        verify.assert_called_once()
        save.assert_called_once()

    def test_rejects_missing_or_changed_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            best = Path(directory) / 'best.ckpt'
            boundary = Path(directory) / 'epoch50.ckpt'
            best.write_text('validation best')
            boundary.touch()
            record = dict(epoch=50, global_step=1234, best_checkpoint=str(best),
                          best_checkpoint_sha256=rerun.sha(best), epoch_boundary_checkpoint=str(boundary),
                          epoch_boundary_metadata=dict(resume_completed_epochs=50,
                                                       resume_at_epoch_boundary=True, resume_global_step=1234))
            state = dict(experiment='stride4_dim256', status='complete', **record)
            self.assertEqual(rerun.verify_stride4_completion(state, record), record)
            best.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'verified final and best'):
                rerun.verify_stride4_completion(state, record)

    def test_failed_stride4_does_not_release_waiting_l3(self):
        queue = MagicMock()
        state = dict(status='failed', experiment='stride4_dim256', controller_pid=33, code_commit='same')
        with (patch.object(rerun, 'read', return_value=state),
              patch.object(rerun, 'matches', return_value=False),
              patch.object(rerun, 'verify_stride4_completion') as verify):
            with self.assertRaises(RuntimeError):
                rerun.wait_for_stride4(queue, 8800)
            verify.assert_not_called()


if __name__ == '__main__':
    unittest.main()
