import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.evaluation.theoretical_phone69_macs import encoder_macs, mel_frames, projected_trace
from scripts.server import phone69_stride4_dim256 as stride


class Stride4AppendTests(unittest.TestCase):
    def test_requested_shape_budget_and_reference_cost(self):
        config = stride.recipe()
        model = config['model']
        self.assertEqual((model['time_reduction_stride'], model['time_reduction_input_dim']), (4, 64))
        self.assertEqual(config['data']['max_token_limit'], 8800)
        self.assertEqual(config['training']['epochs'], 50)
        trace = projected_trace(model, mel_frames(30), [])
        self.assertEqual(trace['layer_frames'], [751] * 10)
        self.assertEqual(sum(encoder_macs(model, trace).values()), 10_394_497_536)

    def test_waits_for_predecessor_process_even_with_completion_state(self):
        queue = stride.Stride4Queue.__new__(stride.Stride4Queue)
        queue.state = MagicMock()
        state = dict(status='complete', controller_pid=123)
        with (patch.object(stride, 'read', return_value=state),
              patch.object(stride, 'matches', side_effect=[True, False]),
              patch.object(stride, 'verified_predecessor', return_value=stride.NAMES) as verify,
              patch.object(stride.time, 'sleep', side_effect=lambda _: verify.assert_not_called())):
            queue.wait_previous()
        self.assertEqual([call.args[0] for call in queue.state.call_args_list],
                         ['waiting_for_predecessor', 'waiting_for_training_lock'])
        verify.assert_called_once()

    def test_refuses_incomplete_or_missing_predecessor_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'existing.ckpt'
            checkpoint.touch()
            rows = [dict(name=name, status='trained', epoch=50, global_step=1234,
                         best_checkpoint=str(checkpoint), epoch_boundary_checkpoint=str(checkpoint),
                         epoch_boundary_metadata=dict(resume_completed_epochs=50,
                                                      resume_at_epoch_boundary=True, resume_global_step=1234))
                    for name in stride.NAMES]
            queue = dict(status='complete', completed=rows)
            self.assertEqual(stride.verified_predecessor(queue, dict(models=rows)), stride.NAMES)
            historical_rows = rows + [dict(name='previously_completed_run')]
            historical_queue = dict(status='complete', completed=historical_rows)
            self.assertEqual(stride.verified_predecessor(historical_queue, dict(models=historical_rows)),
                             stride.NAMES)
            with self.assertRaisesRegex(RuntimeError, 'matching completion'):
                stride.verified_predecessor(queue, dict(models=rows[:-1]))
            checkpoint.unlink()
            with self.assertRaisesRegex(RuntimeError, 'Incomplete predecessor'):
                stride.verified_predecessor(queue, dict(models=rows))

    def test_calibration_failure_does_not_start_training(self):
        queue = stride.Stride4Queue.__new__(stride.Stride4Queue)
        with tempfile.TemporaryDirectory() as directory:
            queue.exp = Path(directory) / 'experiment'
            with (patch.object(stride, 'LOCK', Path(directory) / 'lock'),
                  patch.object(queue, 'wait_previous'),
                  patch.object(queue, 'wait_idle'),
                  patch.object(queue, 'unchanged'),
                  patch.object(queue, 'confirm_batch', side_effect=RuntimeError('Batch did not fit')),
                  patch.object(queue, 'train') as train):
                with self.assertRaisesRegex(RuntimeError, 'Batch did not fit'):
                    queue.run()
                train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
