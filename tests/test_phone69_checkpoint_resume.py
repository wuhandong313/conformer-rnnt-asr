import copy
import unittest

from conformer_rnnt.checkpoint_utils import phone69_resume_metadata


class Phone69CheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = {
            'optimizer_states': [{}], 'lr_schedulers': [{}],
            'hyper_parameters': {'max_token_limit': 8000}, 'global_step': 400,
            'loops': {'fit_loop': {'epoch_progress': {'current': {'ready': 1, 'completed': 0}}}},
        }

    def test_same_budget_keeps_mid_epoch_resume(self):
        result = phone69_resume_metadata(self.checkpoint, 8000)
        self.assertEqual(result['resume_global_step'], 400)
        self.assertFalse(result['resume_at_epoch_boundary'])

    def test_larger_batch_rejects_old_mid_epoch_offsets(self):
        with self.assertRaisesRegex(ValueError, 'completed epoch boundary'):
            phone69_resume_metadata(self.checkpoint, 24000)

    def test_larger_batch_accepts_completed_epoch(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed'] = 1
        result = phone69_resume_metadata(checkpoint, 24000)
        self.assertEqual(result['previous_max_token_limit'], 8000)
        self.assertEqual(result['resume_completed_epochs'], 1)
        self.assertTrue(result['resume_at_epoch_boundary'])

    def test_weights_only_is_not_a_training_resume(self):
        del self.checkpoint['optimizer_states']
        with self.assertRaisesRegex(ValueError, 'optimizer and scheduler'):
            phone69_resume_metadata(self.checkpoint, 8000)


if __name__ == '__main__':
    unittest.main()
