"""Local safety checks for waiting on L3/L9 before any new GPU work."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.server import append_phone69_placement_queue as append


class PlacementAppendTests(unittest.TestCase):
    def test_waits_for_live_predecessor_even_after_its_complete_marker(self):
        self.assertFalse(append.predecessor_done({'status':'waiting_for_running_layer3'},True))
        self.assertFalse(append.predecessor_done({'status':'complete'},True))
        self.assertTrue(append.predecessor_done({'status':'complete'},False))

    def test_failed_or_disappeared_predecessor_never_permits_new_training(self):
        for state,alive in [({'status':'failed'},True),({'status':'failed'},False),
                            ({'status':'starting_layer9_only'},False),({},False)]:
            with self.subTest(state=state,alive=alive),self.assertRaises((ValueError,RuntimeError)):
                append.predecessor_done(state,alive)

    def test_missing_l9_completion_is_rejected_before_checkpoint_loading(self):
        with patch.object(append,'read',return_value=None),patch.object(append,'verify_completed') as verify:
            with self.assertRaisesRegex(ValueError,'L9 completion'):
                append.verify_predecessor()
            verify.assert_not_called()

    def test_returned_manifest_contains_only_the_requested_mel_runs(self):
        root=Path(__file__).resolve().parents[1]
        plan=json.loads((root/'configs/phone69_placement_current_server_three_20261115.json').read_text())
        self.assertEqual([x['name'] for x in plan['experiments']],append.NAMES)
        self.assertTrue(all(x['action']=='train' for x in plan['experiments']))
        old=json.loads((root/'configs/phone69_placement_remote_three_20261114.json').read_text())
        self.assertEqual(plan['experiments'],old['experiments'])
        self.assertNotEqual(plan['runtime_dir'],old['runtime_dir'])


if __name__=='__main__':unittest.main()
