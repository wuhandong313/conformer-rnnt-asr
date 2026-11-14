"""Verify queue-parent retirement never signals the training worker."""
import json
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

from scripts.server import continue_phone69_placement_local as handoff


class PlacementHandoffTests(unittest.TestCase):
    def test_only_verified_controller_pid_receives_termination(self):
        calls={538:0,557:0}
        def matches(pid,filename):
            calls[pid]+=1
            return pid==557 or calls[pid]==1
        with patch.object(handoff,'process_matches',side_effect=matches),patch.object(handoff.os,'kill') as kill:
            handoff.retire_queue_parent(538,557)
            kill.assert_called_once_with(538,signal.SIGTERM)

    def test_pid_reuse_or_missing_worker_never_gets_signalled(self):
        for controller_ok,worker_ok in [(False,True),(True,False),(False,False)]:
            with self.subTest(controller=controller_ok,worker=worker_ok), \
                    patch.object(handoff,'process_matches',side_effect=lambda pid,name: worker_ok if pid==557 else controller_ok), \
                    patch.object(handoff.os,'kill') as kill:
                with self.assertRaises(ValueError):handoff.retire_queue_parent(538,557)
                kill.assert_not_called()

    def test_identical_worker_and_controller_pid_is_rejected(self):
        with patch.object(handoff.os,'kill') as kill:
            with self.assertRaises(ValueError):handoff.retire_queue_parent(557,557)
            kill.assert_not_called()

    def test_split_manifests_are_disjoint_and_never_require_old_checkpoints(self):
        root=Path(__file__).resolve().parents[1]
        here=json.loads((root/'configs/phone69_placement_current_server_layer9_20261114.json').read_text())
        there=json.loads((root/'configs/phone69_placement_remote_three_20261114.json').read_text())
        self.assertEqual([x['name'] for x in here['experiments']],['layer9'])
        self.assertEqual([x['name'] for x in there['experiments']],['mel_layer3','mel_layer7','layers2_7_n3'])
        self.assertTrue(all(x['action']=='train' for x in here['experiments']+there['experiments']))


if __name__=='__main__':unittest.main()
