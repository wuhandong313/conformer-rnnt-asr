"""Exercise triple-stage launch guards locally without spawning a trainer."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from scripts.server import phone69_triple_compression as launcher

REPO = Path(__file__).resolve().parents[1]
COMMIT = 'a' * 40


@contextmanager
def environment(*, busy=False, dirty=False):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = yaml.safe_load((REPO / launcher.BASE_CONFIG).read_text())
        config = yaml.safe_load((REPO / launcher.CONFIG).read_text())
        config['paths']['exp_dir'] = str(root / launcher.EXPERIMENT)
        for name, value in ((launcher.BASE_CONFIG, base), (launcher.CONFIG, config)):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(value))
        fingerprint = hashlib.sha256((root / launcher.CONFIG).read_bytes()).hexdigest()

        def command(arguments):
            if arguments == ['git', 'rev-parse', 'HEAD']:
                return COMMIT
            if arguments == ['git', 'rev-parse', '--abbrev-ref', 'HEAD']:
                return 'main'
            if arguments[:2] == ['git', 'status']:
                return ' M model.py' if dirty else ''
            if '--query-compute-apps=pid' in arguments:
                return '9876' if busy else ''
            if '--query-gpu=memory.used' in arguments:
                return '4'
            raise AssertionError(arguments)

        with patch.object(launcher, 'REPO', root), patch.object(launcher, 'LOCK', root / 'serial.lock'), \
                patch.object(launcher, 'output', side_effect=command), \
                patch.object(launcher.subprocess, 'Popen', return_value=Mock(pid=12345)) as spawn:
            yield root, fingerprint, spawn


class TripleCompressionLaunchTests(unittest.TestCase):
    def test_busy_gpu_leaves_experiment_absent_and_other_processes_untouched(self):
        with environment(busy=True) as (root, fingerprint, spawn):
            with self.assertRaisesRegex(RuntimeError, 'GPU is occupied'):
                launcher.launch(COMMIT, fingerprint)
            self.assertFalse((root / launcher.EXPERIMENT).exists())
            spawn.assert_not_called()

    def test_existing_experiment_is_never_restarted(self):
        with environment() as (root, fingerprint, spawn):
            experiment = root / launcher.EXPERIMENT
            experiment.mkdir()
            (experiment / 'checkpoint.ckpt').write_bytes(b'preserve')
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                launcher.launch(COMMIT, fingerprint)
            self.assertEqual((experiment / 'checkpoint.ckpt').read_bytes(), b'preserve')
            spawn.assert_not_called()

    def test_changed_git_or_configuration_prevents_launch(self):
        for dirty, commit, fingerprint_override in ((True, COMMIT, None), (False, 'b'*40, None),
                                                     (False, COMMIT, 'bad-hash')):
            with self.subTest(dirty=dirty, commit=commit), environment(dirty=dirty) as (root, fingerprint, spawn):
                with self.assertRaises(RuntimeError):
                    launcher.launch(commit, fingerprint_override or fingerprint)
                self.assertFalse((root / launcher.EXPERIMENT).exists())
                spawn.assert_not_called()

    def test_fresh_launch_only_starts_the_existing_training_entrypoint(self):
        with environment() as (root, fingerprint, spawn):
            record = launcher.launch(COMMIT, fingerprint)
            arguments, options = spawn.call_args.args[0], spawn.call_args.kwargs
            self.assertEqual(arguments[1:3], ['-u', 'scripts/server/train_phone69_460h_one.py'])
            self.assertEqual(options['env']['RESUME_LAST'], '0')
            self.assertEqual(options['env']['CUDA_VISIBLE_DEVICES'], '0')
            self.assertTrue(options['pass_fds'])
            self.assertTrue(options['start_new_session'])
            self.assertEqual(record['stages'], [0, 5, 8])
            self.assertEqual(record['max_token_limit'], 11050)
            self.assertIsNone(record['resume_checkpoint'])
            self.assertEqual(json.loads((root / launcher.EXPERIMENT / 'launch.json').read_text()), record)


if __name__ == '__main__':
    unittest.main()
