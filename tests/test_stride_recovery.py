"""Local safety checks for restart recovery; no training or server access."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'scripts/server/recover_phone69_stride_sweep.py'
spec = importlib.util.spec_from_file_location('stride_recovery', MODULE)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class StrideRecoveryTests(unittest.TestCase):
    def test_boundary_order_uses_completed_epoch_and_step(self):
        names = ['epoch-boundary-epoch=09-step=9000-abcd.ckpt',
                 'epoch-boundary-epoch=10-step=10000-1234.ckpt',
                 'epoch-boundary-step=11000.tmp']
        self.assertEqual(sorted(names, key=recovery.boundary_key)[-1], names[1])
        self.assertEqual(recovery.boundary_key(names[2]), (-1, -1))

    def test_source_identity_accepts_new_commit_only_when_source_bytes_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audit = root / 'audit'
            audit.mkdir()
            source = root / 'train.py'
            source.write_text('original training implementation')
            (audit / 'deployment.json').write_text(json.dumps({
                'code_commit': 'original', 'source_sha256': {'train.py': recovery.sha(source)}}))
            with patch.object(recovery, 'ROOT', root), patch.object(recovery, 'ORIGINAL', audit), \
                    patch.object(recovery, 'commit', return_value='new-docs-commit'):
                self.assertTrue(recovery.source_identity()['original_source_files_unchanged'])
                source.write_text('changed training implementation')
                with self.assertRaisesRegex(ValueError, 'training source/configuration changed'):
                    recovery.source_identity()

    def test_original_checkpoint_survives_top_k_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            exp = root / 'experiment'
            checkpoints = exp / 'checkpoints'
            checkpoints.mkdir(parents=True)
            original = checkpoints / 'old_best.ckpt'
            original.write_bytes(b'immutable original checkpoint')
            with patch.object(recovery, 'OUT', root / 'recovery'), \
                    patch.object(recovery, 'experiment', return_value=exp):
                recovery.preserve_checkpoints(8)
                original.unlink()
                preserved = recovery.OUT / 'original_checkpoints/stride8/old_best.ckpt'
                self.assertEqual(preserved.read_bytes(), b'immutable original checkpoint')

    def test_trained_label_cannot_hide_missing_final_epoch_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            exp = Path(folder)
            (exp / 'run.json').write_text(json.dumps({
                'status': 'trained', 'epoch': 50, 'max_token_limit': 9700,
                'internal_dim': 1024, 'config_sha256': 'expected',
                'best_checkpoint': str(exp / 'missing.ckpt')}))
            with patch.object(recovery, 'experiment', return_value=exp), \
                    patch.object(recovery, 'latest_boundary', return_value=(exp / 'boundary.ckpt',
                        {'resume_completed_epochs': 49})):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    recovery.completed(8, {'budget': 9700}, {'expected'})


if __name__ == '__main__':
    unittest.main()
