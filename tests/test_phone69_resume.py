"""Guard batch changes against incomplete checkpoints and active experiments."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from scripts.server.run_phone69_100h_none import resume_metadata
    AVAILABLE = True
except ModuleNotFoundError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Requires YAML")
class Phone69ResumeTests(unittest.TestCase):
    def check_resume(self, progress, optimizer=True):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            state.write_text(json.dumps({"status": "failed", "max_token_limit": 4000}))
            cfg = {"paths": {"from_checkpoint": "boundary.ckpt"},
                   "data": {"max_token_limit": 8000}}
            checkpoint = {"optimizer_states": [{}] if optimizer else [], "lr_schedulers": [{}],
                          "epoch": 2, "global_step": 2682,
                          "loops": {"fit_loop": {"epoch_progress": {"current": progress}}}}
            with patch.dict("sys.modules", {"torch": SimpleNamespace(load=lambda *a, **kw: checkpoint)}):
                return resume_metadata(state, cfg)

    def test_boundary_preserves_resume_offsets(self):
        previous, metadata = self.check_resume({"ready": 3, "completed": 3})
        self.assertEqual(previous["max_token_limit"], 4000)
        self.assertEqual(metadata["resume_epoch"], 3)
        self.assertEqual(metadata["resume_global_step"], 2682)

    def test_mid_epoch_budget_change_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "epoch boundary"):
            self.check_resume({"ready": 3, "completed": 2})

    def test_weights_only_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            self.check_resume({"ready": 3, "completed": 3}, optimizer=False)

    def test_existing_run_needs_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps({"status": "failed"}))
            with self.assertRaisesRegex(RuntimeError, "explicit resume"):
                resume_metadata(path, {"paths": {"from_checkpoint": None}})
