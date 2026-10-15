"""Ensure the baseline saves the model at the validation producing its score."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from pytorch_lightning import LightningModule
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset
    from train import build_trainer
    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    RUNTIME_AVAILABLE = False


@unittest.skipUnless(RUNTIME_AVAILABLE, "Requires the training runtime")
class ValidationCheckpointTests(unittest.TestCase):
    def test_improved_validation_saves_that_step(self):
        class TinyModel(LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def training_step(self, batch, batch_idx):
                self.log("Losses/train_loss", self.weight.square(), batch_size=1)
                return self.weight.square()

            def validation_step(self, batch, batch_idx):
                self.log("Losses/val_loss", float(10 - self.global_step), batch_size=1)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        # This local-filesystem check does not need optional cloud storage SDKs.
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"s3fs": None}):
            cfg = {
                "paths": {"exp_dir": directory}, "training": {"epochs": 1},
                "model": {"compression": {"filter": "none"}},
                "trainer": {
                    "accelerator": "cpu", "devices": 1, "val_check_interval": 2,
                    "num_sanity_val_steps": 0,
                    "checkpoint": {"save_on_validation_end": True, "save_last": True,
                                   "train_checkpoint_interval": 100},
                },
            }
            trainer = build_trainer(cfg)
            self.assertFalse(any(isinstance(c, EarlyStopping) for c in trainer.callbacks))
            loader = DataLoader(TensorDataset(torch.ones(4)), batch_size=1)
            trainer.fit(TinyModel(), loader, DataLoader(TensorDataset(torch.ones(1)), batch_size=1))
            callback = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)
                            and c.monitor == "Losses/val_loss")
            self.assertEqual(float(callback.best_model_score), 6.0)
            checkpoint = torch.load(callback.best_model_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["global_step"], 4)
            self.assertTrue(Path(callback.last_model_path).is_file())
            self.assertIn("optimizer_states", checkpoint)
