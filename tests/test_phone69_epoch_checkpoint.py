"""A saved epoch boundary must resume the next epoch after a batch resize."""
import tempfile
from pathlib import Path
import unittest

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from pytorch_lightning import LightningModule, Trainer
    from scripts.server.phone69_epoch_checkpoint import Phone69EpochCheckpoint
    from conformer_rnnt.checkpoint_utils import phone69_resume_metadata
    AVAILABLE = True
except ModuleNotFoundError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Requires the training runtime')
class EpochCheckpointTests(unittest.TestCase):
    def test_initial_and_completed_epoch_resume_with_full_optimizer(self):
        class TinyModel(LightningModule):
            def __init__(self):
                super().__init__()
                self.save_hyperparameters({'max_token_limit': 4})
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def training_step(self, batch, batch_idx):
                return self.weight.square()

            def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
                return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))

        options = dict(accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, enable_model_summary=False)
        data = TensorDataset(torch.ones(4))
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(max_epochs=1, callbacks=[Phone69EpochCheckpoint(directory)], **options)
            trainer.fit(TinyModel(), DataLoader(data, batch_size=1))
            initial = next(Path(directory).glob('epoch-boundary-epoch=00-*.ckpt'))
            boundary = next(Path(directory).glob('epoch-boundary-epoch=01-*.ckpt'))
            zero = torch.load(initial, map_location='cpu', weights_only=False)
            self.assertEqual(phone69_resume_metadata(zero, 8)['resume_completed_epochs'], 0)
            checkpoint = torch.load(boundary, map_location='cpu', weights_only=False)
            self.assertEqual(phone69_resume_metadata(checkpoint, 8)['resume_completed_epochs'], 1)
            resumed = Trainer(max_epochs=2, **options)
            resumed.fit(TinyModel(), DataLoader(data, batch_size=2), ckpt_path=boundary)
            self.assertEqual(resumed.current_epoch, 2)
            self.assertEqual(resumed.global_step, 6)
            state = next(iter(resumed.optimizers[0].state.values()))
            self.assertEqual(int(state['step']), 6)
            self.assertEqual(resumed.lr_scheduler_configs[0].scheduler.last_epoch, 6)
            self.assertEqual(len(list(Path(directory).glob('*.ckpt'))), 2)


if __name__ == '__main__':
    unittest.main()
