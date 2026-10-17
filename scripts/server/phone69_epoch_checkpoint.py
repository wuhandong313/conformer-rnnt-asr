"""Preserve full, immutable epoch boundaries for safe batch-budget recovery."""
from pathlib import Path
import uuid

import torch
from pytorch_lightning.callbacks import Callback

from conformer_rnnt.checkpoint_utils import phone69_resume_metadata


class Phone69EpochCheckpoint(Callback):
    def __init__(self, directory, save_initial=True, on_saved=None):
        self.directory = Path(directory)
        self.save_initial = save_initial
        self.on_saved = on_saved

    def _save(self, trainer, pl_module, completed_epoch=False):
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f'epoch-boundary-step={trainer.global_step}-{uuid.uuid4().hex[:12]}'
        temporary = self.directory / (stem + '.tmp')
        trainer.save_checkpoint(temporary, weights_only=False)
        checkpoint = torch.load(temporary, map_location='cpu', weights_only=False)
        if completed_epoch:
            # Lightning invokes on_train_epoch_end immediately before its
            # completed counter advances. Materialize the equivalent post-epoch
            # state on disk, without changing the live trainer's loop state.
            fit = checkpoint['loops']['fit_loop']
            batches = fit['epoch_loop.batch_progress']['current']
            assert batches['ready'] == batches['completed']
            for progress in fit['epoch_progress'].values():
                assert progress['ready'] == progress['processed'] == progress['completed'] + 1
                progress['completed'] += 1
            checkpoint['epoch'] += 1
        metadata = phone69_resume_metadata(checkpoint, pl_module.hparams.max_token_limit)
        assert metadata['resume_at_epoch_boundary']
        checkpoint['phone69_epoch_boundary'] = metadata
        destination = self.directory / (
            f'epoch-boundary-epoch={metadata["resume_completed_epochs"]:02d}-'
            f'step={trainer.global_step}-{uuid.uuid4().hex[:12]}.ckpt'
        )
        torch.save(checkpoint, temporary)
        temporary.replace(destination)
        if self.on_saved:
            self.on_saved(str(destination), metadata)

    def on_fit_start(self, trainer, pl_module):
        if self.save_initial:
            self._save(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self._save(trainer, pl_module, completed_epoch=True)
