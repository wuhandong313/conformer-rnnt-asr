#!/usr/bin/env python3
"""Finish the checkpoint's current epoch with its original batch recipe."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('output')
    parser.add_argument('--state')
    args = parser.parse_args()
    os.chdir(REPO)
    import torch
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import Callback
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from train import build_trainer, load_config

    state_path = Path(args.state) if args.state else None
    def status(**updates):
        if state_path:
            state = json.loads(state_path.read_text())
            state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
            temporary = state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(state_path)

    cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not checkpoint.get('optimizer_states') or not checkpoint.get('lr_schedulers'):
        raise ValueError('Epoch finishing requires a full checkpoint')
    old_budget = checkpoint['hyper_parameters']['max_token_limit']
    if cfg['data']['max_token_limit'] != old_budget:
        raise ValueError('Finish the epoch using the checkpoint original batch budget')
    progress = checkpoint['loops']['fit_loop']['epoch_progress']['current']
    if progress['ready'] == progress['completed']:
        shutil.copy2(args.checkpoint, args.output)
        status(status='boundary_saved', boundary_checkpoint=args.output)
        return
    cfg['training']['epochs'] = min(checkpoint['epoch'] + 1, cfg['training']['epochs'])
    del checkpoint
    seed_everything(cfg['seed'], workers=True)
    torch.set_float32_matmul_precision(cfg['trainer']['matmul_precision'])
    model = LibriSpeechRNNTModule.from_config(cfg)
    trainer = build_trainer(cfg)

    class TransitionProgress(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step % 100 == 0:
                status(epoch=trainer.current_epoch + 1, global_step=trainer.global_step)

    trainer.callbacks.append(TransitionProgress())
    status(status='finishing_epoch', pid=os.getpid(), target_boundary_epoch=cfg['training']['epochs'])
    try:
        trainer.fit(model, ckpt_path=args.checkpoint)
        if trainer.interrupted:
            raise RuntimeError('Epoch finishing was interrupted')
        trainer.save_checkpoint(args.output)
        status(status='boundary_saved', boundary_checkpoint=args.output,
               completed_epochs=trainer.current_epoch, global_step=trainer.global_step)
        print('EPOCH_BOUNDARY_SAVED ' + json.dumps({
            'checkpoint': args.output, 'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
        }), flush=True)
    except BaseException as error:
        status(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
