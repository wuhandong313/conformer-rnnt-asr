#!/usr/bin/env python3
"""Train one phone69 experiment and write its best validation checkpoint."""

import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

import train as train_entry
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule


def main(config_path):
    os.chdir(REPO)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    paths = config.get("paths", {})
    exp_dir = paths["exp_dir"]
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(config.get("seed", 42), workers=True)
    torch.set_float32_matmul_precision(
        config.get("trainer", {}).get("matmul_precision", "high")
    )

    model = LibriSpeechRNNTModule.from_config(config)
    trainer = train_entry.build_trainer(config)

    checkpoint_path = paths.get("from_checkpoint")
    if os.environ.get("RESUME_LAST", "1") == "1":
        candidates = glob.glob(os.path.join(checkpoint_dir, "last*.ckpt"))
        if candidates:
            checkpoint_path = max(candidates, key=os.path.getmtime)
            print("Resuming from {}".format(checkpoint_path), flush=True)
    if checkpoint_path is None:
        print("Starting from scratch", flush=True)

    state_path = Path(exp_dir) / 'run.json'
    state = {
        'status': 'training', 'config': config_path,
        'config_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
        'epochs': config['training']['epochs'], 'early_stopping': False,
        'max_token_limit': config['data']['max_token_limit'],
        'train_subset': config['data']['train_subset'],
        'val_subset': config['data']['val_subset'], 'test_subset': config['data']['test_subset'],
        'tokenizer': config['tokenizer']['type'], 'resume_checkpoint': checkpoint_path,
        'internal_dim': config['model']['time_reduction_input_dim'] * config['model']['time_reduction_stride'],
        'num_heads': config['model']['num_heads'],
    }

    def save_status(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)

    class ProgressState(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            save_status(epoch=trainer.current_epoch + 1, global_step=trainer.global_step)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step % 100 == 0:
                save_status(epoch=trainer.current_epoch + 1, global_step=trainer.global_step)

    trainer.callbacks.append(ProgressState())
    save_status()
    try:
        trainer.fit(model, ckpt_path=checkpoint_path)
    except BaseException as error:
        save_status(status='failed', error=repr(error))
        raise

    validation_checkpoints = [
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ModelCheckpoint)
        and (callback.monitor or "").endswith("val_loss")
    ]
    if not validation_checkpoints or not validation_checkpoints[0].best_model_path:
        raise RuntimeError("Training ended without a best validation checkpoint")

    best_path = validation_checkpoints[0].best_model_path
    marker_path = os.path.join(exp_dir, "best_checkpoint.txt")
    temporary_path = marker_path + ".tmp"
    with open(temporary_path, "w") as stream:
        stream.write(best_path + "\n")
    os.replace(temporary_path, marker_path)
    save_status(status='trained', best_checkpoint=best_path, global_step=trainer.global_step)
    print("BEST_CKPT: {}".format(best_path), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/server/train_phone69_460h_one.py CONFIG.yaml")
    main(sys.argv[1])
