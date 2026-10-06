#!/usr/bin/env python3
"""Train a Conformer-RNNT ASR model.

All configuration comes from a single YAML file. No other arguments.

Usage::

    python train.py configs/default.yaml
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict

import yaml
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from conformer_rnnt.lightning_module import LibriSpeechRNNTModule


# ═══════════════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════════════


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file '{path}' is empty.")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Trainer builder
# ═══════════════════════════════════════════════════════════════════════════


def build_trainer(config: Dict[str, Any]) -> Trainer:
    paths = config.get("paths", {})
    trainer_cfg = config.get("trainer", {})
    training_cfg = config.get("training", {})
    ckpt = trainer_cfg.get("checkpoint", {})

    exp_dir = paths.get("exp_dir", "./exp")
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    filt = config.get("model", {}).get("compression", {}).get("filter", "none")

    val_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"val-{filt}-{{epoch:02d}}-{{step}}",
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=ckpt.get("save_top_k", 2),
        save_weights_only=ckpt.get("save_weights_only", True),
        every_n_train_steps=ckpt.get("val_checkpoint_interval", 500),
        verbose=True,
    )
    train_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"train-{filt}-{{epoch:02d}}-{{step}}",
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=ckpt.get("save_top_k", 2),
        save_weights_only=ckpt.get("save_weights_only", True),
        every_n_train_steps=ckpt.get("train_checkpoint_interval", 100),
        verbose=True,
    )

    # Early stopping (config-driven; absent/disabled by default).
    # patience counts validation checks — with val_check_interval=2000 that is
    # roughly 2000*patience training steps without val_loss improvement.
    callbacks = [val_ckpt, train_ckpt]
    es_cfg = training_cfg.get("early_stopping")
    if es_cfg:
        callbacks.insert(
            0,
            EarlyStopping(
                monitor=es_cfg.get("monitor", "Losses/val_loss"),
                mode="min",
                patience=es_cfg.get("patience", 10),
                min_delta=es_cfg.get("min_delta", 0.0),
                strict=True,
                verbose=True,
            ),
        )

    return Trainer(
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        default_root_dir=exp_dir,
        max_epochs=training_cfg.get("epochs", 120),
        min_epochs=training_cfg.get("min_epochs", 0),
        num_nodes=trainer_cfg.get("num_nodes", 1),
        devices=trainer_cfg.get("devices", 1),
        accelerator=trainer_cfg.get("accelerator", "cuda"),
        strategy=trainer_cfg.get("strategy", "auto"),
        gradient_clip_val=training_cfg.get("gradient_clip_val", 10.0),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        num_sanity_val_steps=trainer_cfg.get("num_sanity_val_steps", 0),
        val_check_interval=trainer_cfg.get("val_check_interval", 500),
        fast_dev_run=trainer_cfg.get("fast_dev_run", False),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main(config_path: str) -> None:
    # 1. Load config
    config = load_config(config_path)

    # 2. Logging
    debug = config.get("debug", False)
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")

    # 3. Seed
    seed = config.get("seed", 42)
    seed_everything(seed, workers=True)
    logging.info(f"Config: {config_path}  |  seed: {seed}")

    # 4. Tensor Core optimization (A100 / H100)
    precision = config.get("trainer", {}).get("matmul_precision", "medium")
    torch.set_float32_matmul_precision(precision)
    logging.info(f"Float32 matmul precision: {precision}")

    # 5. Build model
    logging.info("Building model …")
    model = LibriSpeechRNNTModule.from_config(config)

    # 6. Build trainer
    logging.info("Building trainer …")
    trainer = build_trainer(config)
    logging.info(
        "Trainer: accelerator=%s  devices=%s  nodes=%s  epochs=%s",
        config["trainer"].get("accelerator", "cuda"),
        config["trainer"].get("devices", 1),
        config["trainer"].get("num_nodes", 1),
        config["training"].get("epochs", 120),
    )

    # 7. Train
    ckpt_path = config.get("paths", {}).get("from_checkpoint", None)
    logging.info("Starting training …")
    trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
