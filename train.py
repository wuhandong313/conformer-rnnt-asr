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
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from conformer_rnnt.lightning_module import LibriSpeechRNNTModule


# ===========================================================================
# Checkpoint 命名与配置快照
# ===========================================================================


def model_ckpt_tag(config: Dict[str, Any]) -> str:
    """从训练实际使用的 config 生成 checkpoint 文件名标记。

    方向 A + C: 真实训练参数标记 —— 压缩模式 + Medusa 头数 + 冻结状态。
    与 from_config 使用同一个 config, 保证文件名反映模型真实训练模式。

    例子:
        naive + 1 head + 联合训练  -> "naive-h1-free"
        naive + 2 head + 冻结基底  -> "naive-h2-frz"
        无 Medusa (base 模型)      -> "naive-base"
    """
    model_cfg = config.get("model", {})
    comp = model_cfg.get("compression", {})
    medusa = model_cfg.get("medusa", {})

    filt = comp.get("filter", "none")
    heads = int(medusa.get("num_heads", 0))
    freeze = "frz" if medusa.get("freeze_base", False) else "free"

    # 有 Medusa heads 才区分冻结状态, 无 heads 时省略 (避免无意义后缀)
    if heads > 0:
        return f"{filt}-h{heads}-{freeze}"
    return f"{filt}-base"


class CheckpointSnapshot(Callback):
    """每次保存 checkpoint 时, 把训练配置快照内嵌进 checkpoint dict。

    解决「文件名可能和真实训练模式不符」: 在官方推荐的
    on_save_checkpoint 时机往 checkpoint 加一个 _config_snapshot key,
    保存后与 ckpt 原子共存 (不存在 sidecar 不同步问题)。
    eval 加载时读出并和当前 eval config 比对, 不一致即告警。

    兼容性: PL 1.x/2.x 的 Callback.on_save_checkpoint(trainer, pl_module,
    checkpoint) 签名一致; 不依赖 last_checkpoint_path 等时序敏感属性。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        # 只保留可序列化的关键字段, 完整 config 里可能有不可 picklable 的对象
        self.snapshot = {
            "model": {
                "compression": config.get("model", {}).get("compression", {}),
                "medusa": config.get("model", {}).get("medusa", {}),
            },
            "paths": {
                "exp_dir": config.get("paths", {}).get("exp_dir"),
                "from_checkpoint": config.get("paths", {}).get("from_checkpoint"),
            },
            "decoding": config.get("decoding", {}),
        }

    def on_save_checkpoint(self, trainer: Trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        checkpoint["_config_snapshot"] = self.snapshot


# ===========================================================================
# Config helpers
# ===========================================================================


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file '{path}' is empty.")
    return cfg


# ===========================================================================
# Trainer builder
# ===========================================================================


def build_trainer(config: Dict[str, Any]) -> Trainer:
    paths = config.get("paths", {})
    trainer_cfg = config.get("trainer", {})
    training_cfg = config.get("training", {})
    ckpt = trainer_cfg.get("checkpoint", {})

    exp_dir = paths.get("exp_dir", "./exp")
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    # 真实训练参数标记: 压缩模式 + Medusa 头数 + 冻结状态 (方向 A + C)
    tag = model_ckpt_tag(config)
    logging.info("Checkpoint tag: %s", tag)

    val_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"val-{tag}-{{epoch:02d}}-{{step}}",
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=ckpt.get("save_top_k", 2),
        save_weights_only=ckpt.get("save_weights_only", True),
        every_n_train_steps=ckpt.get("val_checkpoint_interval", 500),
        verbose=True,
    )
    train_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"train-{tag}-{{epoch:02d}}-{{step}}",
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=ckpt.get("save_top_k", 2),
        save_weights_only=ckpt.get("save_weights_only", True),
        every_n_train_steps=ckpt.get("train_checkpoint_interval", 100),
        verbose=True,
    )

    snapshot_cb = CheckpointSnapshot(config)

    return Trainer(
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        default_root_dir=exp_dir,
        max_epochs=training_cfg.get("epochs", 120),
        num_nodes=trainer_cfg.get("num_nodes", 1),
        devices=trainer_cfg.get("devices", 1),
        accelerator=trainer_cfg.get("accelerator", "cuda"),
        strategy=trainer_cfg.get("strategy", "auto"),
        gradient_clip_val=training_cfg.get("gradient_clip_val", 10.0),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        callbacks=[val_ckpt, train_ckpt, snapshot_cb],
        num_sanity_val_steps=trainer_cfg.get("num_sanity_val_steps", 0),
        val_check_interval=trainer_cfg.get("val_check_interval", 500),
        fast_dev_run=trainer_cfg.get("fast_dev_run", False),
    )


# ===========================================================================
# Main
# ===========================================================================


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
    logging.info("Building model ...")
    model = LibriSpeechRNNTModule.from_config(config)

    # 6. Build trainer
    logging.info("Building trainer ...")
    trainer = build_trainer(config)
    logging.info(
        "Trainer: accelerator=%s  devices=%s  nodes=%s  epochs=%s",
        config["trainer"].get("accelerator", "cuda"),
        config["trainer"].get("devices", 1),
        config["trainer"].get("num_nodes", 1),
        config["training"].get("epochs", 120),
    )

    # 7. Load weights if resuming from checkpoint
    ckpt_path = config.get("paths", {}).get("from_checkpoint", None)
    if ckpt_path:
        logging.info("Loading weights from: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict)

    # 8. Train
    logging.info("Starting training ...")
    trainer.fit(model)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
