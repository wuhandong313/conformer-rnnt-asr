"""Finish an interrupted phone69 epoch before changing the batch budget.

Use the original recipe and a full checkpoint. Saving after fit returns leaves
the epoch loop at a boundary, so a different number of batches can be resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    os.chdir(REPO)
    import torch
    from pytorch_lightning import seed_everything
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from train import build_trainer, load_config

    cfg = load_config("configs/server_phone69_100h_none.yaml")
    assert cfg["data"]["max_token_limit"] == 4000
    assert cfg["training"]["epochs"] == 40 and not cfg["training"].get("early_stopping")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint.get("optimizer_states") and checkpoint.get("lr_schedulers")
    cfg["training"]["epochs"] = min(checkpoint["epoch"] + 1, 40)
    del checkpoint
    seed_everything(cfg["seed"], workers=True)
    torch.set_float32_matmul_precision(cfg["trainer"]["matmul_precision"])
    model = LibriSpeechRNNTModule.from_config(cfg)
    trainer = build_trainer(cfg)
    trainer.fit(model, ckpt_path=args.checkpoint)
    trainer.save_checkpoint(args.output)
    print("EPOCH_BOUNDARY_SAVED " + json.dumps({
        "epoch": trainer.current_epoch, "global_step": trainer.global_step,
        "checkpoint": args.output,
    }), flush=True)


if __name__ == "__main__":
    main()
