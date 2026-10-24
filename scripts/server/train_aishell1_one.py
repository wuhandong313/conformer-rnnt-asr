#!/usr/bin/env python3
"""Formal Chinese training: explicit execution plus an unchanged passed gate."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--gate", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", help="Explicit full boundary checkpoint within this experiment")
    parser.add_argument("--lease-fd", type=int, help="Inherited shared GPU mutex from the Chinese pipeline")
    args = parser.parse_args()
    os.chdir(REPO)
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    if not args.execute or config["execution"].get("allow_training") is not True:
        raise SystemExit("Training disabled: code preparation only. Explicit authorization and a passed gate are required.")
    from conformer_rnnt.aishell.resources import verify_inherited_lease, gpu_processes
    verify_inherited_lease(args.lease_fd)
    if gpu_processes():
        raise RuntimeError("GPU became occupied; training was not started")
    import torch
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import CSVLogger
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from conformer_rnnt.aishell.protocol import file_sha, write_json
    from conformer_rnnt.aishell.runtime import ImmutableCheckpoints, checkpoint_metadata, validate_gate

    gate, record = validate_gate(args.gate, args.config, REPO)
    exp = Path(config["paths"]["exp_dir"]).resolve()
    if exp.parent != REPO or not exp.name.startswith("exp_aishell1_lexicon_"):
        raise ValueError("Experiment must stay in the independent Chinese checkout")
    if ((exp / "TRAINING_COMPLETE").exists()
            or (exp.exists() and any(p.name != "logs" for p in exp.iterdir()) and not args.resume)):
        raise ValueError("Refusing a duplicate start or overwriting an existing experiment")
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    module = LibriSpeechRNNTModule.from_config(config)
    if module.manifest_protocol != gate["data_protocol"]:
        raise ValueError("Frozen data protocol differs from preflight")
    if module.offline_native_definition != record["model_definition"]:
        raise ValueError("Model definition differs from preflight")
    resume = None
    if args.resume:
        resume = Path(args.resume).resolve()
        if not resume.is_relative_to(exp / "checkpoints"):
            raise ValueError("Resume checkpoint must belong to this experiment")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("preflight_only"):
            raise ValueError("A preflight state cannot resume formal training")
        checkpoint_metadata(checkpoint, module)
    exp.mkdir(exist_ok=True)
    (exp / "logs").mkdir(exist_ok=True)
    state = {"status": "training", "pid": os.getpid(), "config": str(Path(args.config).resolve()),
             "config_sha256": file_sha(args.config), "gate_sha256": file_sha(args.gate),
             "gate": str(Path(args.gate).resolve()),
             "data_protocol": module.manifest_protocol, "model_definition": module.offline_native_definition,
             "epochs": 50, "epoch": 0, "global_step": 0, "resume_checkpoint": str(resume) if resume else None}

    def progress(**updates):
        state.update(updates, server_utc=datetime.now(timezone.utc).isoformat())
        write_json(exp / "run.json", state, replace=True)
        with (exp / "logs/events.jsonl").open("a") as stream:
            stream.write(json.dumps(state) + "\n")

    snapshots = ImmutableCheckpoints(exp / "checkpoints", save_initial=resume is None, interval=500, on_progress=progress)
    trainer = Trainer(accelerator="cuda", devices=1, strategy="auto", precision="32-true", max_epochs=50,
                      callbacks=[snapshots], enable_checkpointing=False, num_sanity_val_steps=0,
                      gradient_clip_val=10., accumulate_grad_batches=1, val_check_interval=200,
                      log_every_n_steps=10, enable_progress_bar=False,
                      logger=CSVLogger(str(exp / "logs"), name="metrics"), default_root_dir=str(exp))
    progress()
    try:
        trainer.fit(module, ckpt_path=str(resume) if resume else None)
        if trainer.interrupted or trainer.current_epoch != 50 or not snapshots.best_path or not snapshots.latest_boundary:
            raise ValueError("Training did not finish 50 complete epochs with a best checkpoint")
        boundary = checkpoint_metadata(torch.load(snapshots.latest_boundary, map_location="cpu", weights_only=False), module)
        best = checkpoint_metadata(torch.load(snapshots.best_path, map_location="cpu", weights_only=False), module,
                                   require_boundary=False)
        if boundary["completed_epochs"] != 50:
            raise ValueError("Final complete boundary is not epoch 50")
        progress(status="trained", epoch=50, global_step=trainer.global_step,
                 best_checkpoint=snapshots.best_path, epoch_boundary_checkpoint=snapshots.latest_boundary,
                 completion_validation={"boundary": boundary, "best": best})
        (exp / "TRAINING_COMPLETE").write_text("50 epochs, full states and frozen protocol verified.\n")
    except BaseException as error:
        progress(status="interrupted" if trainer.interrupted else "failed", error=repr(error), global_step=trainer.global_step)
        raise


if __name__ == "__main__":
    main()
