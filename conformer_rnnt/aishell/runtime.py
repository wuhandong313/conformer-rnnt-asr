"""Full immutable Chinese checkpoints and strict gate validation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import random
from pathlib import Path
import uuid

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback

from .launch import source_fingerprints
from .protocol import file_sha, write_json


def finite_tensors(tensors):
    values = [torch.isfinite(t).all() for t in tensors]
    return not values or bool(torch.stack(values).all())


def validate_gate(gate_path, config_path, repo):
    gate = json.loads(Path(gate_path).read_text())
    if gate.get("valid") is not True or gate.get("phase") != "aishell1_lexicon_20261024":
        raise ValueError("A passed Chinese preflight gate is required")
    if gate["source_sha256"] != source_fingerprints(repo):
        raise ValueError("Source changed since preflight")
    record = next((r for r in gate["models"] if Path(r["config"]).resolve() == Path(config_path).resolve()), None)
    if record is None or record["config_sha256"] != file_sha(config_path) or record["status"] != "passed":
        raise ValueError("Configuration does not match a passed real-batch preflight")
    if (file_sha(record["report"]) != record["report_sha256"]
            or file_sha(Path(gate_path).parent / "regression_tests.log") != gate["tests_sha256"]):
        raise ValueError("Preflight evidence changed")
    return gate, record


def checkpoint_metadata(checkpoint, module, *, require_boundary=True):
    module.on_load_checkpoint(checkpoint)
    if not checkpoint.get("state_dict") or not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers"):
        raise ValueError("Full model/optimizer/scheduler state required")
    progress = checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]
    boundary = progress["ready"] == progress["completed"]
    if require_boundary and not boundary:
        raise ValueError("Chinese recovery requires a complete epoch boundary")
    if checkpoint["hyper_parameters"]["max_token_limit"] != module.hparams.max_token_limit:
        raise ValueError("Common batch budget changed; pair must be replanned together")
    if any(not torch.isfinite(value).all() for value in checkpoint["state_dict"].values()
           if torch.is_tensor(value) and value.is_floating_point()):
        raise ValueError("Non-finite model state")
    return {"global_step": checkpoint["global_step"], "completed_epochs": progress["completed"],
            "at_epoch_boundary": boundary, "batch_budget": module.hparams.max_token_limit}


class ImmutableCheckpoints(Callback):
    def __init__(self, directory, *, save_initial=True, interval=500, on_progress=None):
        self.directory = Path(directory)
        self.save_initial, self.interval, self.on_progress = save_initial, interval, on_progress
        self.best_score, self.best_path = None, None
        self.latest_boundary = None
        self.pending_rng = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        numpy_state = np.random.get_state()
        # Plain lists avoid NumPy-specific pickle globals in modern Torch loaders.
        numpy_serialized = [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]
        checkpoint["aishell_rng_state"] = {"python": random.getstate(), "numpy": numpy_serialized,
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(pl_module.device) if pl_module.device.type == "cuda" else None}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.pending_rng = checkpoint.get("aishell_rng_state")

    def state_dict(self):
        return {"best_score": self.best_score, "best_path": self.best_path, "latest_boundary": self.latest_boundary}

    def load_state_dict(self, state_dict):
        self.best_score, self.best_path = state_dict["best_score"], state_dict["best_path"]
        self.latest_boundary = state_dict["latest_boundary"]

    def _save(self, trainer, module, kind, *, completed=False):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{kind}-epoch={trainer.current_epoch:02d}-step={trainer.global_step}-{uuid.uuid4().hex[:12]}.ckpt"
        if kind == "best":
            self.best_path = str(path)
        if kind in {"initial", "epoch-boundary"}:
            self.latest_boundary = str(path)
        temporary = path.with_suffix(".tmp")
        trainer.save_checkpoint(str(temporary), weights_only=False)
        checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
        if completed:
            fit = checkpoint["loops"]["fit_loop"]
            batches = fit["epoch_loop.batch_progress"]["current"]
            if batches["ready"] != batches["completed"]:
                raise ValueError("Incomplete training batches at epoch boundary")
            for progress in fit["epoch_progress"].values():
                if not progress["ready"] == progress["processed"] == progress["completed"] + 1:
                    raise ValueError("Unexpected Lightning epoch progress; cannot safely advance")
                progress["completed"] += 1
            checkpoint["epoch"] += 1
        metadata = checkpoint_metadata(checkpoint, module, require_boundary=kind in {"initial", "epoch-boundary"})
        checkpoint["aishell_checkpoint"] = {"kind": kind, **metadata}
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        write_json(self.directory.parent / "latest_checkpoint.json", {"path": str(path), "metadata": metadata}, replace=True)
        if kind == "best":
            write_json(self.directory.parent / "best_checkpoint.json", {"path": str(path), "validation_loss": self.best_score}, replace=True)
        if self.on_progress:
            self.on_progress(**({"epoch_boundary_checkpoint": str(path)} if kind in {"initial", "epoch-boundary"} else {}))
        return path

    def on_fit_start(self, trainer, pl_module):
        if self.pending_rng:
            random.setstate(self.pending_rng["python"])
            numpy_state = self.pending_rng["numpy"]
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
            torch.random.set_rng_state(self.pending_rng["torch_cpu"])
            if self.pending_rng["torch_cuda"] is not None and pl_module.device.type == "cuda":
                torch.cuda.set_rng_state(self.pending_rng["torch_cuda"], pl_module.device)
            self.pending_rng = None
        if self.save_initial:
            self._save(trainer, pl_module, "initial")

    def on_before_backward(self, trainer, pl_module, loss):
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss; no numeric threshold or algorithm change applied")

    def on_after_backward(self, trainer, pl_module):
        if not finite_tensors(p.grad for p in pl_module.parameters() if p.grad is not None):
            raise ValueError("Non-finite gradient; retained prior complete snapshots")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not finite_tensors(pl_module.parameters()):
            raise ValueError("Non-finite parameter after optimizer step")
        if trainer.global_step and trainer.global_step % self.interval == 0:
            self._save(trainer, pl_module, "train")
        if self.on_progress and trainer.global_step % 100 == 0:
            observed = getattr(pl_module.model.transcriber.compression, "last_lengths", None)
            self.on_progress(epoch=trainer.current_epoch + 1, global_step=trainer.global_step,
                             observed_compression_lengths=observed)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get("Losses/val_loss")
        if value is None:
            return
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("Non-finite validation loss; preserved existing snapshots")
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self._save(trainer, pl_module, "best")

    def on_train_epoch_end(self, trainer, pl_module):
        self._save(trainer, pl_module, "epoch-boundary", completed=True)
