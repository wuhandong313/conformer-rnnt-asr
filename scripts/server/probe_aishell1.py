#!/usr/bin/env python3
"""Real local-data CUDA pressure checks; never downloads data or starts a queue."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def gpu_idle():
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if text:
        raise RuntimeError("GPU occupied; will not interrupt running processes")


def checkpoint_roundtrip(config, directory, accelerator="cuda"):
    import torch
    from pytorch_lightning import Callback, Trainer
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from conformer_rnnt.aishell.runtime import checkpoint_metadata

    records = []

    class InitialCaptured(Exception):
        pass

    class CaptureInitial(Callback):
        def __init__(self, name):
            self.name = name

        def on_fit_start(self, trainer, module):
            path = Path(directory) / (self.name + ".ckpt")
            trainer.save_checkpoint(path, weights_only=False)
            state = torch.load(path, map_location="cpu", weights_only=False)
            state["preflight_only"] = True
            torch.save(state, path)
            records.append({"path": str(path), **checkpoint_metadata(state, module)})
            # Stop before the training loop; should_stop may still allow one batch.
            raise InitialCaptured()

    first = LibriSpeechRNNTModule.from_config(config)
    a = Trainer(accelerator=accelerator, devices=1, precision="32-true", max_epochs=1, limit_train_batches=1,
                limit_val_batches=0, num_sanity_val_steps=0, callbacks=[CaptureInitial("initial-preflight")],
                enable_checkpointing=False, logger=False, enable_progress_bar=False, enable_model_summary=False)
    try:
        a.fit(first)
    except InitialCaptured:
        pass
    before = {name: value.detach().cpu().clone() for name, value in first.state_dict().items()}
    scheduler = first.warmup_lr_scheduler.state_dict()
    second = LibriSpeechRNNTModule.from_config(config)
    b = Trainer(accelerator=accelerator, devices=1, precision="32-true", max_epochs=1, limit_train_batches=1,
                limit_val_batches=0, num_sanity_val_steps=0, callbacks=[CaptureInitial("restored-preflight")],
                enable_checkpointing=False, logger=False, enable_progress_bar=False, enable_model_summary=False)
    try:
        b.fit(second, ckpt_path=records[0]["path"])
    except InitialCaptured:
        pass
    for name, value in second.state_dict().items():
        torch.testing.assert_close(value.cpu(), before[name], rtol=0, atol=0)
    if second.warmup_lr_scheduler.state_dict() != scheduler or b.global_step != 0:
        raise ValueError("Initial scheduler/global-step recovery differs")
    return records


def pressure_cases(module):
    selected = []
    for split, loader in (("train", module.train_dataloader()), ("dev", module.val_dataloader())):
        batches = loader.dataset
        rows = batches.dataset.rows
        by_id = {r["utterance_id"]: r for r in rows}
        scored = []
        for i, ids in enumerate(batches.batches):
            batch = [by_id[uid] for uid in ids]
            t = max(r["num_frames"] for r in batch)
            u = max(len(r["phones"]) for r in batch)
            scored.append((i, {"longest_T": t, "longest_U": u, "joint": len(batch) * t * (u + 1) * 218,
                               "attention": len(batch) * t * t, "batch_size": len(batch)}))
        names = ("longest_T", "longest_U", "joint", "attention", "batch_size") if split == "train" else ("joint",)
        seen = set()
        for name in names:
            i, shape = max(scored, key=lambda pair: pair[1][name])
            if i not in seen:
                selected.append((split, i, batches, shape))
                seen.add(i)
    return selected


def run(config, report_path, fraction=.9):
    import torch
    from pytorch_lightning import seed_everything
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from conformer_rnnt.aishell.launch import source_fingerprints
    from conformer_rnnt.aishell.protocol import file_sha, write_json

    gpu_idle()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA preflight is required; CPU results cannot approve A100 launch")
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    import yaml
    cfg = yaml.safe_load(Path(config).read_text())
    module = LibriSpeechRNNTModule.from_config(cfg).cuda()
    module.log = lambda *args, **kwargs: None
    report = {"status": "running", "config_sha256": file_sha(config), "source_sha256": source_fingerprints(REPO),
              "data_protocol": module.manifest_protocol, "model_definition": module.offline_native_definition,
              "parameters": sum(p.numel() for p in module.parameters()), "budget": cfg["data"]["max_token_limit"],
              "results": [], "preflight_only": True}
    total = torch.cuda.get_device_properties(0).total_memory
    if "A100" not in torch.cuda.get_device_name(0) or total < 75 * 2**30:
        raise ValueError("This launch gate requires the planned A100 80GB device")
    report["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "lightning": __import__("pytorch_lightning").__version__,
                         "gpu": torch.cuda.get_device_name(0), "device_total_bytes": total}
    for split, index, batches, shape in pressure_cases(module):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        samples = batches[(0, index)]
        batch = module._train_collate_fn(samples) if split == "train" else module._valid_collate_fn(samples)
        batch = type(batch)(*(value.cuda() for value in batch))
        module.train(split == "train")
        for _ in range(2 if split == "train" else 1):
            module.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(split == "train"):
                loss = module._step(batch, 0, split)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite real-batch loss")
                if split == "train":
                    loss.backward()
                    if any(not torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None):
                        raise ValueError("Non-finite real-batch gradient")
                    torch.nn.utils.clip_grad_norm_(module.model.parameters(), 10.)
                    module.optimizer.step()
                    if any(not torch.isfinite(p).all() for p in module.parameters()):
                        raise ValueError("Non-finite real-batch parameter after Adam update")
        torch.cuda.synchronize()
        used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"], text=True)
        item = {"split": split, "batch_id": index, **shape,
                "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
                "device_used_GiB": float(used.strip()) / 1024, "loss": float(loss.detach()),
                "observed_lengths": getattr(module.model.transcriber.compression, "last_lengths", None)}
        report["results"].append(item)
        write_json(report_path, report, replace=True)
        if max(torch.cuda.max_memory_reserved(), float(used.strip()) * 2**20) > fraction * total:
            raise MemoryError("Real-batch reserved/device-memory safety margin exceeded")
        del batch, loss, samples
    definition = module.offline_native_definition
    del module
    torch.cuda.empty_cache()
    directory = Path(report_path).parent / (Path(report_path).stem + "_checkpoints")
    directory.mkdir(exist_ok=False)
    report["checkpoint_roundtrip"] = checkpoint_roundtrip(cfg, directory)
    report["status"] = "passed"
    write_json(report_path, report, replace=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-memory-fraction", type=float, default=.9)
    parser.add_argument("--lease-fd", type=int)
    args = parser.parse_args()
    import os
    os.chdir(REPO)
    from conformer_rnnt.aishell.resources import verify_inherited_lease
    verify_inherited_lease(args.lease_fd)
    if not 0 < args.max_memory_fraction <= .9:
        parser.error("Memory margin must be positive and at most 90 percent")
    path = Path(args.out)
    if path.exists():
        raise FileExistsError("Probe reports are immutable between attempts")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args.config, path, args.max_memory_fraction)
    except BaseException as error:
        from conformer_rnnt.aishell.protocol import write_json
        report = json.loads(path.read_text()) if path.exists() else {}
        reducible = isinstance(error, MemoryError) or "out of memory" in str(error).lower()
        report.update(status="oom_or_memory_margin" if reducible else "failed", error=repr(error), traceback=traceback.format_exc())
        write_json(path, report, replace=True)
        raise


if __name__ == "__main__":
    main()
