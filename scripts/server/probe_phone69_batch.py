"""Measure real training memory on the largest batches before a budget change.

Writes JSON to the supplied output; checkpoint weights/optimizer are read only.
Selects batches by actual FLAC duration, target length and utterance count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("budget", type=int)
    parser.add_argument("output")
    args = parser.parse_args()
    os.chdir(REPO)
    import torch
    from pytorch_lightning import seed_everything
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from train import load_config

    cfg = load_config("configs/server_phone69_100h_none.yaml")
    cfg["data"].update(max_token_limit=args.budget, num_workers=0)
    seed_everything(cfg["seed"], workers=True)
    torch.set_float32_matmul_precision(cfg["trainer"]["matmul_precision"])
    model = LibriSpeechRNNTModule.from_config(cfg)
    dataset = model.train_dataloader().dataset.datasets[0]
    base = dataset.base_dataset
    lengths, frames, texts = [], [], {}
    for fileid in base._walker:
        speaker, chapter, _ = fileid.split("-")
        with (Path(base._path) / speaker / chapter / (fileid + ".flac")).open("rb") as f:
            header = f.read(26)
        assert header[:4] == b"fLaC" and header[4] & 127 == 0
        samples = int.from_bytes(header[18:26], "big") & ((1 << 36) - 1)
        frames.append((samples // cfg["data"]["hop_length"] + 1 + 3) // 4 + 4)
        lengths.append(len(model.tokenizer.encode(dataset._transcript(fileid, texts))))
    scores = []
    for batch in dataset.batches:
        b, t, u = len(batch), max(frames[i] for i in batch), max(lengths[i] for i in batch)
        scores.append((b * t * t, b * t * (u + 1), b, t))
    selected = sorted({max(range(len(scores)), key=lambda i: scores[i][axis]) for axis in range(4)})
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().train()
    model.optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    del checkpoint
    model.log = lambda *a, **kw: None
    results = []
    report = {"budget": args.budget, "batches_per_epoch": len(dataset),
              "utterances": sum(map(len, dataset.batches)), "selected_batches": selected,
              "results": results, "status": "running"}
    def save():
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        for index in selected:
            model.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            cpu_batch = dataset[index]
            batch = model._train_collate_fn(cpu_batch)
            batch = type(batch)(*(v.cuda() for v in batch))
            del cpu_batch
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            loss = model._step(batch, index, "train")
            assert bool(torch.isfinite(loss)), "Non-finite loss"
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            assert bool(torch.isfinite(norm)), "Non-finite gradient"
            model.optimizer.step()
            torch.cuda.synchronize()
            result = {"batch_index": index, "utterances": len(batch.targets),
                      "phones": int(batch.target_lengths.sum()),
                      "max_feature_frames": int(batch.feature_lengths.max()),
                      "max_target_phones": int(batch.target_lengths.max()),
                      "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                      "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
                      "seconds": time.monotonic() - started, "loss": float(loss)}
            results.append(result)
            print(json.dumps(result), flush=True)
            del loss, batch
            save()
        report["status"] = "passed"
    except torch.cuda.OutOfMemoryError as error:
        report.update(status="oom", error=str(error))
    finally:
        save()
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
