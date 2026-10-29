#!/usr/bin/env python3
"""Validate prepared placement recipes with synthetic RNNT forward/backward.

This never starts a dataset training job or approves a real-batch GPU budget.
It can run locally on CPU or explicitly on CUDA for a synthetic smoke check.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import torchaudio
import yaml

from common import WarmupLR
from conformer_rnnt.model import conformer_rnnt_model_from_config


VARIANTS = ("mel", "layer5", "layers2_5_n3")
CONFIGS = [REPO / f"configs/server_phone69_460h_placement_{name}_stride2_s42_e50.yaml" for name in VARIANTS]


def validate_recipe(path, device):
    config = yaml.safe_load(path.read_text())
    torch.manual_seed(config["seed"])
    model = conformer_rnnt_model_from_config(config).to(device).train()
    encoder = model.transcriber
    parameters = sum(parameter.numel() for parameter in model.parameters())
    predictor_shapes = [
        list(stage.mixer.weight_predictor[0].weight.shape)
        for stage in encoder.compression.stages if hasattr(stage.mixer, "weight_predictor")
    ]
    features = torch.randn(2, 47, config["model"]["input_dim"], device=device, requires_grad=True)
    lengths = torch.tensor([47, 35], dtype=torch.int32, device=device)
    targets = torch.tensor([[1, 2, 3], [4, 5, 69]], dtype=torch.int32, device=device)
    target_lengths = torch.tensor([3, 2], dtype=torch.int32, device=device)
    predictor_input = torch.cat([targets.new_full((2, 1), 69), targets], dim=1)
    logits, encoded_lengths, _, _ = model(features, lengths, predictor_input, target_lengths + 1)
    loss = torchaudio.transforms.RNNTLoss(blank=69, reduction="sum", clamp=1.0)(
        logits, targets, encoded_lengths, target_lengths
    )
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite RNNT loss: {path}")
    loss.backward()
    if features.grad is None or not torch.isfinite(features.grad).all():
        raise ValueError(f"Missing/non-finite input gradient: {path}")
    for parameter in model.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"Missing/non-finite model gradient: {path}")
    training = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training["learning_rate"],
        betas=tuple(training["adam_betas"]), eps=training["adam_eps"],
    )
    scheduler = WarmupLR(optimizer, training["warmup_steps"])
    torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_val"])
    optimizer.step()
    scheduler.step()
    if not optimizer.state_dict()["state"] or not scheduler.state_dict():
        raise ValueError("Missing optimizer/scheduler state after the synthetic step")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Non-finite parameter after the synthetic optimizer step")
    report = {
        "config": str(path.relative_to(REPO)),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameters": parameters, "layers": encoder.block_sizes,
        "placement_definition": encoder.placement_definition,
        "weight_predictor_first_layer_shapes": predictor_shapes,
        "logits_shape": list(logits.shape), "encoded_lengths": encoded_lengths.tolist(),
        "observed_compression_lengths": encoder.compression.last_lengths,
        "finite_rnnt_loss": float(loss.detach()),
        "finite_gradients_and_optimizer_step": True,
    }
    del scheduler, optimizer, loss, logits, model
    gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--configs", type=Path, nargs="+", default=CONFIGS)
    parser.add_argument("--out", type=Path, help="Write a new JSON report; existing reports are never overwritten")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; no validation was performed")
    torch.set_num_threads(2)
    rows = [validate_recipe(path.resolve(), torch.device(args.device)) for path in args.configs]
    report = {
        "valid": True, "scope": "synthetic_rnnt_forward_backward_optimizer_step",
        "device": args.device, "torch": torch.__version__, "torchaudio": torchaudio.__version__,
        "real_batch_gpu_memory_validated": False, "launch_ready": False, "models": rows,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
