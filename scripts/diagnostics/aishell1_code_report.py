#!/usr/bin/env python3
"""CPU synthetic shape/parameter evidence only; never a data or CUDA launch gate."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import os
    os.chdir(REPO)
    import torch
    import torchaudio
    from conformer_rnnt.model import conformer_rnnt_model_from_config
    from conformer_rnnt.aishell.launch import read_configs, readiness, SCHEMES
    from conformer_rnnt.aishell.protocol import write_json
    configs = read_configs([REPO / f"configs/作废_aishell1_lexicon_20261024/server_aishell1_lexicon_{s}_s42_e50.yaml" for s in SCHEMES])
    torch.manual_seed(42)
    features = torch.randn(2, 101, 80)
    lengths = torch.tensor([101, 87], dtype=torch.int32)
    targets = torch.tensor([[217, 0, 1, 2, 3], [217, 0, 1, 2, 3]], dtype=torch.int32)
    target_lengths = torch.tensor([5, 5], dtype=torch.int32)
    records = []
    for config, scheme in zip(configs, SCHEMES):
        model = conformer_rnnt_model_from_config(config).eval()
        seen = []
        handle = model.transcriber.compression.register_forward_hook(
            lambda _m, a, o: seen.append({"input": a[1].tolist(), "output": o[1].tolist()}))
        with torch.inference_mode():
            logits, output_lengths, _, _ = model(features, lengths, targets, target_lengths)
        handle.remove()
        if not torch.isfinite(logits).all() or logits.shape[-1] != 218:
            raise ValueError("Synthetic logits invalid")
        records.append({"scheme": scheme, "parameters": sum(p.numel() for p in model.parameters()),
                        "compression_parameters": sum(p.numel() for p in model.transcriber.compression.parameters()),
                        "logit_shape": list(logits.shape), "output_lengths": output_lengths.tolist(),
                        "observed_lengths_on_random_features": seen})
    write_json(args.out, {"status": "synthetic_code_checks_only", "server_or_local_utc": datetime.now(timezone.utc).isoformat(),
                         "dataset_downloaded": False, "official_data_audit_run": False, "frozen_labels_created": False,
                         "a100_preflight_run": False, "formal_training_started": False, "formal_evaluation_run": False,
                         "launch_gate_valid": False, "torch": str(torch.__version__), "torchaudio": torchaudio.__version__,
                         "models": records, "readiness": readiness(configs)})
    for record in records:
        print(record["scheme"], record["parameters"], record["compression_parameters"], record["logit_shape"])


if __name__ == "__main__":
    main()
