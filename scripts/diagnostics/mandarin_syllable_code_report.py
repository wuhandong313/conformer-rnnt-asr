#!/usr/bin/env python3
"""Synthetic model shapes only. No real vocabulary, data, GPU preflight or training."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import os
    os.chdir(REPO)
    import torch
    from conformer_rnnt.model import conformer_rnnt_model_from_config
    from conformer_rnnt.aishell.launch import read_configs, readiness, SCHEMES
    from conformer_rnnt.aishell.protocol import write_json
    configs = read_configs([REPO / f"configs/server_mandarin_syllable_mix_{s}_s42_e50.yaml" for s in SCHEMES])
    torch.manual_seed(42)
    records = []
    with tempfile.TemporaryDirectory() as temporary:
        vocab = Path(temporary) / "synthetic_vocab.json"
        write_json(vocab, {"version": 1, "type": "mandarin_syllable", "tokenization": "whitespace",
                          "normalization": "none_preserve_original", "tokens": ["guo2", "zhong1"]})
        features = torch.randn(2, 101, 80)
        lengths = torch.tensor([101, 87], dtype=torch.int32)
        targets = torch.tensor([[2, 1, 0], [2, 1, 0]], dtype=torch.int32)
        u = torch.tensor([3, 3], dtype=torch.int32)
        for original, scheme in zip(configs, SCHEMES):
            config = copy.deepcopy(original); config["tokenizer"]["vocab_path"] = str(vocab)
            model = conformer_rnnt_model_from_config(config).eval()
            observed = []
            hook = model.transcriber.compression.register_forward_hook(
                lambda _m, a, o: observed.append({"input": a[1].tolist(), "output": o[1].tolist()}))
            with torch.inference_mode():
                output = model(features, lengths, targets, u)[0]
            hook.remove()
            if output.shape[-1] != 3 or not torch.isfinite(output).all():
                raise ValueError("Synthetic syllable head invalid")
            parameters = sum(p.numel() for p in model.parameters())
            records.append({"scheme": scheme, "synthetic_parameters": parameters,
                "real_parameter_formula": {"constant": parameters - 385 * 3, "per_output_class": 385, "output_classes": "K+1"},
                "compression_parameters": sum(p.numel() for p in model.transcriber.compression.parameters()),
                "synthetic_logits": list(output.shape), "observed_lengths_on_random_features": observed})
    write_json(args.out, {"utc": datetime.now(timezone.utc).isoformat(), "scope": "synthetic_code_checks_only",
        "target_type": "tone_marked_syllables", "corpora": ["aishell3", "thchs30"],
        "real_vocabulary_size": None, "real_vocabulary_status": "pending_local_data_and_frozen_inventory",
        "dataset_downloaded": False, "real_data_audit_run": False, "a100_preflight_run": False,
        "formal_training_started": False, "launch_gate_valid": False, "models": records, "readiness": readiness(configs)})
    for record in records:
        print(record["scheme"], record["real_parameter_formula"], record["synthetic_logits"])


if __name__ == "__main__":
    main()
