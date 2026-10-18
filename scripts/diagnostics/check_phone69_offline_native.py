#!/usr/bin/env python3
"""Audit the four offline natural-compression recipes before fresh training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import yaml
from conformer_rnnt.compression import NaturalPeakCompression
from conformer_rnnt.model import conformer_rnnt_model_from_config

SCHEMES = ("stride4_none", "stride1_cosd_learned", "stride2_cosd_learned", "stride4_cosd_learned")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    records = []
    for scheme in SCHEMES:
        torch.manual_seed(42)
        path = REPO / f"configs/server_phone69_460h_offline_native_{scheme}_s42_e50.yaml"
        cfg = yaml.safe_load(path.read_text())
        model = conformer_rnnt_model_from_config(cfg).eval()
        encoder = model.transcriber
        attn = encoder.conformer1.conformer_layers[0].self_attn
        assert isinstance(encoder.compression, NaturalPeakCompression)
        assert not hasattr(encoder.compression, "keep_rate")
        assert not hasattr(encoder.compression, "right_context_length")
        assert cfg["paths"]["from_checkpoint"] is None
        assert cfg["training"]["epochs"] == 50 and cfg["seed"] == 42
        assert cfg["data"]["train_subset"] == ["train-clean-100", "train-clean-360"]
        record = dict(
            scheme=scheme, config=str(path.relative_to(REPO)),
            config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            projection=encoder.input_linear.out_features, stride=encoder.time_reduction_stride,
            internal_dim=attn.embed_dim, output_dim=encoder.output_linear.out_features,
            num_heads=attn.num_heads, ffn_dim=cfg["model"]["ffn_dim"],
            total_parameters=sum(p.numel() for p in model.parameters()),
            block_parameters=sum(p.numel() for p in encoder.conformer1.parameters())
                             + sum(p.numel() for p in encoder.conformer2.parameters()),
            compression_parameters=sum(p.numel() for p in encoder.compression.parameters()),
            batch_budget=cfg["data"]["max_token_limit"],
        )
        if not encoder.compression.passthrough:
            mixer = encoder.compression.mixer
            mode = cfg["model"]["compression"]["weight_predictor_mode"]
            inputs = {"diff5": 4, "five_frames": 5 * 128, "single_frame": 128}[mode]
            assert mixer.weight_predictor_mode == mode and mixer.window_size == 5
            assert mixer.weight_predictor[0].in_features == inputs
            expected = inputs * 32 + 32 + 32 * 5 + 5
            assert record["compression_parameters"] == expected
            record.update(weight_predictor_mode=mode, weight_predictor_input_dim=inputs,
                          weight_predictor_output_dim=5)
        assert record["projection"] * record["stride"] == record["internal_dim"] == 128
        assert (record["output_dim"], record["ffn_dim"], record["num_heads"]) == (256, 512, 8)
        x = torch.randn(2, 101, 80)
        lengths = torch.tensor([101, 83])
        output, output_lengths = model.transcribe(x, lengths)
        assert output.size(2) == 256 and (output_lengths > 0).all()
        assert (output_lengths <= output.size(1)).all()
        output.square().mean().backward()
        assert encoder.input_linear.weight.grad is not None
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        record.update(output_lengths=output_lengths.tolist(), finite_forward_backward=True)
        records.append(record)
    assert len({r["block_parameters"] for r in records}) == 1
    assert len({r["weight_predictor_mode"] for r in records[1:]}) == 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(valid=True, models=records), indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
