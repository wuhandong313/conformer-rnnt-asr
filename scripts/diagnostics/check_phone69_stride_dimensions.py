#!/usr/bin/env python3
"""Audit actual dimensions/parameter counts and stride2 forward/backward shapes."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import yaml
from conformer_rnnt.model import conformer_rnnt_model_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    records = []
    for scheme in ('stride4_none', 'stride1_cosd_learned', 'stride2_cosd_learned', 'stride4_cosd_learned'):
        path = REPO / f'configs/server_phone69_460h_{scheme}_s42_e50.yaml'
        cfg = yaml.safe_load(path.read_text())
        model = conformer_rnnt_model_from_config(cfg)
        encoder = model.transcriber
        attn = encoder.conformer1.conformer_layers[0].self_attn
        record = dict(scheme=scheme, projection=encoder.input_linear.out_features,
                      stride=encoder.time_reduction_stride, internal_dim=attn.embed_dim,
                      output_dim=encoder.output_linear.out_features, num_heads=attn.num_heads,
                      ffn_dim=cfg['model']['ffn_dim'],
                      compression_right_context=encoder.compression.right_context_length,
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      input_projection_parameters=sum(p.numel() for p in encoder.input_linear.parameters()),
                      block_parameters=sum(p.numel() for p in encoder.conformer1.parameters()) +
                                       sum(p.numel() for p in encoder.conformer2.parameters()),
                      compression_parameters=sum(p.numel() for p in encoder.compression.parameters()))
        assert record['internal_dim'] == record['projection'] * record['stride'] == 128
        assert record['output_dim'] == 256 and record['ffn_dim'] == 512 and record['num_heads'] == 8
        if scheme != 'stride4_none':
            assert encoder.compression.right_context_length == 4 // record['stride']
            assert cfg['model']['compression']['keep_rate'] == 0.27
            assert cfg['model']['compression']['weight_predictor_mode'] == 'diff5'
        if scheme == 'stride2_cosd_learned':
            for main_length in (101, 100):
                model.zero_grad(set_to_none=True)
                inputs = torch.randn(2, main_length + 4, 80)
                lengths = torch.tensor([main_length, main_length - 17])
                out, out_lengths = model.transcribe(inputs, lengths)
                assert out.shape[0] == 2 and out.shape[2] == 256
                assert (out_lengths > 0).all() and (out_lengths <= out.shape[1]).all()
                # Four appended feature frames become two preserved stride2 frames.
                expected = torch.round(torch.div(lengths, 2, rounding_mode='trunc') * 0.27).clamp(min=1).long() + 2
                assert torch.equal(out_lengths, expected), (out_lengths, expected)
                out.square().mean().backward()
                assert encoder.input_linear.weight.grad is not None
                assert torch.isfinite(encoder.input_linear.weight.grad).all()
            record['odd_even_forward_backward_passed'] = True
        records.append(record)
    assert len({r['block_parameters'] for r in records}) == 1
    assert len({r['compression_parameters'] for r in records if r['scheme'] != 'stride4_none'}) == 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(valid=True, models=records), indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
