"""Audit fixed-stride encoder MACs from actual CPU operator tensor shapes."""

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from conformer_rnnt.model import conformer_rnnt_model_from_config
from scripts.evaluation.estimate_phone69_stride_encoder_compute import encoder_macs


class EncoderOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = func._schema.name
        count = 0
        if name in {'aten::mm', 'aten::bmm', 'aten::matmul'}:
            count = result.numel() * args[0].shape[-1]
        elif name in {'aten::addmm', 'aten::baddbmm'}:
            count = result.numel() * args[1].shape[-1]
        elif name == 'aten::linear':
            count = result.numel() * args[0].shape[-1]
        elif name in {'aten::convolution', 'aten::_convolution', 'aten::conv1d'}:
            count = result.numel() * math.prod(args[1].shape[1:])
        elif name in {'aten::_scaled_dot_product_flash_attention_for_cpu', 'aten::scaled_dot_product_attention'}:
            query, key, value = args[:3]
            count = math.prod(query.shape[:-1]) * key.shape[-2] * (query.shape[-1] + value.shape[-1])
        elif name == 'aten::_native_multi_head_attention':
            raise ValueError('Native fused MHA is not covered by this CPU operator audit')
        if count:
            self.counts[name] += count
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    records = []
    for stride in (1, 2, 4, 8):
        suffix = 'dim512' if stride == 4 else 'proj128'
        path = ROOT / f'configs/server_phone69_460h_offline_native_stride{stride}_none_{suffix}_s42_e50.yaml'
        config = yaml.safe_load(path.read_text())
        network = conformer_rnnt_model_from_config(config).cpu().eval()
        observer = EncoderOps()
        length = 1001
        with torch.no_grad(), observer:
            output, out_lengths = network.transcribe(torch.randn(1, length, 80), torch.tensor([length]))
        expected = encoder_macs(config['model'], length)
        assert sum(observer.counts.values()) == sum(expected.values()), (stride, observer.counts, expected)
        record = dict(stride=stride, mel_frames=length, encoder_frames=int(out_lengths.item()),
                      observed_macs=sum(observer.counts.values()), dispatched_operators=dict(observer.counts),
                      formula_components=expected, exact_match=True,
                      config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record)
        print(json.dumps(record), flush=True)
        del network, output
        gc.collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(status='passed', torch=torch.__version__,
        method='actual dispatched CPU matrix/convolution/attention shapes, no checkpoint or decoder',
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
