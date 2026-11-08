"""Weight-independent major encoder MACs for the fixed-stride 128-D sweep.

Uses the exact utterance lengths in the frozen test-clean manifest, with no
checkpoint, decoding, audio inference or server work. Optional mock forwards
check the formulas against the existing execution observer on random weights.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARCHIVE = ROOT / 'papers/phone69_460h_cpu_authoritative_20261108/supporting/original_archive'
CATEGORIES = ('encoder_weights', 'attention_products', 'projections')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoder_macs(model, mel_frames):
    """Dense batch-one forward; ceil padding is included in encoder length."""
    if model['sequence_mode'] != 'offline' or model['compression'] != {'filter': 'none'}:
        raise ValueError('Expected the offline fixed-stride, no-compression encoder')
    if type(mel_frames) is not int or mel_frames <= 0:
        raise ValueError('Expected a positive integer Mel-frame count')
    projection, stride = model['time_reduction_input_dim'], model['time_reduction_stride']
    width = projection * stride
    steps = (mel_frames + stride - 1) // stride
    layers = model['num_layers_block1'] + model['num_layers_block2']
    coefficient = 4 * width * model['ffn_dim'] + 7 * width**2 + width * model['depthwise_conv_kernel_size']
    return dict(encoder_weights=layers * steps * coefficient,
                attention_products=layers * 2 * width * steps**2,
                projections=mel_frames * model['input_dim'] * projection + steps * width * model['encoding_dim'])


def verify_mock(config):
    import torch
    from conformer_rnnt.model import conformer_rnnt_model_from_config
    from measure_phone69_460h_compute import observe

    torch.set_num_threads(4)
    torch.manual_seed(42)
    network = conformer_rnnt_model_from_config(config).cpu().eval()
    counts, frames = Counter(), []
    observe(SimpleNamespace(model=network), counts, frames)
    checks = []
    with torch.inference_mode():
        for length in (101, 1001):
            for content in ('zeros', 'random'):
                inputs = (torch.zeros if content == 'zeros' else torch.randn)(1, length, config['model']['input_dim'])
                counts.clear()
                frames.clear()
                output, lengths = network.transcribe(inputs, torch.tensor([length], dtype=torch.int32))
                expected = encoder_macs(config['model'], length)
                assert all(counts[key] == expected[key] for key in CATEGORIES)
                steps = (length + config['model']['time_reduction_stride'] - 1) // config['model']['time_reduction_stride']
                assert frames == [steps] * 10 and lengths.tolist() == [steps]
                assert output.shape == (1, steps, config['model']['encoding_dim'])
                assert torch.isfinite(output).all()
                assert counts['predictor'] == counts['joiner'] == counts['fusion_mlp'] == 0
                checks.append(dict(mel_frames=length, content=content, encoder_steps=steps,
                                   components_macs=expected, observed_exact_match=True))
    del network
    gc.collect()
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-mock', action='store_true')
    args = parser.parse_args()
    manifest_path = ARCHIVE / 'manifest.json'
    authority = json.loads((ARCHIVE / 'AUTHORITATIVE.json').read_text())
    assert sha(manifest_path) == authority['original_verification']['manifest_sha256']
    audio = json.loads(manifest_path.read_text())['audio']
    assert len(audio) == 2620 and all(x['sample_rate'] == 16000 for x in audio)
    seconds = math.fsum(x['samples'] / x['sample_rate'] for x in audio)
    baseline_path = ROOT / 'artifacts/research/phone69_460h_compute_20261108/full/E09.json'
    baseline = json.loads(baseline_path.read_text())
    models = []
    for stride in (1, 2, 4, 8):
        suffix = 'dim512' if stride == 4 else 'proj128'
        path = ROOT / f'configs/server_phone69_460h_offline_native_stride{stride}_none_{suffix}_s42_e50.yaml'
        config = yaml.safe_load(path.read_text())
        m, d = config['model'], config['data']
        assert m['time_reduction_input_dim'] == 128 and m['time_reduction_stride'] == stride
        assert m['num_layers_block1'] + m['num_layers_block2'] == 10
        assert (d['sample_rate'], d['hop_length'], d['n_fft']) == (16000, 160, 400)
        total = Counter()
        lengths = [x['samples'] // d['hop_length'] + 1 for x in audio]
        for index, (item, length) in enumerate(zip(audio, lengths, strict=True)):
            counts = encoder_macs(m, length)
            total.update(counts)
            if stride == 4:
                reference = baseline['rows'][index]
                assert reference['uid'] == item['uid']
                assert all(reference[k] == counts[k] for k in CATEGORIES)
        components = {k: total[k] for k in CATEGORIES}
        examples = []
        for duration in (1, 5, 10, 20, 30):
            length = duration * d['sample_rate'] // d['hop_length'] + 1
            values = encoder_macs(m, length)
            examples.append(dict(audio_seconds=duration, mel_frames=length,
                                 encoder_frames=(length + stride - 1) // stride,
                                 total_gmac=sum(values.values()) / 1e9,
                                 gmac_per_audio_second=sum(values.values()) / 1e9 / duration))
        record = dict(stride=stride, input_projection=128, encoder_width=128 * stride, encoder_layers=10,
                      config=str(path.relative_to(ROOT)), config_sha256=sha(path),
                      utterances=len(audio), input_mel_frames=sum(lengths),
                      output_frames=sum((n + stride - 1) // stride for n in lengths),
                      audio_seconds=seconds, components_macs=components, total_macs=sum(components.values()),
                      encoder_gmac_per_audio_second=sum(components.values()) / seconds / 1e9,
                      examples=examples)
        if args.verify_mock:
            record['mock_forward_verification'] = verify_mock(config)
        models.append(record)
        print(json.dumps({k: record[k] for k in ('stride', 'encoder_width', 'encoder_gmac_per_audio_second')}), flush=True)
    reference_cost = next(r['encoder_gmac_per_audio_second'] for r in models if r['stride'] == 4)
    for model in models:
        model['encoder_ratio_to_stride4'] = model['encoder_gmac_per_audio_second'] / reference_cost
    report = dict(status='verified', method='exact structural major-MAC count from frozen corpus lengths',
                  scope='encoder forward including input/output projections; no predictor/joiner, training or feature extraction',
                  checkpoint_required=False, batch_size=1, subset='test-clean',
                  normalization='sum(utterance encoder MACs) / sum(actual FLAC seconds) / 1e9',
                  manifest_sha256=sha(manifest_path), baseline_source_sha256=sha(baseline_path),
                  script_sha256=sha(__file__),
                  encoder_source_sha256=sha(ROOT / 'conformer_rnnt/encoder.py'),
                  stride4_archived_utterances_exactly_matched=2620, models=models)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
