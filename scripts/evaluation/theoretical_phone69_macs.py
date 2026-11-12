"""Fixed-duration major MACs calibrated only by observed stage retention.

No audio decoding, model forward, checkpoint load or timing in report mode.
Archive IDs retain their historical meanings; paper IDs are separate fields.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal, ROUND_FLOOR
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

import yaml

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'papers/phone69_460h_cpu_authoritative_20261108/supporting/original_archive'
COUNTS = ROOT / 'artifacts/research/phone69_460h_compute_20261108/full'
SWEEP = ROOT / 'artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111'
TRIPLE = ROOT / 'artifacts/evaluations/phone69_460h_mel_layer5_layer8_local_cpu_20261108'
DEFAULT_OUT = ROOT / 'artifacts/research/phone69_fixed_duration_20261112'
ENCODER_KEYS = ('encoder_weights', 'attention_products', 'projections',
                'fusion_mlp', 'detector_dots', 'fusion_weighted_sum')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def layers(model):
    return model.get('num_layers', model.get('num_layers_block1', 0)
                     + model.get('num_layers_block2', 0))


def stages(model):
    if model.get('sequence_mode') != 'offline':
        raise ValueError('Only whole-utterance offline models are supported')
    comp = model['compression']
    if comp == {'filter': 'none'}:
        return {}
    items = comp.get('stages')
    if items is None:
        items = [dict(after_layer=model['num_layers_block1'], **comp)]
    result = {}
    for item in items:
        site = item['after_layer']
        if site in result or not 0 <= site <= layers(model):
            raise ValueError('Invalid compression placement')
        if item['detector']['type'] != 'cosine_diff':
            raise ValueError('Unsupported detector')
        if item['detector'].get('preprocess', 'none') != 'none':
            raise ValueError('Preprocessing requires its own MAC formula')
        kind = item['mixer']['type']
        if kind not in ('learned', 'hard'):
            raise ValueError('Unsupported mixer')
        if kind == 'learned' and (item.get('window_size', 5) != 5
                                 or item.get('weight_predictor_mode') != 'five_frames'):
            raise ValueError('Only five_frames fusion is counted')
        result[site] = item
    return result


def mel_frames(seconds, sample_rate=16000, hop_length=160):
    """Existing centered STFT: floor(samples / hop) + 1; no fake padding."""
    seconds = Decimal(str(seconds))
    if not seconds.is_finite() or seconds <= 0:
        raise ValueError('Duration must be finite and positive')
    samples = int((seconds * sample_rate).to_integral_value(rounding=ROUND_FLOOR))
    if samples <= 0:
        raise ValueError('Duration is shorter than one sample')
    return samples // hop_length + 1


def frame_trace(model, raw, retained):
    """retained(site, input_length) -> output_length, at the actual stage site."""
    if type(raw) is not int or raw < 1:
        raise ValueError('Expected positive integer frames')
    schedule, trace, frame_counts = stages(model), [], []
    current = raw
    for site in range(layers(model) + 1):
        if site in schedule:
            output = retained(site, current)
            if type(output) is not int or not 1 <= output <= current:
                raise ValueError('Invalid stage length')
            trace.append(dict(after_layer=site, input_frames=current, output_frames=output))
            current = output
        if site == 0:
            projected = current
            stride = model['time_reduction_stride']
            current = (current + stride - 1) // stride
        if site < layers(model):
            frame_counts.append(current)
    return dict(projected_frames=projected, layer_frames=frame_counts,
                output_frames=current, stages=trace)


def observed_trace(model, raw, row):
    def retain(site, current):
        if site == 0 and model['time_reduction_stride'] != 1:
            raise ValueError('Mel-stage output cannot be recovered after stride > 1')
        return row['layer_frames'][site] if site < layers(model) else row['output_steps']
    trace = frame_trace(model, raw, retain)
    if trace['layer_frames'] != row['layer_frames'] or trace['output_frames'] != row['output_steps']:
        raise ValueError('Frame trace does not match the architecture')
    return trace


def calibrate(model, raw_lengths, rows):
    if not rows or len(rows) != len(raw_lengths):
        raise ValueError('Incomplete calibration corpus')
    inputs, outputs, ratios = Counter(), Counter(), {site: [] for site in stages(model)}
    for raw, row in zip(raw_lengths, rows):
        for stage in observed_trace(model, raw, row)['stages']:
            site, n, k = stage['after_layer'], stage['input_frames'], stage['output_frames']
            inputs[site] += n
            outputs[site] += k
            ratios[site].append(k / n)
    return [dict(after_layer=site, input_frames=inputs[site], output_frames=outputs[site],
                 retention=outputs[site] / inputs[site], reduction=1 - outputs[site] / inputs[site],
                 utterance_ratio_mean=statistics.mean(ratios[site]),
                 utterance_ratio_std=statistics.pstdev(ratios[site]),
                 utterances=len(rows)) for site in sorted(ratios)]


def projected_trace(model, raw, rates):
    rates = {r['after_layer']: Fraction(r['output_frames'], r['input_frames']) for r in rates}
    if set(rates) != set(stages(model)) or any(not 0 < r <= 1 for r in rates.values()):
        raise ValueError('Missing or invalid stage retention')

    def retain(site, current):
        value = current * rates[site]
        # Nearest integer, ties up. Use integer arithmetic for reproducibility.
        rounded = (2 * value.numerator + value.denominator) // (2 * value.denominator)
        return min(current, max(1, rounded))
    return frame_trace(model, raw, retain)


def encoder_macs(model, trace):
    d = model['time_reduction_input_dim'] * model['time_reduction_stride']
    lengths = trace['layer_frames']
    per_frame = 4 * d * model['ffn_dim'] + 7 * d * d + d * model['depthwise_conv_kernel_size']
    out = dict(encoder_weights=per_frame * sum(lengths),
               attention_products=2 * d * sum(t * t for t in lengths),
               projections=model['input_dim'] * model['time_reduction_input_dim'] * trace['projected_frames']
               + d * model['encoding_dim'] * trace['output_frames'],
               fusion_mlp=0, detector_dots=0, fusion_weighted_sum=0)
    schedule = stages(model)
    for stage in trace['stages']:
        site, n, k = stage['after_layer'], stage['input_frames'], stage['output_frames']
        if n <= 1:  # Actual compressor bypasses the detector and mixer.
            continue
        width = model['input_dim'] if site == 0 else d
        out['detector_dots'] += (n - 1) * width
        if schedule[site]['mixer']['type'] == 'learned':
            w = schedule[site].get('window_size', 5)
            out['fusion_mlp'] += k * (w * width * 32 + 32 * w)
            out['fusion_weighted_sum'] += k * w * width
    return out


def reference_decoder_macs(model, output_frames, target_phones):
    """One prescribed monotone path, cached predictor; NOT beam-search cost."""
    if type(target_phones) is not int or target_phones < 0:
        raise ValueError('Target phone count must be a nonnegative integer')
    p = model['predictor']
    h, emb, count = p['lstm_hidden_dim'], p['symbol_embedding_dim'], p['num_lstm_layers']
    per_token = 4 * h * (emb + h) + (count - 1) * 8 * h * h + h * model['encoding_dim']
    return dict(predictor=(target_phones + 1) * per_token,
                joiner=(output_frames + target_phones) * model['encoding_dim'] * (model['num_symbols'] + 1))


def inventory(extra_calibration=None):
    manifest = read(ARCHIVE / 'manifest.json')
    audio = manifest['audio']
    uids = [a['uid'] for a in audio]
    raw = [a['samples'] // 160 + 1 for a in audio]
    if len(uids) != 2620 or len(set(uids)) != 2620 or any(a['sample_rate'] != 16000 for a in audio):
        raise ValueError('Unexpected reference corpus')
    aliases = {m['archive_id']: m['id'] for m in read(ROOT / 'papers/extremafuse_acl2027/compute_snapshot.json')['models']}
    entries = []
    for item in manifest['models']:
        folder = ARCHIVE / item['id']
        count_path = COUNTS / (item['id'] + '.json')
        result = read(count_path)
        if (result['config_sha256'] != sha(folder / 'config.yaml')
                or result['checkpoint_sha256'] != item['checkpoint_sha256']
                or result['provenance']['sample_per_quartile'] is not None
                or result['provenance']['manifest_sha256'] != sha(ARCHIVE / 'manifest.json')):
            raise ValueError('Calibration identity differs from archive')
        entries.append(dict(id=item['id'], paper_id=aliases.get(item['id']), name=item['name'],
                            config_path=folder / 'config.yaml', checkpoint_sha256=item['checkpoint_sha256'],
                            rows=result['rows'], source=count_path,
                            per_percent=100 * read(folder / 'result.json')['corpus_PER']))
    for stride in (1, 2, 8):
        folder = SWEEP / f'stride{stride}_proj128'
        result = read(folder / 'attempt_01/result.json')
        path = folder / 'attempt_01/utterances.csv'
        if sha(path) != result['csv_sha256'] or sha(folder / 'config.yaml') != result['model']['config_sha256']:
            raise ValueError('Stride inputs changed')
        with path.open() as f:
            rows = [dict(uid=r['uid'], layer_frames=json.loads(r['layer_frames']),
                         output_steps=json.loads(r['layer_frames'])[-1]) for r in csv.DictReader(f)]
        entries.append(dict(id=f'S{stride}', paper_id=None, name=f'stride{stride}_proj128',
                            config_path=folder / 'config.yaml', checkpoint_sha256=result['model']['checkpoint_sha256'],
                            rows=rows, source=path, per_percent=100 * result['corpus_PER']))
    triple = read(TRIPLE / 'result.json')
    extra = read(extra_calibration) if extra_calibration else None
    if extra and (extra['checkpoint_sha256'] != triple['checkpoint_sha256']
                  or extra['config_sha256'] != sha(TRIPLE / 'config.yaml')):
        raise ValueError('Three-stage calibration uses another checkpoint/config')
    entries.append(dict(id='E11', paper_id=None, name='placement_mel_layer5_layer8_stride1',
                        config_path=TRIPLE / 'config.yaml', checkpoint_sha256=triple['checkpoint_sha256'],
                        rows=extra['rows'] if extra else None, source=Path(extra_calibration) if extra else None,
                        per_percent=100 * triple['corpus_PER']))
    for item in read(ROOT / 'configs/phone69_placement_supplement_20261112.json')['experiments']:
        if item['action'] == 'train':
            entries.append(dict(id='supp_' + item['name'], name=item['name'], paper_id=None,
                                config_path=ROOT / item['config'], checkpoint_sha256=None,
                                rows=None, source=None, per_percent=None))
    for entry in entries:
        config = yaml.safe_load(entry['config_path'].read_text())
        if (config['data']['sample_rate'], config['data']['hop_length']) != (16000, 160):
            raise ValueError('Inconsistent feature extraction')
        entry['model'] = config['model']
        entry['config_sha256'] = sha(entry['config_path'])
        if entry['rows'] is not None:
            if [r['uid'] for r in entry['rows']] != uids:
                raise ValueError('Calibration must cover the full frozen corpus in order')
            entry['rates'] = calibrate(entry['model'], raw, entry['rows'])
    return entries, manifest, raw


def generate(durations, extra_calibration=None, target_phones=None):
    entries, manifest, raw = inventory(extra_calibration)
    seconds = math.fsum(a['samples'] / a['sample_rate'] for a in manifest['audio'])
    models, table = [], []
    for entry in entries:
        record = {k: entry[k] for k in ('id', 'name', 'paper_id', 'checkpoint_sha256', 'per_percent', 'config_sha256')}
        record['config'] = str(entry['config_path'].relative_to(ROOT))
        record['model'] = entry['model']
        if entry['rows'] is None:
            record.update(status='pending_stage_retention', reason='No full-corpus stage lengths for this checkpoint', results=[])
        else:
            source = entry['source']
            record.update(status='calibrated_theory', calibration_source=str(source.relative_to(ROOT)),
                          calibration_sha256=sha(source), calibration_utterances=2620,
                          stage_rates=entry['rates'], results=[])
            for duration in durations:
                n = mel_frames(duration)
                trace = projected_trace(entry['model'], n, entry['rates'])
                parts = encoder_macs(entry['model'], trace)
                u = target_phones if target_phones is not None else math.ceil(189023 * duration / seconds)
                decoder = reference_decoder_macs(entry['model'], trace['output_frames'], u)
                encoder = sum(parts.values())
                result = dict(duration_seconds=duration, input_mel_frames=n, target_phones=u, trace=trace,
                              components_macs=parts, encoder_compression_gmac=encoder / 1e9,
                              encoder_compression_gmac_per_audio_s=encoder / duration / 1e9,
                              reference_decoder_macs=decoder,
                              reference_path_total_gmac=(encoder + sum(decoder.values())) / 1e9,
                              reference_path_total_gmac_per_audio_s=(encoder + sum(decoder.values())) / duration / 1e9,
                              actual_beam20_macs=None)
                record['results'].append(result)
                table.append(dict(id=record['id'], name=record['name'], duration_s=duration,
                                  input_frames=n, output_frames=trace['output_frames'], target_phones=u,
                                  encoder_compression_gmac=result['encoder_compression_gmac'],
                                  encoder_compression_gmac_per_audio_s=result['encoder_compression_gmac_per_audio_s'],
                                  reference_predictor_gmac=decoder['predictor'] / 1e9,
                                  reference_joiner_gmac=decoder['joiner'] / 1e9,
                                  reference_path_total_gmac=result['reference_path_total_gmac'],
                                  reference_path_total_gmac_per_audio_s=result['reference_path_total_gmac_per_audio_s']))
        models.append(record)
    return dict(schema=1, method='fixed-duration structure with pooled stage retention',
                manifest_sha256=sha(ARCHIVE / 'manifest.json'), script_sha256=sha(__file__),
                primary_duration_seconds=30, durations_seconds=durations, calibration_subset='test-clean',
                calibration_utterances=2620, calibration_audio_seconds=seconds,
                common_reference_phones=189023, target_phones_override=target_phones,
                decoder='prescribed single monotone path, U+1 cached predictor tokens, T_encoder+U joiner rows; not beam20',
                models=models), table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seconds', type=float, nargs='+', default=[1, 5, 10, 20, 30])
    parser.add_argument('--stage-calibration', type=Path)
    parser.add_argument('--target-phones', type=int, help='Common U for every model; default ceil(corpus phones/s * duration)')
    args = parser.parse_args()
    if any(not math.isfinite(t) or t <= 0 for t in args.seconds):
        parser.error('Durations must be finite and positive')
    report, rows = generate(args.seconds, args.stage_calibration, args.target_phones)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'models_calculated': sum(bool(m['results']) for m in report['models']),
                      'pending': [m['id'] for m in report['models'] if not m['results']], 'output': str(args.out)}))


if __name__ == '__main__':
    main()
