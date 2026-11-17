#!/usr/bin/env python3
"""Independently verify completed placement CSVs and calibrate 30-s MACs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

from eval_phone69_placement_checkpoint import COMPONENTS, now, sha, verified
from theoretical_phone69_macs import calibrate, encoder_macs, mel_frames, projected_trace

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'artifacts/evaluations/phone69_placement_l1_l3_l9_local_cpu_20261117'


def edit_distance(reference, hypothesis):
    """Rolling-row Levenshtein distance over phone tokens, independent of Torch."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def verify_model(root, name, attempt_name='attempt_01'):
    """Verify a preserved evaluation attempt; the original default is unchanged."""
    folder = root / name
    if Path(attempt_name).name != attempt_name or not attempt_name.startswith('attempt_'):
        raise ValueError('Expected an evaluation attempt directory name')
    attempt = folder / attempt_name
    manifest = json.loads((folder / 'manifest.json').read_text())
    result = json.loads((attempt / 'result.json').read_text())
    verified(attempt / 'utterances.csv', result['csv_sha256'])
    verified(folder / 'manifest.json', result['manifest_sha256'])
    verified(folder / 'config.yaml', manifest['model']['config_sha256'])
    verified(folder / 'training_config.yaml', manifest['model']['training_config_sha256'])
    for relative, digest in manifest['code_sha256'].items():
        verified(folder / 'model_source' / relative, digest)
    checkpoint = Path(manifest['model']['checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError('Verify assets on the evaluation machine: ' + str(checkpoint))
    verified(checkpoint, manifest['model']['checkpoint_sha256'])
    with (attempt / 'utterances.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2620 or [r['uid'] for r in rows] != [a['uid'] for a in manifest['audio']]:
        raise ValueError('Incomplete or reordered test-clean records')
    reference = Path(manifest['reference_csv'])
    verified(reference, manifest['reference_csv_sha256'])
    with reference.open(newline='') as stream:
        references = list(csv.DictReader(stream))
    phones = errors = 0
    components = {key: 0 for key in COMPONENTS}
    traces = []
    for index, (row, audio, baseline) in enumerate(zip(rows, manifest['audio'], references, strict=True)):
        ref, hyp = row['ref'].split(), row['hyp'].split()
        if row['ref'] != baseline['ref'] or int(row['idx']) != index:
            raise ValueError('Reference differs from the original evaluation')
        distance = edit_distance(ref, hyp)
        if distance != int(row['phone_errors']) or len(ref) != int(row['reference_phones']):
            raise ValueError('Independent edit distance differs at ' + row['uid'])
        if int(row['input_mel_frames']) != audio['samples'] // 160 + 1:
            raise ValueError('Mel input length differs from the frozen audio')
        if not math.isclose(float(row['audio_seconds']), audio['samples'] / audio['sample_rate'], abs_tol=1e-12):
            raise ValueError('Audio duration differs from the frozen header')
        phones += len(ref)
        errors += distance
        for key in components:
            value = int(row[key])
            if value < 0:
                raise ValueError('Negative major MAC count')
            components[key] += value
        traces.append(dict(uid=row['uid'], layer_frames=json.loads(row['layer_frames']),
                           output_steps=int(row['output_steps'])))
    if phones != 189023 or errors != result['phone_errors'] or components != result['components_macs']:
        raise ValueError('Corpus totals disagree with the persisted result')
    if result['reference_phones'] != phones or not math.isclose(result['corpus_PER'], errors / phones, abs_tol=1e-15):
        raise ValueError('Corpus PER differs from independent recomputation')
    duration = math.fsum(a['samples'] / a['sample_rate'] for a in manifest['audio'])
    if not math.isclose(duration, 19452.480625, abs_tol=1e-8):
        raise ValueError('Unexpected complete-corpus duration')
    if not math.isclose(result['audio_seconds'], duration, abs_tol=1e-8):
        raise ValueError('Result reports another audio duration')
    if result['total_major_macs'] != sum(components.values()):
        raise ValueError('Total MACs disagree with their components')
    if not math.isclose(result['gmac_per_audio_second'], sum(components.values()) / duration / 1e9, abs_tol=1e-12):
        raise ValueError('Full-corpus MAC rate differs from the independent sum')
    model = yaml.safe_load((folder / 'config.yaml').read_text())['model']
    raw = [a['samples'] // 160 + 1 for a in manifest['audio']]
    rates = calibrate(model, raw, traces)
    trace = projected_trace(model, mel_frames(30), rates)
    counts = encoder_macs(model, trace)
    theory = dict(duration_seconds=30, method='pooled observed stage retention and fixed-length structure',
                  calibration_utterances=2620, stage_rates=rates, trace=trace,
                  components_macs=counts, encoder_compression_gmac=sum(counts.values()) / 1e9,
                  encoder_compression_gmac_per_audio_s=sum(counts.values()) / 30e9,
                  actual_beam20_macs=None, checkpoint_sha256=manifest['model']['checkpoint_sha256'],
                  manifest_sha256=sha(folder / 'manifest.json'),
                  theoretical_counter_sha256=sha(ROOT / 'scripts/evaluation/theoretical_phone69_macs.py'))
    (folder / 'fixed_duration_30s.json').write_text(json.dumps(theory, indent=2) + '\n')
    return dict(name=name, corpus_PER=errors / phones, per_percent=100 * errors / phones,
                utterances=len(rows), reference_phones=phones, phone_errors=errors,
                encoder_compression_gmac_30s=theory['encoder_compression_gmac'],
                encoder_compression_gmac_per_audio_s_30s=theory['encoder_compression_gmac_per_audio_s'],
                full_corpus_components_macs=components,
                full_corpus_total_gmac_per_audio_s=result['gmac_per_audio_second'],
                checkpoint_sha256=manifest['model']['checkpoint_sha256'],
                csv_sha256=sha(attempt / 'utterances.csv'),
                independent_edit_distance_checked=len(rows), checkpoint_hash_verified=True,
                frozen_model_source_verified=True, manifest_hash_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / 'EVALUATION_COMPLETE').is_file():
        raise ValueError('Wait for all three complete evaluations')
    models = [verify_model(root, name) for name in ('layer1', 'layer3', 'layer9')]
    verification = dict(verified_at_utc=now(), verifier_sha256=sha(__file__), complete=True, models=models)
    (root / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    lines = ['# L1／L3／L9家中CPU全量评测', '',
             '三组各覆盖test-clean全部2620句、189023个参考音素、19452.480625秒音频。',
             '协议：CPU／FP32／batch1／beam20，4线程、interop1、worker0、预热10句。', '',
             '| 插入位置 | Corpus PER | 编码器＋压缩GMAC／30秒 | GMAC／音频秒（30秒口径） | 音素错误数 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for item in models:
        lines.append('| {name} | {per_percent:.6f}% | {encoder_compression_gmac_30s:.6f} | {encoder_compression_gmac_per_audio_s_30s:.6f} | {phone_errors} |'.format(**item))
    lines += ['', 'PER来自完整原始音频，固定30秒理论MAC由本次完整逐层帧数校准；未将音频裁切到30秒，也未将变长全量MAC线性缩放。',
              '独立重算每句编辑距离、语料PER、参考覆盖、MAC分项之和，并核验checkpoint、配置、冻结源码、manifest和CSV哈希。',
              '实际beam20的predictor／joiner计数保留在各组result和verification中；不加入固定长度理论编码器主指标。',
              '主要MAC包括投影、线性、卷积、注意力矩阵乘积、检测点积和融合；不包括特征提取、归一化、softmax、激活、比较排序和访存。',
              '不同训练的单seed点估计，未新增配对区间，不据此声称统计显著。checkpoint文件名的epoch字段采用0起算。', '',
              '[独立核验](verification.json) · [L1记录](layer1/attempt_01/result.json) · [L3记录](layer3/attempt_01/result.json) · [L9记录](layer9/attempt_01/result.json)', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
