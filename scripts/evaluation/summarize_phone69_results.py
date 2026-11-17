#!/usr/bin/env python3
"""Collect verified Phone69 PER and fixed-30-second encoder/compression MACs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

from theoretical_phone69_macs import encoder_macs, mel_frames, projected_trace, sha

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'artifacts/research/phone69_fixed_duration_20261112/results.json'
ARCHIVE = ROOT / 'papers/phone69_460h_cpu_authoritative_20261108/supporting/original_archive'
SWEEP = ROOT / 'artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111'
PLACEMENTS = ROOT / 'artifacts/evaluations/phone69_placement_l1_l3_l9_local_cpu_20261117'
TRAIN_BEST = ROOT / 'artifacts/evaluations/phone69_layer3_train_best_local_cpu_20261117'
MEL_L3 = ROOT / 'artifacts/evaluations/phone69_mel_layer3_local_cpu_20261117'
OUT = ROOT / 'artifacts/research/phone69_all_results_20261117'
LABELS = dict(E01='Stride 4（主干128）', E02='L2', E03='Stride 2＋L2',
              E04='Stride 4＋L2', E05='Mel', E06='L5', E07='L7',
              E08='L2＋L5*', E09='Stride 4（主干512）', E10='Mel＋L5',
              E11='Mel＋L5＋L8', S1='Stride 1', S2='Stride 2', S8='Stride 8',
              P_L1='L1', P_L3='L3', P_L9='L9', P_MEL_L3='Mel＋L3',
              D_L3_TRAIN='L3（训练回调best，诊断）')
ORDER = ['S1', 'S2', 'E09', 'E05', 'P_L1', 'E02', 'P_L3', 'E06', 'E07',
         'P_L9', 'P_MEL_L3', 'E10', 'E08', 'D_L3_TRAIN',
         'E01', 'E03', 'E04', 'S8', 'E11']


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).relative_to(ROOT))


def verify_hash(path, digest):
    if sha(path) != digest:
        raise ValueError('Changed source: ' + str(path))


def record(identity, model, result_path, csv_path, theory, sources, inclusion, note=''):
    result = read(result_path)
    if result['utterances'] != 2620 or result['reference_phones'] != 189023:
        raise ValueError('Only verified full test-clean results belong in the table')
    verify_hash(csv_path, result['csv_sha256'])
    errors = result['phone_errors']
    if not math.isclose(result['corpus_PER'], errors / 189023, abs_tol=1e-15):
        raise ValueError('Corpus PER does not match the error numerator')
    duration = result['audio_seconds']
    if not math.isclose(duration, 19452.480625, abs_tol=1e-8):
        raise ValueError('Different test corpus duration')
    trace = projected_trace(model, mel_frames(30), theory['stage_rates'])
    parts = encoder_macs(model, trace)
    if theory['trace'] != trace or theory['components_macs'] != parts:
        raise ValueError('Archived 30-second count does not match the common formula')
    if not math.isclose(sum(parts.values()) / 1e9, theory['encoder_compression_gmac'], abs_tol=1e-12):
        raise ValueError('Archived theoretical MAC sum differs')
    actual = result.get('components_macs')
    if actual is None and identity.startswith('E') and identity != 'E11':
        count = read(ROOT / 'artifacts/research/phone69_460h_compute_20261108/full' / (identity + '.json'))
        if (count['checkpoint_sha256'] != result['checkpoint_sha256']
                or count['utterances'] != 2620 or count['provenance']['sample_per_quartile'] is not None):
            raise ValueError('Actual decoder count is not the matching full corpus')
        actual = count['components_macs']
    actual_rates = None
    if actual is not None:
        encoder = sum(v for k, v in actual.items() if k not in ('predictor', 'joiner'))
        actual_rates = dict(encoder_compression=encoder / duration / 1e9,
                            predictor=actual['predictor'] / duration / 1e9,
                            joiner=actual['joiner'] / duration / 1e9,
                            total=sum(actual.values()) / duration / 1e9)
    entry = result.get('model', result)
    checkpoint = entry['checkpoint']
    source_paths = list(dict.fromkeys([result_path, csv_path, *sources]))
    return dict(id=identity, system=LABELS[identity], inclusion=inclusion, note=note,
                checkpoint_selection='saved training callback batch loss' if identity == 'D_L3_TRAIN' else 'dev-clean validation loss',
                backbone=model['time_reduction_input_dim'] * model['time_reduction_stride'],
                per_percent=100 * errors / 189023, utterances=2620,
                reference_phones=189023, phone_errors=errors,
                encoder_compression_gmac_30s=sum(parts.values()) / 1e9,
                encoder_compression_gmac_per_audio_s_30s=sum(parts.values()) / 30e9,
                stage_rates=theory['stage_rates'], trace_30s=trace,
                components_macs_30s=parts, actual_beam20_gmac_per_audio_s=actual_rates,
                checkpoint=Path(checkpoint).name, checkpoint_sha256=entry['checkpoint_sha256'],
                sources={relative(p): sha(p) for p in source_paths})


def collect(allow_pending=False):
    old = read(OLD)
    verify_hash(ROOT / 'scripts/evaluation/theoretical_phone69_macs.py', old['script_sha256'])
    selection = read(ROOT / 'papers/extremafuse_acl2027/fixed_duration_selection.json')
    selected = {m['archive_id'] for m in selection['selected']}
    excluded = {m['archive_id']: m['reason'] for m in selection['excluded']}
    records = []
    for item in old['models']:
        if not item['results']:
            continue
        identity = item['id']
        verify_hash(ROOT / item['config'], item['config_sha256'])
        verify_hash(ROOT / item['calibration_source'], item['calibration_sha256'])
        if identity.startswith('S'):
            folder = SWEEP / ('stride' + identity[1:] + '_proj128')
            result_path, csv_path = folder / 'attempt_01/result.json', folder / 'attempt_01/utterances.csv'
        elif identity == 'E11':
            folder = ROOT / 'artifacts/evaluations/phone69_460h_mel_layer5_layer8_local_cpu_20261108'
            result_path, csv_path = folder / 'result.json', folder / 'utterances.csv'
        else:
            result_path, csv_path = ARCHIVE / identity / 'result.json', ARCHIVE / identity / 'attempt_01/utterances.csv'
        row = next(r for r in item['results'] if r['duration_seconds'] == 30)
        theory = dict(row, stage_rates=item['stage_rates'])
        entry = read(result_path).get('model', read(result_path))
        if entry['checkpoint_sha256'] != item['checkpoint_sha256']:
            raise ValueError('PER and retention belong to different checkpoints')
        records.append(record(identity, item['model'], result_path, csv_path, theory,
                              [OLD, ROOT / item['config'], ROOT / item['calibration_source']],
                              'existing_paper_selection' if identity in selected else 'historically_excluded',
                              excluded.get(identity, '')))
    for root, name, identity, attempt, inclusion in [
            (PLACEMENTS, 'layer1', 'P_L1', 'attempt_01', 'new_placement_comparison'),
            (PLACEMENTS, 'layer3', 'P_L3', 'attempt_01', 'new_placement_comparison'),
            (PLACEMENTS, 'layer9', 'P_L9', 'attempt_01', 'new_placement_comparison'),
            (TRAIN_BEST, 'layer3_train_best', 'D_L3_TRAIN', 'attempt_02', 'checkpoint_diagnostic'),
            (MEL_L3, 'mel_layer3', 'P_MEL_L3', 'attempt_01', 'new_placement_comparison')]:
        folder = root / name
        if not (root / 'verification.json').exists():
            if allow_pending and root == MEL_L3:
                continue
            raise ValueError('Wait for completed independent verification: ' + str(root))
        verification = read(root / 'verification.json')
        if not verification['complete']:
            raise ValueError('Incomplete independent verification')
        verified_models = verification.get('models') or [verification['comparison']['training_best']]
        item = next(m for m in verified_models if m['name'] == name)
        result_path = folder / attempt / 'result.json'
        result = read(result_path)
        manifest = read(folder / 'manifest.json')
        verify_hash(folder / 'manifest.json', result['manifest_sha256'])
        verify_hash(folder / 'config.yaml', manifest['model']['config_sha256'])
        if (item['checkpoint_sha256'], item['csv_sha256']) != (
                result['model']['checkpoint_sha256'], result['csv_sha256']):
            raise ValueError('Independent verification belongs to another evaluation')
        model = yaml.safe_load((folder / 'config.yaml').read_text())['model']
        theory = read(folder / 'fixed_duration_30s.json')
        if theory['checkpoint_sha256'] != result['model']['checkpoint_sha256']:
            raise ValueError('Theoretical MACs belong to another checkpoint')
        records.append(record(identity, model, result_path, folder / attempt / 'utterances.csv', theory,
                              [root / 'verification.json', folder / 'manifest.json', folder / 'config.yaml',
                               folder / 'fixed_duration_30s.json'], inclusion))
    records.sort(key=lambda row: ORDER.index(row['id']))
    if len({row['checkpoint_sha256'] for row in records}) != len(records):
        raise ValueError('Duplicate checkpoint in the complete results table')
    return records


def write_report(out, records):
    def table(rows):
        lines = ['| 实验 | 主干维度 | PER (%) | GMAC／30秒 | GMAC／音频秒 |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for row in rows:
            lines.append('| {system} | {backbone} | {per_percent:.4f} | {encoder_compression_gmac_30s:.6f} | '
                         '{encoder_compression_gmac_per_audio_s_30s:.6f} |'.format(**row))
        return lines

    main = [r for r in records if r['inclusion'] in ('existing_paper_selection', 'new_placement_comparison')]
    diagnostics = [r for r in records if r['inclusion'] == 'checkpoint_diagnostic']
    excluded = [r for r in records if r['inclusion'] == 'historically_excluded']
    lines = ['# Phone69全部实验结果（固定30秒MAC）', '',
             '2026-11-17。PER使用完整test-clean 2620句、189023个参考音素、19452.480625秒音频；CPU／FP32／batch1／beam20、4线程、interop1、worker0。',
             '30秒列是3001个80维Mel帧、batch1的编码器与压缩主要MAC，由各checkpoint完整2620句的逐层条件保留率校准。一次乘加计1 MAC；不将变长语料计数线性缩放。', '',
             '## 原论文入选实验与新增位置对照', '']
    lines += table(main)
    lines += ['', 'L2＋L5*首阶段为hard三帧邻域、第二阶段为learned五帧融合；其余动态压缩位置采用learned五帧。',
              '此表保留原九组论文入选项及新完成的位置对照，不据本次test-clean结果自动重选checkpoint或推断显著性。不同stride的主干宽度、训练预算与更新次数并不完全一致。', '',
              '## 独立选模诊断', ''] + table(diagnostics)
    lines += ['', 'L3训练回调best与验证loss best单列。它是同一训练的额外checkpoint诊断；不可把在test-clean上比较后选出的较好值视为独立的正式验证选模结果。', '',
              '## 历史未入选结果（完整保留）', ''] + table(excluded)
    lines += ['', '128维stride4由用户明确从论文对比移除；stride2/4再加L2、stride8和三阶段结果沿用原未入选记录。完整CSV和JSON保留所有已核验行。', '',
              '## 尚未完成的实验', '',
              '| 实验 | 状态 | PER／30秒MAC |', '| --- | --- | --- |',
              '| Mel＋L7 | 本次检查时正在训练 | 不填未核验值 |',
              '| L2＋L7* | 排队等待 | 不填未核验值 |', '',
              '## 实际beam20解码计数（另一口径）', '',
              '下表单位是完整变长语料的GMAC／实际音频秒，分别报告编码器与压缩、predictor、joiner。它不是30秒理论值。', '',
              '| 实验 | 编码器与压缩 | Predictor | Joiner | 合计 |', '| --- | ---: | ---: | ---: | ---: |']
    for row in records:
        count = row['actual_beam20_gmac_per_audio_s']
        if count is None:
            lines.append('| ' + row['system'] + ' | 待完整计数 | — | — | — |')
        else:
            lines.append('| ' + row['system'] + ' | {encoder_compression:.6f} | {predictor:.6f} | '
                         '{joiner:.6f} | {total:.6f} |'.format(**count))
    lines += ['', '## 范围与复核', '',
              '主要MAC包括输入输出投影、线性、卷积、注意力矩阵乘积、检测点积、融合MLP与加权和；不含特征提取、归一化、激活、softmax、比较排序、访存及训练。',
              '所有行检查完整语料覆盖、错误数与PER、CSV及配置/来源哈希；统一公式重新核对30秒逐层长度及六个MAC分项。新评测另有独立逐句编辑距离复算。历史评测与计时字段原样保留。',
              '保留率按各阶段输入加权统计；理论MAC依赖该checkpoint的保留率，并不表示每条30秒语音都具有同一实际计算量。', '',
              '[完整CSV](summary.csv) · [逐行来源及SHA-256](results.json) · [统一算法与范围](../../../docs/phone69_fixed_duration_macs.md) · '
              '[Mel＋L3全量评测](../../evaluations/phone69_mel_layer3_local_cpu_20261117/REPORT.md)', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--allow-pending', action='store_true', help='Explicit draft without unfinished Mel+L3')
    args = parser.parse_args()
    records = collect(args.allow_pending)
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(schema=1, snapshot_date='2026-11-17', complete=not args.allow_pending,
                  duration_seconds=30, input_mel_frames=3001, counted_scope='encoder and compression major MACs',
                  per_protocol='full test-clean, CPU/FP32/batch1/beam20, threads4/interop1/workers0',
                  script_sha256=sha(__file__), models=records,
                  pending=['mel_layer7', 'layers2_7_n3'])
    (args.out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    fields = ['id', 'system', 'backbone', 'per_percent', 'encoder_compression_gmac_30s',
              'encoder_compression_gmac_per_audio_s_30s', 'phone_errors', 'reference_phones',
              'utterances', 'inclusion', 'checkpoint_selection', 'checkpoint', 'checkpoint_sha256']
    with (args.out / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    write_report(args.out, records)
    print(json.dumps(dict(checkpoints=len(records), complete=report['complete'], out=str(args.out))))


if __name__ == '__main__':
    main()
