#!/usr/bin/env python3
"""Join validated phone69 evaluations to the existing Emformer speech-rate bins.

No inference is performed. --metadata is the server's ordered test-clean ID,
transcript and FLAC-duration export. Existing bin membership is preserved.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    refs = sum(int(r['reference_phones']) for r in rows)
    counts = {key: sum(int(r[key]) for r in rows) for key in ('S', 'D', 'I')}
    errors = sum(counts.values())
    assert errors == sum(int(r['phone_errors']) for r in rows)
    return dict(num_samples=len(rows), reference_phones=refs, phone_errors=errors,
                PER=errors / refs, **counts,
                S_per_reference_phone=counts['S'] / refs,
                D_per_reference_phone=counts['D'] / refs,
                I_per_reference_phone=counts['I'] / refs,
                mean_utterance_RTF=math.fsum(float(r['RTF']) for r in rows) / len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--evaluations', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    # Read/validate all inputs before creating outputs.
    metadata = json.loads(args.metadata.read_text())
    ordered = metadata['rows']
    assert len(ordered) == 2620
    assert [r['idx'] for r in ordered] == list(range(2620))
    ids = [r['utterance_id'] for r in ordered]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    by_id = {r['utterance_id']: r for r in ordered}
    rates_path = args.reference / 'speech_rates_test_clean.csv'
    rates = read_csv(rates_path)
    assert len(rates) == 2620 and {r['utterance_id'] for r in rates} == set(ids)
    rate_by_id = {r['utterance_id']: r for r in rates}
    for rate in rates:
        row = by_id[rate['utterance_id']]
        assert int(rate['num_phonemes']) == len(row['source_phones'])
        assert float(rate['duration_sec']) == round(row['duration_sec'], 3)
        assert float(rate['ppm']) == round(len(row['source_phones']) / row['duration_sec'] * 60, 2)
    assert [float(r['ppm']) for r in rates] == sorted(float(r['ppm']) for r in rates)
    bins = []
    seen = []
    for index in range(1, 11):
        label = f'decile_{index:02d}'
        path = args.reference / 'deciles' / f'{label}.txt'
        members = [line.split()[0] for line in path.read_text().splitlines() if line.strip()]
        assert len(members) == 262
        assert members == [r['utterance_id'] for r in rates[(index - 1) * 262:index * 262]]
        seen.extend(members)
        ppms = [float(rate_by_id[uid]['ppm']) for uid in members]
        bins.append(dict(decile=label, percentile=f'{(index-1)*10}-{index*10}%',
                         num_samples=len(members), min_ppm=min(ppms), max_ppm=max(ppms),
                         mean_ppm=math.fsum(ppms)/len(ppms), utterance_ids=members,
                         source_sha256=sha256(path)))
    assert len(set(seen)) == len(seen) == 2620
    source_summary = json.loads((args.evaluations / 'summary.json').read_text())
    validation = json.loads((args.evaluations / 'validation_report.json').read_text())
    assert validation['rtf_valid'] is True
    assert json.loads((args.evaluations / 'eval_state.json').read_text())['status'] == 'complete'
    experiments = []
    all_summaries = []
    for source in source_summary['experiments']:
        name = source['experiment']
        path = args.evaluations / f'{name}_per_sample_sdi.csv'
        rows = read_csv(path)
        assert len(rows) == len(ordered)
        for idx, row in enumerate(rows):
            assert int(row['idx']) == idx
            assert row['ref'] == ordered[idx]['ref'], (name, idx)
            assert len(row['ref'].split()) == int(row['reference_phones'])
            assert sum(int(row[k]) for k in ('S', 'D', 'I')) == int(row['phone_errors'])
            assert all(int(row[k]) >= 0 for k in ('S', 'D', 'I'))
            assert math.isfinite(float(row['RTF'])) and float(row['RTF']) >= 0
        joined = {}
        for idx, row in enumerate(rows):
            uid = ids[idx]
            rate = rate_by_id[uid]
            joined[uid] = dict(utterance_id=uid, ppm=rate['ppm'],
                               duration_sec=ordered[idx]['duration_sec'], **row)
        global_metrics = summarize(rows)
        for key in ('num_samples', 'reference_phones', 'phone_errors', 'PER',
                    'mean_utterance_RTF', 'S_per_reference_phone',
                    'D_per_reference_phone', 'I_per_reference_phone'):
            assert math.isclose(global_metrics[key], source[key], rel_tol=1e-12, abs_tol=1e-12)
        for key, source_key in [('S', 'substitutions'), ('D', 'deletions'), ('I', 'insertions')]:
            assert global_metrics[key] == source[source_key]
        deciles = []
        details = []
        for bin_info in bins:
            selected = [joined[uid] for uid in bin_info['utterance_ids']]
            metrics = summarize(selected)
            result = dict(experiment=name, **{k: v for k, v in bin_info.items()
                                             if k not in ('utterance_ids', 'source_sha256', 'num_samples')},
                          **metrics)
            deciles.append(result)
            all_summaries.append(result)
            details.extend(dict(decile=bin_info['decile'], **r) for r in selected)
        for key in ('num_samples', 'reference_phones', 'phone_errors', 'S', 'D', 'I'):
            assert sum(d[key] for d in deciles) == global_metrics[key]
        experiments.append(dict(experiment=name, checkpoint=source['checkpoint'],
                                source_csv=str(path), source_sha256=sha256(path),
                                overall=global_metrics, deciles=deciles, details=details))
    assert len(experiments) == 3
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.reference / 'deciles', args.out / 'deciles')
    shutil.copy2(rates_path, args.out / rates_path.name)
    write_csv(args.out / 'sample_mapping.csv', [dict(idx=r['idx'], utterance_id=r['utterance_id'],
              duration_sec=r['duration_sec'], reference_phones=len(r['ref'].split())) for r in ordered])
    write_csv(args.out / 'all_models_deciles.csv', all_summaries)
    for experiment in experiments:
        folder = args.out / experiment['experiment']
        (folder / 'details').mkdir(parents=True)
        write_csv(folder / 'eval_deciles_summary.csv', experiment['deciles'])
        for bin_info in bins:
            write_csv(folder / 'details' / f"{bin_info['decile']}_details.csv",
                      [r for r in experiment['details'] if r['decile'] == bin_info['decile']])
        del experiment['details']
        (folder / 'eval_deciles_summary.json').write_text(json.dumps(experiment, indent=2) + '\n')
    summary = dict(dataset='test-clean', training_data='train-clean-100 + train-clean-360 (460h)',
                   beam_width=source_summary['beam_width'], partition='10 equal-count bins, slow to fast',
                   ppm_definition='source whitespace-delimited phones / FLAC duration seconds * 60',
                   rtf_definition='arithmetic mean of validated monotonic per-utterance RTF; denominator is evaluator feature-frame duration',
                   sdi_denominator='reference phones in each bin', reference_source=str(args.reference),
                   reference_csv_sha256=sha256(rates_path), metadata_server_utc=metadata['server_utc'],
                   bins=bins, experiments=experiments)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ['# Phone69 460h：十档语速评测', '',
              '> 历史实验说明（2026-10-18）：此统计对应已作废的旧三组 keep_rate/上下文实验，不代表当前 offline_native 四组自然峰谷重训结果。', '',
              '训练集：train-clean-100 + train-clean-360，共 460h。评测集：完整 test-clean，2620 句，beam=20，沿用已记录的最佳 checkpoint。', '',
              '完全复用本地 Emformer 的十档句子清单：按源音素数/真实音频时长计算 PPM（音素/分钟），从慢到快排序，等量分档，每档 262 句；不是等宽语速区间。相同语速的分档边界也保留原清单顺序。', '',
              '所有评测 idx 均与服务器按音频 ID 排序的 test-clean 元数据逐条核对；三组的 ref 全部一致，源音素数、音频时长和 PPM 均与 Emformer 记录一致。此数据集无 spn，源计数与 phone69 参考计数一致。', '',
              'PER=(S+D+I)/该档参考音素数；S/D/I 比率也以该档参考音素数为分母。RTF 为有效单调时钟复测的逐句 RTF 算术平均（沿用 evaluator 的特征帧时长分母），不按音频时长加权。未重新运行模型推理。', '']
    for experiment in experiments:
        report.extend([f"## {experiment['experiment']}", '',
                       '| 档位（慢→快） | PPM 范围 | PER | 平均逐句 RTF | 替换 S（比率） | 删除 D（比率） | 插入 I（比率） |',
                       '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for row in experiment['deciles']:
            report.append(f"| {row['decile']} | {row['min_ppm']:.2f}–{row['max_ppm']:.2f} | {row['PER']:.4%} | {row['mean_utterance_RTF']:.5f} | {row['S']} ({row['S_per_reference_phone']:.4%}) | {row['D']} ({row['D_per_reference_phone']:.4%}) | {row['I']} ({row['I_per_reference_phone']:.4%}) |")
        report.append('')
    report.extend(['## 文件', '', '`all_models_deciles.csv` 为三组共 30 行的对照总表；各模型目录含 CSV/JSON 汇总及十个 `details/decile_XX_details.csv`。`deciles/` 保存原始分档清单，`sample_mapping.csv` 保存评测行号和音频 ID 的映射。', '',
                   '已校验每档 262 句、无重复无遗漏；三组全部 7860 行的参考音素与服务器元数据一致；十档累加的参考数、错误数、S/D/I 和总体 PER、平均逐句 RTF 与已验证全量结果一致。', ''])
    (args.out / 'README.md').write_text('\n'.join(report))
    (args.out / 'validation_report.json').write_text(json.dumps(dict(
        valid=True, num_bins=10, samples_per_bin=262, models=3, matched_sample_references=7860,
        source_rate_counts_durations_ppm_match=True, bin_membership_identical_to_emformer=True,
        global_metrics_match=True, all_rtf_finite_nonnegative=True), indent=2) + '\n')
    print(f'Validated 3 models × 10 bins × 262 utterances: {args.out}')


if __name__ == '__main__':
    main()
