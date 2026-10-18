#!/usr/bin/env python3
"""Validate and report fixed-phone69 speech-rate decile results.

Report structure follows Emformer's eval_fast_slow.py and
exp/experiments/plot_decile_errors.py. Inference results come from the
validated full test-clean monotonic-clock evaluation, without new inference.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    'stride4_none': 'Stride 4 / No compression',
    'stride1_cosd_learned': 'Stride 1 / CosDiff learned',
    'stride4_cosd_learned': 'Stride 4 / CosDiff learned',
}
COLORS = ['#1976b5', '#d67817', '#7656a3']
KEYS = ['S', 'D', 'I']


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alignment_counts(reference, hypothesis):
    """Rolling-row minimum edits, ties: match > substitution > deletion > insertion."""
    ref, hyp = reference.split(), hypothesis.split()
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, phone in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, predicted in enumerate(hyp, 1):
            if phone == predicted:
                current.append(previous[j - 1])
                continue
            sub, delete, insert = previous[j - 1], previous[j], current[j - 1]
            costs = (sub[0], delete[0], insert[0])
            best = min(costs)
            if costs[0] == best:
                current.append((best + 1, sub[1] + 1, sub[2], sub[3]))
            elif costs[1] == best:
                current.append((best + 1, delete[1], delete[2] + 1, delete[3]))
            else:
                current.append((best + 1, insert[1], insert[2], insert[3] + 1))
        previous = current
    return previous[-1]


def validate(project, results, summary):
    source_dir = project / 'artifacts/evaluations/phone69_460h_testclean_20261017/rtf_monotonic'
    source_summary = json.loads((source_dir / 'summary.json').read_text())
    source_by_name = {e['experiment']: e for e in source_summary['experiments']}
    original_bins = project / 'speech_rate_analysis/deciles'
    bins = {b['decile']: b for b in summary['bins']}
    checks = []
    assert len(bins) == 10 and len(summary['experiments']) == 3
    source_validation = json.loads((source_dir / 'validation_report.json').read_text())
    assert source_validation['rtf_valid'] is True
    all_uids = set()
    for label, info in bins.items():
        ids = [line.split()[0] for line in (original_bins / f'{label}.txt').read_text().splitlines()]
        assert ids == info['utterance_ids'] and len(ids) == len(set(ids)) == 262
        assert not (set(ids) & all_uids)
        all_uids.update(ids)
    assert len(all_uids) == 2620
    source_fields = ['idx', 'PER', 'phone_errors', 'reference_phones', 'RTF', 'ref', 'hyp', *KEYS]
    for exp in summary['experiments']:
        name = exp['experiment']
        run_path = project / 'artifacts/runs/phone69_460h_serial/batch_resize_20261015' / name / 'run.json'
        run = json.loads(run_path.read_text())
        assert run['status'] == 'trained' and run['epoch'] == run['epochs'] == 50
        assert run['train_subset'] == ['train-clean-100', 'train-clean-360']
        assert run['best_checkpoint'] == exp['checkpoint'] == source_by_name[name]['checkpoint']
        assert digest(project / run['config']) == run['config_sha256']
        source_csv = source_dir / f'{name}_per_sample_sdi.csv'
        assert digest(source_csv) == exp['source_sha256']
        raw = {int(row['idx']): row for row in read_csv(source_csv)}
        assert set(raw) == set(range(2620))
        total_rows = []
        for d in exp['deciles']:
            rows = read_csv(results / name / 'details' / f"{d['decile']}_details.csv")
            assert len(rows) == 262
            assert [r['utterance_id'] for r in rows] == bins[d['decile']]['utterance_ids']
            refs = errors = 0
            counts = dict.fromkeys(KEYS, 0)
            for row in rows:
                source_row = raw[int(row['idx'])]
                assert all(row[key] == source_row[key] for key in source_fields)
                distance, sub, delete, insert = alignment_counts(row['ref'], row['hyp'])
                assert (sub, delete, insert) == tuple(int(row[k]) for k in KEYS)
                assert distance == int(row['phone_errors'])
                assert len(row['ref'].split()) == int(row['reference_phones'])
                assert math.isfinite(float(row['RTF'])) and float(row['RTF']) >= 0
                refs += int(row['reference_phones'])
                errors += distance
                for key in KEYS:
                    counts[key] += int(row[key])
            assert refs == d['reference_phones'] and errors == d['phone_errors']
            assert math.isclose(errors / refs, d['PER'], abs_tol=1e-12)
            for key in KEYS:
                assert counts[key] == d[key]
                assert math.isclose(counts[key] / refs, d[f'{key}_per_reference_phone'], abs_tol=1e-12)
            assert math.isclose(math.fsum(float(r['RTF']) for r in rows) / len(rows),
                                d['mean_utterance_RTF'], abs_tol=1e-12)
            total_rows.extend(rows)
        assert len(total_rows) == len({r['idx'] for r in total_rows}) == 2620
        assert {r['utterance_id'] for r in total_rows} == all_uids
        for key in ('reference_phones', 'phone_errors', *KEYS):
            assert sum(int(r[key]) for r in total_rows) == exp['overall'][key]
        checks.append(dict(experiment=name, checked_samples=len(total_rows),
                           independent_alignment_valid=True, original_rows_preserved=True,
                           best_checkpoint_matches_training=True, train_subset=run['train_subset'],
                           training_epochs=run['epoch'], config_sha256=run['config_sha256'],
                           source_csv_sha256=digest(source_csv)))
        print(f'Validated {name}: 2620 rows, 10 deciles, independent S/D/I alignment', flush=True)
    return dict(valid=True, same_emformer_decile_membership=True, checked_samples=7860,
                rtf_definition='arithmetic mean of monotonic per-utterance RTF', experiments=checks)


def save_figure(fig, out, stem):
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'{stem}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plots(experiments, out):
    out.mkdir()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    x = np.arange(1, 11)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout='constrained')
    for exp, color in zip(experiments, COLORS):
        ds = exp['deciles']
        label = LABELS[exp['short_name']]
        axes[0].plot(x, [100*d['PER'] for d in ds], 'o-', color=color, label=label)
        axes[1].plot(x, [d['mean_utterance_RTF'] for d in ds], 'o-', color=color, label=label)
    axes[0].set_ylabel('Corpus PER within bin (%)')
    axes[1].set_ylabel('Mean utterance RTF')
    axes[1].set_xlabel('Speech-rate decile (D1 slowest; D10 fastest; 262 utterances each)')
    axes[1].set_xticks(x, [f'D{i}' for i in x])
    axes[0].legend(loc='upper left', fontsize=9)
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    fig.suptitle('Phone69 / 460h training / test-clean / beam 20')
    save_figure(fig, out, 'per_rtf_by_decile')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, layout='constrained')
    for exp, ax, color in zip(experiments, axes, COLORS):
        bottom = np.zeros(10)
        for key, label, fill in [('S', 'Substitution', '#80bfa4'), ('D', 'Deletion', '#75a9d1'), ('I', 'Insertion', '#e6a09b')]:
            vals = np.array([d[f'{key}_per_reference_phone']*100 for d in exp['deciles']])
            ax.bar(x, vals, bottom=bottom, color=fill, label=label, width=0.7)
            bottom += vals
        pers = [d['PER']*100 for d in exp['deciles']]
        ax.plot(x, pers, 'o-', color=color, markersize=4, linewidth=1.5, label='PER')
        ax.set_title(LABELS[exp['short_name']], fontsize=11)
        ax.set_xticks(x, [f'D{i}' for i in x], fontsize=8)
        ax.set_xlabel('Slowest  →  Fastest')
        ax.grid(axis='y', alpha=0.25)
    axes[0].set_ylabel('Errors / reference phones (%)')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('S/D/I by speech-rate decile (shared vertical scale)')
    save_figure(fig, out, 'phoneme_error_by_decile')

    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    for exp, color, marker in zip(experiments, COLORS, ['o', 's', '^']):
        ds = exp['deciles']
        rtfs = [d['mean_utterance_RTF'] for d in ds]
        pers = [d['PER']*100 for d in ds]
        ax.plot(rtfs, pers, marker=marker, color=color, label=LABELS[exp['short_name']], alpha=0.8)
        for idx, (rtf, per) in enumerate(zip(rtfs, pers), 1):
            if idx in (1, 5, 10):
                offset = (4, -12) if idx == 5 else (4, 5)
                ax.annotate(f'D{idx}', (rtf, per), xytext=offset,
                            textcoords='offset points', fontsize=8, color=color)
        overall = exp['overall']
        ax.scatter(overall['mean_utterance_RTF'], overall['PER']*100, color=color,
                   marker=marker, s=160, edgecolor='black', zorder=5)
    ax.set_xlabel('Mean utterance RTF (lower is faster)')
    ax.set_ylabel('Corpus PER within bin (%)')
    ax.set_title('PER vs RTF; large markers show full test-clean results\nD1, D5 and D10 labeled; lines follow decile order')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    save_figure(fig, out, 'per_vs_rtf_scatter')


def report(summary, out):
    exps = summary['experiments']
    for exp in exps:
        exp['short_name'] = exp['experiment'].removeprefix('exp_phone69_460h_').removesuffix('_s42_e50')
    plots(exps, out / 'figures')
    lines = ['# 460h 三组模型：十档语速评测报告', '',
             '> 历史实验说明（2026-10-18）：以下旧三组属于已作废的 keep_rate/上下文实验，不代表当前 offline_native 四组自然峰谷重训结果；原始统计保留，旧 checkpoint 已按用户要求删除。', '',
             '训练：train-clean-100 + train-clean-360（共 460h），三组均完成 50 轮；测试：同一份 test-clean，共 2620 句、189023 个参考音素。每组使用训练记录中的最佳 checkpoint，固定 phone69、beam=20、A100 80 GiB。', '',
             '本报告按 Emformer 原始十档 ID 清单，重新执行分档、逐句编辑对齐核验和汇总。识别结果与计时来自已通过校验的完整 test-clean 单调时钟评测，未另行运行模型推理。解码器每句独立创建，分档不改变识别结果。', '',
             '## 统计口径与 Emformer 脚本的对应', '',
             '- 分档不变：PPM=源音素数/FLAC 音频时长×60，D1 最慢、D10 最快，每档 262 句；同速边界沿用原顺序。',
             '- 保留原脚本的总编辑数/总参考数汇总方法；本项目使用完整、保留大小写和重音的 phone69，指标命名为 PER。未沿用旧脚本的小写化、去空格和 SentencePiece 切分。',
             '- S/D/I 为最小编辑对齐中的替换/删除/插入；比率均除以该档参考音素数。并列路径优先 match、substitution、deletion、insertion；旧脚本优先 substitution、insertion、deletion，重复音素处的分类可能不同。',
             '- RTF 使用 time.perf_counter() 复测值，报告逐句 RTF 的算术平均，保留 evaluator 的特征帧时长分母与推理计时范围。原 Emformer 汇总 RTF 按音频时长加权，不能直接与本表比较。旧墙钟计时的无效 RTF 未使用。',
             '- 这是按语速分组的描述性结果；各档说话人、内容和时长也不同，不能把差异单独归因为语速。', '',
             '## 全测试集结果', '',
             '| 模型 | PER | 平均逐句 RTF | 替换 S（比率） | 删除 D（比率） | 插入 I（比率） |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for exp in exps:
        r = exp['overall']
        lines.append(f"| {exp['short_name']} | {r['PER']:.4%} | {r['mean_utterance_RTF']:.5f} | {r['S']} ({r['S_per_reference_phone']:.4%}) | {r['D']} ({r['D_per_reference_phone']:.4%}) | {r['I']} ({r['I_per_reference_phone']:.4%}) |")
    baseline, stride1, stride4 = exps
    assert [e['short_name'] for e in exps] == ['stride4_none', 'stride1_cosd_learned', 'stride4_cosd_learned']
    speedup = baseline['overall']['mean_utterance_RTF']/stride4['overall']['mean_utterance_RTF']
    fast = stride4['deciles'][-1]
    slow = stride4['deciles'][0]
    lines.extend(['', '## 主要观察', '',
                  f"- stride4_none 的全量 PER 最低（{baseline['overall']['PER']:.4%}）。stride1_cosd_learned 为 {stride1['overall']['PER']:.4%}，本次结果没有带来整体准确率或平均逐句 RTF 优势。",
                  f"- stride4_cosd_learned 的平均逐句 RTF 为 {stride4['overall']['mean_utterance_RTF']:.5f}，按本次 RTF 比值约为 baseline 的 {speedup:.2f} 倍推理速度，但全量 PER 上升至 {stride4['overall']['PER']:.4%}。",
                  f"- stride4_cosd_learned 在最快档 D10 的 PER 为 {fast['PER']:.4%}，最慢档 D1 为 {slow['PER']:.4%}；其中删除率从 {slow['D_per_reference_phone']:.4%} 升至 {fast['D_per_reference_phone']:.4%}。最快档的删除错误值得进一步分析。", '',
                  '![各语速档 PER 与 RTF](figures/per_rtf_by_decile.png)', '',
                  '![各档替换、删除、插入分解](figures/phoneme_error_by_decile.png)', '',
                  '![PER 与 RTF 对照](figures/per_vs_rtf_scatter.png)', '',
                  '## 十档范围', '',
                  '| 档位 | PPM 范围（音素/分钟） | 均值 PPM | 句数 | 参考音素数 |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for b, d in zip(summary['bins'], baseline['deciles']):
        lines.append(f"| {b['decile']} | {b['min_ppm']:.2f}–{b['max_ppm']:.2f} | {b['mean_ppm']:.2f} | {b['num_samples']} | {d['reference_phones']} |")
    for exp in exps:
        text = [f"EVALUATION RESULTS BY SPEECH-RATE DECILE: {exp['short_name']}",
                'test-clean; fixed phone69; beam=20; 262 samples/bin; slow -> fast',
                'RTF = arithmetic mean of monotonic per-utterance RTF',
                'Decile       Samples     PER      RTF       S       D       I     S/ref     D/ref     I/ref  EditDist  RefLen']
        lines.extend(['', f"## {exp['short_name']} 分档明细", '',
                      '| 档位 | PER | 平均逐句 RTF | S（比率） | D（比率） | I（比率） | 错误总数/参考数 |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for r in exp['deciles']:
            lines.append(f"| {r['decile']} | {r['PER']:.4%} | {r['mean_utterance_RTF']:.5f} | {r['S']} ({r['S_per_reference_phone']:.4%}) | {r['D']} ({r['D_per_reference_phone']:.4%}) | {r['I']} ({r['I_per_reference_phone']:.4%}) | {r['phone_errors']}/{r['reference_phones']} |")
            text.append(f"{r['decile']:12} {r['num_samples']:7} {r['PER']:8.4%} {r['mean_utterance_RTF']:8.5f} {r['S']:7} {r['D']:7} {r['I']:7} {r['S_per_reference_phone']:9.4%} {r['D_per_reference_phone']:9.4%} {r['I_per_reference_phone']:9.4%} {r['phone_errors']:9} {r['reference_phones']:7}")
        (out / f"{exp['short_name']}_eval_deciles_results.txt").write_text('\n'.join(text) + '\n')
    lines.extend(['', '## 来源与复现', '',
                  '- 参考脚本：本地 emformer_rnnt_copy/eval_fast_slow.py、exp/experiments/plot_decile_errors.py；仅参考分档与报告组织方式，以上口径差异已明确。',
                  '- [三组十档 CSV](../all_models_deciles.csv)、[完整分档汇总 JSON](../summary.json)、各模型目录下的 details/decile_XX_details.csv。',
                  '- [本次独立核验记录](validation_report.json)：7860 条逐句参考、预测、S/D/I 最小编辑对齐、RTF 和源 CSV 一致；十档无重复无遗漏；训练记录与最佳 checkpoint 一致。',
                  '- 图表同时提供 PNG 与矢量 PDF，位于 figures/。', '',
                  '```sh', 'python scripts/diagnostics/report_speech_rate_deciles.py --results eval_deciles_460h_作废_旧方法 --out eval_deciles_460h_作废_旧方法/report_new', '```', '',
                  '各组 checkpoint：', ''])
    for exp in exps:
        lines.append(f"- {exp['short_name']}: `{exp['checkpoint']}`")
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('eval_deciles_460h_作废_旧方法'))
    parser.add_argument('--out', type=Path, default=Path('eval_deciles_460h_作废_旧方法/report'))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    summary = json.loads((args.results / 'summary.json').read_text())
    checks = validate(project, args.results, summary)
    args.out.mkdir(parents=True, exist_ok=False)
    report(summary, args.out)
    checks['script_sha256'] = digest(Path(__file__))
    (args.out / 'validation_report.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(f"Report ready: {args.out / 'REPORT.md'}")


if __name__ == '__main__':
    main()
