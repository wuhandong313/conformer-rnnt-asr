from __future__ import annotations

import csv
import json
import math
from pathlib import Path

root = Path('/test21/conformer-rnnt-asr')
out_dir = root / 'artifacts/evaluations/phone69_460h_testclean_20261017'
plan_path = out_dir / 'run_plan.json'
plan = json.loads(plan_path.read_text())

def edit_breakdown(reference: str, hypothesis: str) -> tuple[int, int, int]:
    ref = reference.split()
    hyp = hypothesis.split()
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i, phone in enumerate(ref, 1):
        for j, predicted in enumerate(hyp, 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (phone != predicted),
            )
    substitutions = deletions = insertions = 0
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, deletions, insertions

summaries = []
for exp in plan['experiments']:
    name = exp['name']
    raw_csv = out_dir / (name + '_per_sample.csv')
    if not raw_csv.is_file():
        raise SystemExit('Missing per-sample CSV: ' + str(raw_csv))
    with raw_csv.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    required = {'PER', 'phone_errors', 'reference_phones', 'RTF', 'ref', 'hyp'}
    if not required.issubset(fields):
        raise SystemExit(f'{name}: unexpected CSV columns {fields}')
    extended_fields = fields + ['S', 'D', 'I']
    per_sample_path = out_dir / (name + '_per_sample_sdi.csv')
    temp_path = per_sample_path.with_suffix(per_sample_path.suffix + '.tmp')
    total = {'substitutions': 0, 'deletions': 0, 'insertions': 0, 'phone_errors': 0, 'reference_phones': 0}
    rtfs = []
    with temp_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=extended_fields)
        writer.writeheader()
        for row in rows:
            s, d, ins = edit_breakdown(row['ref'], row['hyp'])
            errors = int(row['phone_errors'])
            if s + d + ins != errors:
                raise SystemExit(f'{name} sample {row["idx"]}: alignment={s+d+ins}, evaluator={errors}')
            row.update({'S': s, 'D': d, 'I': ins})
            writer.writerow(row)
            total['substitutions'] += s
            total['deletions'] += d
            total['insertions'] += ins
            total['phone_errors'] += errors
            total['reference_phones'] += int(row['reference_phones'])
            rtfs.append(float(row['RTF']))
    temp_path.replace(per_sample_path)
    n = len(rows)
    ref_phones = total['reference_phones']
    denom = ref_phones or 1
    s, d, ins = total['substitutions'], total['deletions'], total['insertions']
    rtf_sorted = sorted(rtfs)
    p95_index = max(0, math.ceil(0.95 * n) - 1)
    summary = {
        'experiment': name,
        'config': exp['config'],
        'checkpoint': exp['checkpoint'],
        'test_subset': plan['test_subset'],
        'metric': 'PER',
        'beam_width': plan['beam_width'],
        'num_samples': n,
        'reference_phones': ref_phones,
        'phone_errors': total['phone_errors'],
        'PER': total['phone_errors'] / denom,
        'substitutions': s,
        'deletions': d,
        'insertions': ins,
        'S_per_reference_phone': s / denom,
        'D_per_reference_phone': d / denom,
        'I_per_reference_phone': ins / denom,
        'S_share_of_errors': s / total['phone_errors'] if total['phone_errors'] else 0.0,
        'D_share_of_errors': d / total['phone_errors'] if total['phone_errors'] else 0.0,
        'I_share_of_errors': ins / total['phone_errors'] if total['phone_errors'] else 0.0,
        'mean_utterance_RTF': sum(rtfs) / n if n else 0.0,
        'median_utterance_RTF': rtf_sorted[n // 2] if n else 0.0,
        'p95_utterance_RTF': rtf_sorted[p95_index] if n else 0.0,
        'per_sample_csv': str(per_sample_path),
        'raw_eval_log': str(out_dir / (name + '.log')),
        'alignment_tie_break': 'match, then substitution, then deletion, then insertion',
    }
    summaries.append(summary)
    print(json.dumps(summary, ensure_ascii=False))

result = {
    'started_at_utc': plan['started_at_utc'],
    'test_subset': plan['test_subset'],
    'metric': 'PER',
    'beam_width': plan['beam_width'],
    'experiments': summaries,
}
summary_path = out_dir / 'summary.json'
tmp_summary = summary_path.with_suffix('.json.tmp')
tmp_summary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
tmp_summary.replace(summary_path)
(out_dir / 'EVALUATION_COMPLETE_WITH_SDI').write_text('Full test-clean evaluation and S/D/I aggregation complete.\n')
print('WROTE', summary_path)
