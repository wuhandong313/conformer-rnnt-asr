"""Derive S/D/I from the completed four-model CSVs; preserve original results."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def edit_breakdown(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Token Levenshtein alignment; traceback ties: match, S, D, I."""
    ref, hyp = reference.split(), hypothesis.split()
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i, phone in enumerate(ref, 1):
        for j, predicted in enumerate(hyp, 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (phone != predicted))
    s = d = ins = 0
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            s, i, j = s + 1, i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            d, i = d + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return s, d, ins


def main() -> None:
    out = Path(__file__).resolve().parent
    original = json.loads((out / 'summary.json').read_text())
    inputs = [out / 'summary.json'] + [out / (r['name'] + '_per_sample.csv') for r in original['results']]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    results, csv_outputs, common_refs = [], [], None
    for recorded in original['results']:
        name = recorded['name']
        with (out / (name + '_per_sample.csv')).open(newline='') as f:
            reader = csv.DictReader(f)
            fields, rows = list(reader.fieldnames or []), list(reader)
        assert len(rows) == recorded['num_samples'] == 2620, name
        assert [int(r['idx']) for r in rows] == list(range(2620)), name
        refs = [r['ref'].split() for r in rows]
        if common_refs is None:
            common_refs = refs
        assert refs == common_refs, name
        s = d = ins = n_ref = errors = 0
        rtfs = []
        for row, ref in zip(rows, refs):
            a, b, c = edit_breakdown(row['ref'], row['hyp'])
            e, n, per, rtf = int(row['phone_errors']), int(row['reference_phones']), float(row['PER']), float(row['RTF'])
            assert a + b + c == e, (name, row['idx'], a, b, c, e)
            assert len(ref) == n and n > 0, (name, row['idx'])
            assert math.isfinite(per) and math.isclose(per, e / n, rel_tol=1e-12, abs_tol=1e-12)
            assert math.isfinite(rtf) and rtf >= 0, (name, row['idx'], rtf)
            row.update(S=a, D=b, I=c)
            s, d, ins, n_ref, errors = s + a, d + b, ins + c, n_ref + n, errors + e
            rtfs.append(rtf)
        per, mean_rtf = errors / n_ref, statistics.mean(rtfs)
        assert errors == recorded['phone_errors'] and n_ref == recorded['reference_phones'] == 189023
        assert math.isclose(per, recorded['PER'], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(mean_rtf, recorded['mean_utterance_RTF'], rel_tol=1e-12, abs_tol=1e-12)
        results.append(dict(recorded, substitutions=s, deletions=d, insertions=ins,
                            S_per_reference_phone=s / n_ref, D_per_reference_phone=d / n_ref,
                            I_per_reference_phone=ins / n_ref,
                            S_share_of_errors=s / errors, D_share_of_errors=d / errors,
                            I_share_of_errors=ins / errors,
                            median_utterance_RTF=statistics.median(rtfs),
                            p95_utterance_RTF=sorted(rtfs)[math.ceil(.95 * len(rtfs)) - 1],
                            perfect_utterances=sum(int(r['phone_errors']) == 0 for r in rows)))
        csv_outputs.append((out / (name + '_per_sample_sdi.csv'), fields + ['S', 'D', 'I'], rows))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name] for p in inputs)
    derived = dict(test_subset=original['test_subset'], beam_width=original['beam_width'],
                   metric='corpus PER', alignment_tie_break='match, substitution, deletion, insertion',
                   phone_tokenization='whitespace-separated complete phones; case and stress preserved',
                   ratio_denominator='reference phone count',
                   rtf_definition='arithmetic mean of per-utterance monotonic-clock RTF',
                   rtf_caveat='up to four concurrent jobs shared one A100; not isolated GPU benchmarks',
                   results=results)
    for path, fields, rows in csv_outputs:
        with path.open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    for filename, data in [('sdi_summary.json', derived), ('sdi_validation_report.json', {
        'valid': True, 'computed_at_utc': datetime.now(timezone.utc).isoformat(),
        'models': len(results), 'samples_per_model': 2620, 'reference_phones_per_model': 189023,
        'every_sample_S_D_I_matches_phone_errors': True, 'same_reference_order': True,
        'corpus_PER_and_mean_RTF_match_original_summary': True, 'RTF_finite_nonnegative': True,
        'original_files_preserved': True, 'input_sha256': hashes})]:
        with (out / filename).open('x') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(derived, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
