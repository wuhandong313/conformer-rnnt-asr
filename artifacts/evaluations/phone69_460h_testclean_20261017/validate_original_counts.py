from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import runpy
import subprocess
import yaml

repo = Path('/test21/conformer-rnnt-asr')
os.chdir(repo)
out = repo / 'artifacts/evaluations/phone69_460h_testclean_20261017'
plan = json.loads((out / 'run_plan.json').read_text())
summary = json.loads((out / 'summary.json').read_text())
assert (out / 'RAW_EVALUATIONS_COMPLETE').is_file()
assert (out / 'EVALUATION_COMPLETE_WITH_SDI').is_file()
assert json.loads((out / 'eval_state.json').read_text())['status'] == 'complete'
assert plan['test_subset'] == summary['test_subset'] == 'test-clean'
assert plan['beam_width'] == summary['beam_width'] == 20
assert len(plan['experiments']) == len(summary['experiments']) == 3
phone_error_counts = runpy.run_path(str(repo / 'conformer_rnnt/metrics.py'))['phone_error_counts']
reference_sequence = None
verified = []
for exp, item in zip(plan['experiments'], summary['experiments']):
    name = exp['name']
    assert item['experiment'] == name
    assert item['checkpoint'] == exp['checkpoint']
    cfg_path = repo / exp['config']
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg['tokenizer']['type'] == 'phone69'
    assert cfg['data']['test_subset'] == 'test-clean'
    assert cfg['decoding']['beam_width'] == 20
    training_run = json.loads((repo / name / 'run.json').read_text())
    cfg_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    assert cfg_hash == training_run['config_sha256']
    assert exp['checkpoint'] == training_run['best_checkpoint']
    with (out / (name + '_per_sample.csv')).open(newline='') as handle:
        raw = list(csv.DictReader(handle))
    with (out / (name + '_per_sample_sdi.csv')).open(newline='') as handle:
        edits = list(csv.DictReader(handle))
    assert len(raw) == len(edits) == item['num_samples'] == 2620
    refs = [(row['idx'], row['ref']) for row in raw]
    if reference_sequence is None:
        reference_sequence = refs
    else:
        assert refs == reference_sequence
    total_s = total_d = total_i = total_errors = total_reference = 0
    rtfs = []
    for idx, (a, b) in enumerate(zip(raw, edits)):
        assert int(a['idx']) == int(b['idx']) == idx
        assert all(b[key] == value for key, value in a.items())
        s, d, ins = int(b['S']), int(b['D']), int(b['I'])
        assert min(s, d, ins) >= 0
        errors, nref = phone_error_counts(a['ref'], a['hyp'])
        assert errors == int(a['phone_errors']) == s + d + ins
        assert nref == int(a['reference_phones']) == len(a['ref'].split())
        assert nref > 0
        assert math.isclose(float(a['PER']), errors / nref, abs_tol=1e-12)
        rtf = float(a['RTF'])
        assert math.isfinite(rtf)
        rtfs.append(rtf)
        total_s += s; total_d += d; total_i += ins
        total_errors += errors; total_reference += nref
    assert total_reference == item['reference_phones'] == 189023
    assert total_errors == item['phone_errors'] == total_s + total_d + total_i
    assert (total_s, total_d, total_i) == (item['substitutions'], item['deletions'], item['insertions'])
    assert math.isclose(item['PER'], total_errors / total_reference, abs_tol=1e-12)
    assert math.isclose(item['mean_utterance_RTF'], sum(rtfs) / len(rtfs), abs_tol=1e-12)
    for label, count in [('S', total_s), ('D', total_d), ('I', total_i)]:
        assert math.isclose(item[label + '_per_reference_phone'], count / total_reference, abs_tol=1e-12)
    snapshot = out / (name + '_config.yaml')
    if snapshot.exists():
        assert snapshot.read_bytes() == cfg_path.read_bytes()
    else:
        with snapshot.open('xb') as handle:
            handle.write(cfg_path.read_bytes())
    verified.append({'experiment': name, 'num_samples': len(raw), 'reference_phones': total_reference,
                     'phone_errors': total_errors, 'S': total_s, 'D': total_d, 'I': total_i,
                     'PER': item['PER'], 'mean_utterance_RTF': item['mean_utterance_RTF'],
                     'config_sha256': cfg_hash, 'checkpoint': exp['checkpoint']})
report = {'rtf_valid': False, 'rtf_note': 'System wall-clock timing produced negative durations; use the independent monotonic remeasurement for RTF.', 'validated_at_server_utc': datetime.now(timezone.utc).isoformat(),
          'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'eval_py_sha256': hashlib.sha256((repo / 'eval.py').read_bytes()).hexdigest(),
          'metrics_py_sha256': hashlib.sha256((repo / 'conformer_rnnt/metrics.py').read_bytes()).hexdigest(),
          'checks': ['2620 ordered unique samples per model', 'identical references across models',
                     'raw CSV fields preserved in S/D/I CSV', 'every edit sum equals independently recomputed phone distance',
                     'corpus PER, reference count, S/D/I totals and ratios match summary',
                     'mean utterance RTF matches raw samples', 'config hash matches training run', 'best checkpoint matches training run'],
          'experiments': verified}
report_path = out / 'validation_report.json'
if not report_path.exists():
    with report_path.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
files = [{'path': str(p.relative_to(repo)), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
assert all(item['bytes'] <= 10 * 1024**2 for item in files)
print(json.dumps({'validation': 'PER and S/D/I passed; original RTF invalid', 'report': report, 'files': files,
                  'git_branch': subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD'], text=True).strip(),
                  'git_status': subprocess.check_output(['git', '-c', 'core.fsmonitor=false', 'status', '--short', '--untracked-files=no'], text=True),
                  'staged_paths': subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).splitlines()}, ensure_ascii=False))
