import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/test21/conformer-rnnt-asr')
OUT = ROOT / 'artifacts/evaluations/phone69_460h_five_frames_20261021_post_restart'
AUDIT = ROOT / 'artifacts/runs/phone69_460h_native_offline_five_frames_20261018'

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, value):
    value['updated_at_utc'] = now()
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def process_state(pid):
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        return False, 'exited'
    try:
        stat = (proc / 'stat').read_text().split()
        cmd = (proc / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
        if len(stat) > 2 and stat[2] == 'Z':
            return False, 'zombie'
        return ('eval_monotonic.py' in cmd, cmd[:400])
    except Exception as e:
        return False, 'unreadable: ' + repr(e)

def check_baseline():
    before = json.loads((AUDIT / 'baseline_preserved_before.json').read_text())
    after = json.loads((AUDIT / 'baseline_preserved_after.json').read_text())
    assert after.get('valid') is True and after.get('all_sha256_size_mtime_unchanged') is True
    rows = before.get('files') or []
    assert rows
    for row in rows:
        p = ROOT / row['path']
        st = p.stat()
        assert st.st_size == row['bytes'] and st.st_mtime_ns == row['mtime_ns']
        assert digest(p) == row['sha256'], 'baseline file changed: ' + row['path']
    return {'files': len(rows), 'bytes': sum(row['bytes'] for row in rows), 'unchanged': True}

def read_csv(job):
    path = Path(job['per_sample_csv'])
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2620, f'{job["name"]}: expected 2620 rows, found {len(rows)}'
    assert [int(r['idx']) for r in rows] == list(range(2620))
    errors = 0
    refs = 0
    rtf_sum = 0.0
    for r in rows:
        e = int(r['phone_errors'])
        n = int(r['reference_phones'])
        rate = float(r['PER'])
        rtf = float(r['RTF'])
        assert e >= 0 and n > 0 and math.isfinite(rate) and rate >= 0
        assert math.isfinite(rtf) and rtf >= 0
        errors += e
        refs += n
        rtf_sum += rtf
    return rows, {'name': job['name'], 'experiment': job['experiment'],
                  'checkpoint': job['checkpoint'], 'checkpoint_sha256': job['checkpoint_sha256'],
                  'num_samples': len(rows), 'reference_phones': refs, 'phone_errors': errors,
                  'PER': errors / refs, 'mean_utterance_RTF': rtf_sum / len(rows)}

state_path = OUT / 'launch_state.json'
state = json.loads(state_path.read_text())
if state.get('status') in {'complete', 'failed', 'partial_launch_failed', 'launch_failed'}:
    print(json.dumps({'server_utc': now(), 'status': state['status'], 'evaluation_dir': str(OUT), 'results': state.get('results'), 'summary': state.get('summary'), 'validation_report': state.get('validation_report'), 'jobs': state.get('jobs', [])}, ensure_ascii=False))
    raise SystemExit(0)

failures = []
active = []
for job in state.get('jobs', []):
    alive, detail = process_state(job['pid'])
    log = Path(job['log'])
    log_text = log.read_text(errors='replace') if log.exists() else ''
    success_text = 'Evaluation complete' in log_text and 'Traceback' not in log_text and 'CUDA out of memory' not in log_text
    if alive:
        job['status'] = 'running'
        job['last_progress'] = next((line[-220:] for line in reversed(log_text.splitlines()) if 'PER:' in line or 'PER ' in line or 'Evaluation complete' in line), '')
        active.append(job['name'])
    elif success_text and Path(job['per_sample_csv']).is_file():
        job['status'] = 'finished'
        job['last_progress'] = 'Evaluation complete'
    else:
        job['status'] = 'failed'
        job['failure'] = detail
        tail = log_text.splitlines()[-10:]
        job['log_tail'] = tail
        failures.append(job['name'])

if failures:
    state['status'] = 'failed'
    state['failures'] = failures
elif active:
    state['status'] = 'running'
    state['active_jobs'] = active
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    state['gpu'] = gpu.stdout.strip()
else:
    try:
        all_rows = []
        summary = []
        for job in state['jobs']:
            rows, item = read_csv(job)
            all_rows.append(rows)
            summary.append(item)
        ref_counts = {x['reference_phones'] for x in summary}
        assert len(ref_counts) == 1 and next(iter(ref_counts)) == 189023
        for rows in all_rows[1:]:
            assert all(a['ref'] == b['ref'] for a, b in zip(all_rows[0], rows)), 'reference alignment differs across models'
        baseline = check_baseline()
        value = {'test_subset': 'test-clean', 'metric': 'PER', 'beam_width': 20,
                 'results': summary, 'baseline_files_unchanged': baseline}
        (OUT / 'summary.json').write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        report = {'valid': True, 'sample_count_per_model': 2620,
                  'reference_phones_per_model': 189023, 'same_reference_order': True,
                  'all_per_and_rtf_finite_nonnegative': True,
                  'baseline_fingerprint_check': baseline, 'models': [x['name'] for x in summary]}
        (OUT / 'validation_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        (OUT / 'EVALUATION_COMPLETE').write_text('Four parallel test-clean PER evaluations validated.\n')
        state['status'] = 'complete'
        state['summary'] = str(OUT / 'summary.json')
        state['validation_report'] = str(OUT / 'validation_report.json')
        state['baseline_fingerprint_check'] = baseline
        state['results'] = summary
    except Exception as e:
        state['status'] = 'failed'
        state['failure'] = repr(e)
atomic_json(state_path, state)
print(json.dumps({'server_utc': now(), 'status': state.get('status'),
                  'active_jobs': state.get('active_jobs', []),
                  'finished_jobs': [j['name'] for j in state.get('jobs', []) if j.get('status') == 'finished'],
                  'failures': state.get('failures', []), 'failure': state.get('failure'),
                  'gpu': state.get('gpu'), 'results': state.get('results'),
                  'evaluation_dir': str(OUT)}, ensure_ascii=False))
