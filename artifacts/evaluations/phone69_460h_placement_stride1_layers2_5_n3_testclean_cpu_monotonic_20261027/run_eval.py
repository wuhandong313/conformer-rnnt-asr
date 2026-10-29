from datetime import datetime, timezone
from pathlib import Path
import csv, json, math, os, subprocess

root = Path('/test21/conformer-rnnt-asr')
out = Path(__file__).resolve().parent
plan = json.loads((out / 'run_plan.json').read_text())
now = lambda: datetime.now(timezone.utc).isoformat()

def save(name, value):
    path = out / name
    tmp = out / (name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)

state = {'status': 'running', 'started_at_utc': now(), 'scheme': plan['scheme'], 'device': 'cpu', 'checkpoint': plan['checkpoint']}
save('eval_state.json', state)
try:
    env = os.environ.copy()
    env.update(plan['cpu_limits'])
    env.update({'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(root) + os.pathsep + os.environ.get('PYTHONPATH', '')})
    command = ['/root/anaconda3/envs/whd/bin/python', '-u', plan['eval_script'], plan['eval_config'], '--checkpoint', plan['checkpoint'], '--metric', 'per', '--out', plan['output_csv'], '--cpu']
    with Path(plan['output_log']).open('x') as log:
        child = subprocess.Popen(command, cwd=str(root), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        state['pid'] = child.pid
        save('eval_state.json', state)
        code = child.wait()
    if code:
        raise RuntimeError('Evaluation process exited ' + str(code))
    with Path(plan['output_csv']).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != plan['expected_samples']:
        raise RuntimeError('Unexpected sample count: ' + str(len(rows)))
    refs = [(int(x['idx']), x['ref']) for x in rows]
    if len(set(idx for idx, ref in refs)) != len(refs):
        raise RuntimeError('Duplicate sample index')
    for source in plan['reference_csvs']:
        with Path(source).open(newline='') as f:
            old_refs = [(int(x['idx']), x['ref']) for x in csv.DictReader(f)]
        if refs != old_refs:
            raise RuntimeError('References differ from earlier evaluation: ' + source)
    errors = sum(int(x['phone_errors']) for x in rows)
    phones = sum(int(x['reference_phones']) for x in rows)
    if errors < 0 or phones != plan['expected_reference_phones']:
        raise RuntimeError('Invalid corpus phone counts: ' + str((errors, phones)))
    rtfs = [float(x['RTF']) for x in rows]
    if not all(math.isfinite(x) and x >= 0 for x in rtfs):
        raise RuntimeError('Non-finite or negative RTF')
    result = {'status': 'complete', 'scheme': plan['scheme'], 'device': 'cpu', 'timer': 'time.perf_counter', 'test_subset': 'test-clean', 'metric': 'per', 'beam_width': 20, 'checkpoint': plan['checkpoint'], 'checkpoint_sha256': plan['checkpoint_sha256'], 'num_samples': len(rows), 'phone_errors': errors, 'reference_phones': phones, 'PER': errors / phones, 'mean_utterance_RTF': sum(rtfs) / len(rtfs), 'raw_csv': plan['output_csv'], 'log': plan['output_log'], 'eval_config': plan['eval_config'], 'eval_config_sha256': plan['eval_config_sha256'], 'completed_at_utc': now()}
    validation = {'valid': True, 'sample_count': len(rows), 'expected_samples': plan['expected_samples'], 'reference_phones': phones, 'expected_reference_phones': plan['expected_reference_phones'], 'reference_order_matches_mel_and_layer5': True, 'rtf_finite_nonnegative': True, 'completed_at_utc': now()}
    save('summary.json', result)
    save('validation_report.json', validation)
    (out / 'EVALUATION_COMPLETE').write_text(now() + '\n')
    state.update({'status': 'complete', 'completed_at_utc': now(), 'result': result, 'validation': validation})
    save('eval_state.json', state)
except BaseException as exc:
    state.update({'status': 'failed', 'failed_at_utc': now(), 'error': repr(exc)})
    save('eval_state.json', state)
    raise
