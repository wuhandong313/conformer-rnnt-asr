from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, subprocess, sys

repo = Path('/test21/conformer-rnnt-asr')
out = Path(__file__).resolve().parent
plan = json.loads((out / 'run_plan.json').read_text())
now = lambda: datetime.now(timezone.utc).isoformat()

def save(name, value):
    path = out / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)

state = {'status': 'running', 'started_at_utc': now(), 'models': {m['name']: 'queued' for m in plan['models']}, 'pids': {}}
save('analysis_state.json', state)
children = []
try:
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(repo) + os.pathsep + os.environ.get('PYTHONPATH', '')})
    for item in plan['models']:
        log = (out / (item['name'] + '.log')).open('x')
        child = subprocess.Popen([sys.executable, '-u', str(out / 'worker.py'), item['name']], cwd=str(repo), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children.append((item, child, log))
        state['models'][item['name']] = 'running'
        state['pids'][item['name']] = child.pid
    save('analysis_state.json', state)
    failures = []
    for item, child, log in children:
        rc = child.wait()
        log.close()
        state['models'][item['name']] = 'complete' if rc == 0 else 'failed'
        if rc != 0:
            failures.append({'name': item['name'], 'returncode': rc})
        save('analysis_state.json', state)
    if failures:
        raise RuntimeError('One or more measurements failed: ' + json.dumps(failures))
    results = [json.loads((out / (x['name'] + '.result.json')).read_text()) for x in plan['models']]
    assert all(x['status'] == 'complete' and x['num_samples'] == plan['expected_samples'] and x['reference_order_matches_previous_eval'] for x in results)
    assert len({x['input_sequence_sha256'] for x in results}) == 1
    assert len({x['frame_totals']['input_mel_frames'] for x in results}) == 1
    assert all(x['feature_signature'] == plan['feature_signature'] and x['definition_matches_checkpoint'] for x in results)
    for item in plan['models']:
        assert hashlib.sha256(Path(item['config']).read_bytes()).hexdigest() == item['config_sha256']
        assert hashlib.sha256(Path(item['checkpoint']).read_bytes()).hexdigest() == item['checkpoint_sha256']
    for result, item in zip(results, plan['models']):
        result['test_clean_PER'] = item['test_clean_PER']
    summary = {
        'valid': True, 'dataset': 'test-clean', 'samples': plan['expected_samples'],
        'reference_order_matches_previous_evaluations': True, 'feature_lengths_match_across_models': True,
        'device': 'cpu', 'threads_per_model': 2,
        'overall_retention_definition': 'sum of final encoder output frames / sum of original input Mel frames',
        'overall_reduction_definition': '1 - overall_retention',
        'stage_retention_definition': 'sum of stage output frames / sum of stage input frames',
        'counts_are_observed_on_test_clean': True, 'models': results, 'completed_at_utc': now(),
    }
    save('compression_summary.json', summary)
    (out / 'ANALYSIS_COMPLETE').write_text(now() + '\n')
    state.update(status='complete', completed_at_utc=now())
    save('analysis_state.json', state)
except BaseException as exc:
    state.update(status='failed', failed_at_utc=now(), error=repr(exc))
    save('analysis_state.json', state)
    raise
