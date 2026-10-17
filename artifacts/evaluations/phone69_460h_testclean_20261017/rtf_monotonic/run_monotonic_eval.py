from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import os
import subprocess
import sys
import time

repo = Path('/test21/conformer-rnnt-asr')
out = Path(__file__).resolve().parent
original = out.parent
plan = json.loads((out / 'run_plan.json').read_text())
started = time.perf_counter()
def state(status, **extra):
    value = {'status': status, 'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_monotonic_seconds': time.perf_counter() - started, **extra}
    tmp = out / 'eval_state.json.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(out / 'eval_state.json')
def run(args, log_name):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with (out / log_name).open('x') as log:
        completed = subprocess.run(args, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{log_name}: process exited {completed.returncode}')
try:
    comparisons = []
    for exp in plan['experiments']:
        name = exp['name']
        state('evaluating', active_experiment=name, timer='time.perf_counter')
        output = out / (name + '_per_sample.csv')
        run([sys.executable, '-u', str(out / 'eval_monotonic.py'), exp['config'], '--checkpoint', exp['checkpoint'], '--metric', 'per', '--out', str(output)], name + '.log')
        with output.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        with (original / output.name).open(newline='') as handle:
            prior = list(csv.DictReader(handle))
        assert len(rows) == len(prior) == 2620
        mismatches = []
        for idx, (row, old) in enumerate(zip(rows, prior)):
            assert int(row['idx']) == int(old['idx']) == idx
            assert row['ref'] == old['ref']
            assert math.isfinite(float(row['RTF'])) and float(row['RTF']) >= 0
            if row['hyp'] != old['hyp'] or row['phone_errors'] != old['phone_errors']:
                mismatches.append(idx)
        comparisons.append({'experiment': name, 'num_samples': len(rows), 'prediction_mismatch_indices': mismatches, 'mean_utterance_RTF': sum(float(row['RTF']) for row in rows) / len(rows)})
        (out / 'prediction_comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
        assert not mismatches, f'{name}: predictions changed; preserve both runs for investigation'
    (out / 'RAW_EVALUATIONS_COMPLETE').touch()
    state('aggregating', timer='time.perf_counter')
    run([sys.executable, '-u', str(out / 'aggregate_phone_edits.py')], 'aggregation.log')
    state('validating', timer='time.perf_counter')
    run([sys.executable, '-u', str(out / 'validate_results.py')], 'validation.log')
    state('complete', timer='time.perf_counter', summary=str(out / 'summary.json'), validation=str(out / 'validation_report.json'))
except Exception as error:
    state('failed', reason=str(error), timer='time.perf_counter')
    raise
