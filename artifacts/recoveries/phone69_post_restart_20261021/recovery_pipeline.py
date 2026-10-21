"""Restore the recorded runtime, validate protected models, then evaluate only."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path('/test21/conformer-rnnt-asr')
HERE = ROOT / 'artifacts/recoveries/phone69_post_restart_20261021'
OLD = ROOT / 'artifacts/evaluations/phone69_460h_five_frames_20261020'
OUT = ROOT / 'artifacts/evaluations/phone69_460h_five_frames_20261021_post_restart'
AUDIT = ROOT / 'artifacts/runs/phone69_460h_native_offline_five_frames_20261018'
PYTHON = Path('/root/anaconda3/envs/whd/bin/python')
STATE_PATH = HERE / 'recovery_state.json'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(path, value):
    value['updated_at_utc'] = now()
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

state = json.loads(STATE_PATH.read_text())
state['pid'] = os.getpid()
def stage(name, **fields):
    state.update(status='running', stage=name, **fields)
    save(STATE_PATH, state)
    print(now(), name, flush=True)

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

def active_evaluators():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / 'stat').read_text().split()[2] == 'Z':
                continue
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'eval_monotonic.py' in command and (str(OLD) in command or str(OUT) in command):
                found.append({'pid': int(proc.name), 'command': command})
        except (OSError, IndexError):
            pass
    return found

def checked(command, filename, env=None, timeout=3600):
    with (HERE / filename).open('a') as log:
        log.write(now() + ' start\n'); log.flush()
        result = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(filename + ' returned ' + str(result.returncode))

def json_command(command, filename, env):
    with (HERE / (filename + '.stderr.log')).open('a') as errors:
        result = subprocess.run(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                stderr=errors, universal_newlines=True, timeout=900)
    (HERE / filename).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(filename + ' returned ' + str(result.returncode))
    return json.loads(result.stdout)

try:
    os.chdir(str(ROOT))
    stage('protected_artifact_precheck')
    queue = json.loads((AUDIT / 'queue_state.json').read_text())
    assert queue['status'] == 'complete' and not queue.get('error')
    assert not active_evaluators(), 'Existing evaluation process remains'
    before = json.loads((AUDIT / 'baseline_preserved_before.json').read_text())
    after = json.loads((AUDIT / 'baseline_preserved_after.json').read_text())
    assert after['valid'] and after['all_sha256_size_mtime_unchanged']
    assert len(before['files']) == 63
    for entry in before['files']:
        path = ROOT / entry['path']; st = path.stat()
        assert st.st_size == entry['bytes'] and st.st_mtime_ns == entry['mtime_ns'], str(path)
        assert digest(path) == entry['sha256'], str(path)
    plan = json.loads((OLD / 'run_plan.json').read_text())
    assert len(plan['experiments']) == 4 and plan['parallelism'] == 4
    assert not (OUT / 'launch_state.json').exists(), 'New attempt already launched'
    for item in plan['experiments']:
        run = json.loads((ROOT / item['experiment'] / 'run.json').read_text())
        assert run['status'] == 'trained' and run['epoch'] == 50
        assert (ROOT / item['experiment'] / 'TRAINING_COMPLETE').is_file()
        assert run['best_checkpoint'] == item['checkpoint']
        assert digest(item['checkpoint']) == item['checkpoint_sha256']
        assert digest(ROOT / item['config']) == item['config_sha256']
    save(HERE / 'protected_artifacts_precheck.json', {'valid': True, 'baseline_files_unchanged': 63,
         'queue_complete': True, 'models': plan['experiments'], 'old_evaluators_running': []})

    # Extract the exact successful package versions from the preserved install log.
    log = (ROOT / 'artifacts/logs/env_rebuild.log').read_text(errors='replace')
    success = [line for line in log.splitlines() if line.startswith('Successfully installed ')][-1]
    pins = dict(token.rsplit('-', 1) for token in success[len('Successfully installed '):].split())
    assert pins['torch'] == pins['torchaudio'] == '2.6.0'
    assert pins['pytorch-lightning'] == '2.6.6'
    if 'Requirement already satisfied: packaging' in log and '(26.3)' in log:
        pins['packaging'] = '26.3'
    requirements = HERE / 'requirements-restored.txt'
    requirements.write_text(''.join(k + '==' + v + '\n' for k, v in sorted(pins.items())))
    stage('creating_python_environment', package_versions=pins, python_version='3.10.18')
    if not PYTHON.exists():
        checked(['/root/anaconda3/bin/conda', 'create', '-n', 'whd', 'python=3.10.18', '-y',
                 '--override-channels', '-c', 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main'],
                'conda_create.log', timeout=1800)
    assert PYTHON.exists()
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='0', PIP_DISABLE_PIP_VERSION_CHECK='1',
                       PIP_CACHE_DIR='/test21/.cache/phone69_runtime_pip')
    stage('installing_recorded_dependencies')
    checked([str(PYTHON), '-m', 'pip', 'install', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple',
             '--progress-bar', 'off', '--timeout', '60', '--retries', '3', '-r', str(requirements)],
            'pip_install.log', env=environment, timeout=7200)
    stage('verifying_runtime')
    checked([str(PYTHON), '-m', 'pip', 'check'], 'pip_check.log', env=environment, timeout=120)
    checked([str(PYTHON), '-m', 'pip', 'freeze'], 'pip_freeze.txt', env=environment, timeout=120)
    runtime = json_command([str(PYTHON), str(HERE / 'verify_runtime.py')], 'runtime_validation.json', environment)
    assert runtime['valid'] and runtime['rnnt_cuda_forward_backward_finite']
    stage('verifying_training_checkpoints', runtime=runtime)
    training = json_command([str(PYTHON), str(HERE / 'validate_training.py')], 'training_validation.json', environment)
    assert training['queue_complete'] and training['five_frames_definitions_consistent']
    assert training['baseline']['files_fingerprinted'] == 63 and training['baseline']['manifest_unchanged']
    assert not active_evaluators(), 'Evaluation appeared during environment recovery'

    stage('launching_four_model_test_clean')
    OUT.mkdir(parents=True, exist_ok=True)
    new_plan = copy.deepcopy(plan)
    new_plan.update(created_at_utc=now(), previous_attempt=str(OLD), recovery_directory=str(HERE),
                    authorization='2026-10-21 user requested environment reinstall and testing after restart',
                    runtime_validation=str(HERE / 'runtime_validation.json'),
                    training_validation=str(HERE / 'training_validation.json'))
    save(OUT / 'run_plan.json', new_plan)
    evaluation = {'status': 'launching', 'parallelism': 4, 'test_subset': 'test-clean', 'metric': 'per',
                  'beam_width': 20, 'previous_attempt': str(OLD), 'recovery_directory': str(HERE),
                  'queue_complete': True, 'baseline_manifest_unchanged': True, 'jobs': []}
    with (OUT / 'launch_state.json').open('x') as stream:
        json.dump(evaluation, stream, ensure_ascii=False, indent=2)
    for item in plan['experiments']:
        csv_path = OUT / (item['name'] + '_per_sample.csv')
        log_path = OUT / (item['name'] + '.log')
        arguments = [str(PYTHON), '-u', plan['source_evaluator'], item['config'],
                     '--checkpoint', item['checkpoint'], '--metric', 'per', '--out', str(csv_path)]
        with log_path.open('x') as stream:
            process = subprocess.Popen(arguments, cwd=str(ROOT), env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        job = dict(item, pid=process.pid, status='running', log=str(log_path),
                   per_sample_csv=str(csv_path), argv=arguments)
        evaluation['jobs'].append(job)
        save(OUT / 'launch_state.json', evaluation)
    evaluation['status'] = 'running'
    save(OUT / 'launch_state.json', evaluation)
    stage('evaluating', evaluation_directory=str(OUT), evaluation_pids=[j['pid'] for j in evaluation['jobs']])
    while True:
        checked([str(PYTHON), str(HERE / 'monitor_evaluation.py')], 'evaluation_monitor.log',
                env=environment, timeout=600)
        current = json.loads((OUT / 'launch_state.json').read_text())
        state['evaluation_status'] = current['status']
        state['last_evaluation_poll_utc'] = now()
        save(STATE_PATH, state)
        if current['status'] == 'complete':
            state.update(status='complete', stage='complete', completed_at_utc=now(),
                         summary=str(OUT / 'summary.json'), validation_report=str(OUT / 'validation_report.json'))
            save(STATE_PATH, state)
            print('RECOVERY_AND_EVALUATION_COMPLETE', now(), flush=True)
            break
        if current['status'] != 'running':
            raise RuntimeError('Evaluation state: ' + current['status'])
        time.sleep(30)
except Exception as error:
    state.update(status='failed', failed_stage=state.get('stage'), error=repr(error), failed_at_utc=now())
    save(STATE_PATH, state)
    traceback.print_exc()
    sys.exit(1)
