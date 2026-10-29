#!/usr/bin/env python3
"""Preflight and serially train the three approved stride1 placement models."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
AUDIT = REPO / 'artifacts/runs/phone69_460h_placement_stride1_20261022'
PHASE = 'placement_stride1_20261022'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def state(status, **fields):
    value = dict(status=status, run_phase=PHASE, pid=os.getpid(),
                 server_utc=datetime.now(timezone.utc).isoformat(), **fields)
    save(AUDIT / 'queue_state.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def protected_unchanged():
    inventory = json.loads((AUDIT / 'protected_manifest.json').read_text())
    for root, files in inventory['experiments'].items():
        actual = {str(p.relative_to(REPO)) for p in (REPO / root).rglob('*') if p.is_file()}
        assert actual == {row['path'] for row in files}, root
        for row in files:
            p = REPO / row['path']
            stat = p.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (row['bytes'], row['mtime_ns']), row['path']
            if row.get('sha256'):
                assert sha(p) == row['sha256'], row['path']
    for path, fingerprint in inventory['configs'].items():
        assert sha(REPO / path) == fingerprint, path


def gpu_idle():
    result = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'
    ], text=True).strip()
    assert not result, 'GPU is occupied; will not interrupt another process: ' + result


def run_logged(arguments, filename):
    path = AUDIT / filename
    with path.open('x') as stream:
        result = subprocess.run([sys.executable, '-u', *arguments], cwd=REPO,
                                stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def expected_definition(config):
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    return LibriSpeechRNNTModule.from_config(config).offline_native_definition


def check_complete(item, definition):
    import torch
    root = REPO / item['experiment']
    run = json.loads((root / 'run.json').read_text())
    assert run['status'] == 'trained' and run['epoch'] == 50
    assert run['offline_native_definition'] == definition
    boundary = run['epoch_boundary_checkpoint']
    records = []
    for name, path in [('best', run['best_checkpoint']), ('epoch_boundary', boundary)]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint.get('state_dict') and checkpoint.get('optimizer_states')
        assert checkpoint.get('lr_schedulers')
        assert checkpoint.get('offline_native_definition') == definition
        assert all(not value.is_floating_point() or torch.isfinite(value).all()
                   for value in checkpoint['state_dict'].values() if torch.is_tensor(value))
        if name == 'epoch_boundary':
            current = checkpoint['loops']['fit_loop']['epoch_progress']['current']
            assert current['ready'] == current['completed']
            assert checkpoint['phone69_epoch_boundary']['resume_completed_epochs'] == 50
        records.append(dict(kind=name, path=path, full_model_optimizer_scheduler=True,
                            definition_matches=True, model_finite=True))
        del checkpoint
    return dict(scheme=item['scheme'], epoch=50, global_step=run['global_step'], checkpoints=records)


def main(reuse_passed_preflight=False):
    os.chdir(REPO)
    os.environ.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    plan = json.loads((AUDIT / 'queue_plan.json').read_text())
    assert [x['scheme'] for x in plan['experiments']] == ['mel', 'layer5', 'layers2_5_n3']
    with open('/tmp/phone69_460h_serial.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protected_unchanged()
        gpu_idle()
        for item in plan['experiments']:
            root = REPO / item['experiment']
            assert not (root / 'run.json').exists() and not list((root / 'checkpoints').glob('*.ckpt'))
        state('preflight', stage='regression_tests')
        suffix = '_resume_preflight' if reuse_passed_preflight else ''
        assert run_logged(['-m', 'unittest', 'discover', '-s', 'tests'], 'tests' + suffix + '.log') == 0, 'Regression tests failed'
        configs = [item['config'] for item in plan['experiments']]
        assert run_logged(['scripts/server/validate_phone69_placement.py', '--configs', *configs,
                           '--out', str(AUDIT / ('synthetic_validation' + suffix + '.json'))], 'synthetic_validation' + suffix + '.log') == 0
        accepted = []
        for item in plan['experiments']:
            config_path = REPO / item['config']
            if reuse_passed_preflight:
                previous = []
                for path in AUDIT.glob('probe_' + item['scheme'] + '_b*.json'):
                    report = json.loads(path.read_text())
                    if report['status'] == 'passed' and report['config_sha256'] == sha(config_path):
                        assert all(sha(REPO / p) == value for p, value in report['source_sha256'].items())
                        assert report['budget'] == yaml.safe_load(config_path.read_text())['data']['max_token_limit']
                        previous.append((path, report))
                assert len(previous) == 1, 'No unique passed preflight matching the exact current configuration'
                path, report = previous[0]
                accepted.append(dict(**item, budget=report['budget'], probe=str(path.relative_to(REPO)),
                                     config_sha256=sha(config_path),
                                     peak_reserved_GiB=max(row['peak_reserved_GiB'] for row in report['results'])))
                continue
            for budget in plan['batch_candidates']:
                gpu_idle()
                state('preflight', stage='real_batch_memory', experiment=item['experiment'], budget=budget)
                text = config_path.read_text()
                text, replacements = re.subn(r'(?m)^  max_token_limit: \d+$',
                                              '  max_token_limit: ' + str(budget), text)
                assert replacements == 1
                config_path.write_text(text)
                stem = 'probe_' + item['scheme'] + '_b' + str(budget)
                report_path = AUDIT / (stem + '.json')
                rc = run_logged(['scripts/server/probe_phone69_training_memory.py', item['config'],
                                 '--out', str(report_path), '--keep-cache',
                                 '--max-reserved-fraction', str(plan['max_reserved_fraction'])], stem + '.log')
                report = json.loads(report_path.read_text())
                if rc == 0 and report['status'] == 'passed':
                    assert report['results'] and report['budget'] == budget
                    assert report['config_sha256'] == sha(config_path)
                    accepted.append(dict(**item, budget=budget, probe=str(report_path.relative_to(REPO)),
                                         config_sha256=sha(config_path),
                                         peak_reserved_GiB=max(row['peak_reserved_GiB'] for row in report['results'])))
                    break
                reducible = report['status'] in ('oom', 'unsafe_joint') or 'reserved-memory safety margin' in report.get('error', '')
                assert reducible, 'Probe failed; preserved report: ' + str(report_path)
            else:
                raise RuntimeError('No safe batch budget found for ' + item['scheme'])
        sources = sorted((REPO / 'conformer_rnnt').rglob('*.py')) + [REPO / p for p in [
            'common.py', 'train.py', 'scripts/server/train_phone69_460h_one.py',
            'scripts/server/phone69_epoch_checkpoint.py', 'scripts/server/probe_phone69_training_memory.py',
            'scripts/server/phone69_placement_pipeline.py']]
        definitions = {}
        for item in accepted:
            config = yaml.safe_load((REPO / item['config']).read_text())
            assert config['model']['sequence_mode'] == 'offline'
            assert config['model']['time_reduction_stride'] == 1
            assert config['model']['time_reduction_input_dim'] == 128
            assert config['paths']['from_checkpoint'] is None
            assert config['paths']['exp_dir'] == str(REPO / item['experiment'])
            definitions[item['scheme']] = expected_definition(config)
        gate = dict(valid=True, run_phase=PHASE, models=accepted, definitions=definitions,
                    source_sha256={str(p.relative_to(REPO)): sha(p) for p in sources},
                    server_utc=datetime.now(timezone.utc).isoformat())
        protected_unchanged()
        save(AUDIT / 'prelaunch_validation.json', gate)
        completed = []
        for item in accepted:
            gpu_idle()
            protected_unchanged()
            assert all(sha(REPO / path) == value for path, value in gate['source_sha256'].items())
            assert all(sha(REPO / row['config']) == row['config_sha256'] for row in accepted)
            root = REPO / item['experiment']
            assert not (root / 'run.json').exists() and not list((root / 'checkpoints').glob('*.ckpt'))
            (root / 'logs').mkdir(parents=True, exist_ok=True)
            state('training', experiment=item['experiment'], budget=item['budget'], completed=completed)
            with (root / 'logs/train.log').open('x') as stream, (root / 'logs/gpu_usage.csv').open('x') as gpu_log:
                child = subprocess.Popen([sys.executable, '-u', 'scripts/server/train_phone69_460h_one.py', item['config']],
                                         cwd=REPO, env=dict(os.environ, RESUME_LAST='0'), stdout=stream, stderr=subprocess.STDOUT)
                state('training', experiment=item['experiment'], budget=item['budget'], training_pid=child.pid, completed=completed)
                while child.poll() is None:
                    subprocess.run(['nvidia-smi', '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader', '-i', '0'], stdout=gpu_log, stderr=subprocess.DEVNULL)
                    gpu_log.flush()
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        pass
            assert child.returncode == 0, item['scheme'] + ' training failed; logs preserved'
            result = check_complete(item, definitions[item['scheme']])
            save(AUDIT / ('complete_' + item['scheme'] + '.json'), result)
            (root / 'TRAINING_COMPLETE').touch(exist_ok=False)
            completed.append(result)
            protected_unchanged()
        save(AUDIT / 'completion_validation.json', dict(valid=True, models=completed, protected_artifacts_unchanged=True))
        (AUDIT / 'TRAINING_QUEUE_COMPLETE').touch(exist_ok=False)
        state('complete', completed=completed)


if __name__ == '__main__':
    if sys.argv[1:] == ['--check-imports']:
        from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
        print('Standalone placement imports passed')
        raise SystemExit(0)
    try:
        assert sys.argv[1:] in ([], ['--reuse-passed-preflight'])
        main(reuse_passed_preflight=bool(sys.argv[1:]))
    except BaseException as error:
        state('failed', error=repr(error), message='Preserved all logs and checkpoints; no automatic restart')
        raise
