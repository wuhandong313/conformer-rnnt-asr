#!/usr/bin/env python3
"""Serially calibrate and train missing, position-only Phone69 experiments.

Calibration executes disposable real-data optimizer steps. This entry point
never retries failed training, changes a backend, or overwrites an experiment.
The validate/status modes do not import PyTorch or start GPU work.
"""

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.server.phone69_stride_sweep import maximize_budget


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def local_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Manifest path escapes the repository')
    return path


def validate_recipe(config, baseline, positions, *, fresh):
    """Every substantive recipe setting is inherited from its matching baseline."""
    expected = copy.deepcopy(baseline)
    stages = expected['model']['compression']['stages']
    if len(positions) != len(stages) or positions != sorted(set(positions)):
        raise ValueError('Positions must identify every stage in increasing order')
    if any(type(p) is not int or not 0 <= p < 10 for p in positions):
        raise ValueError('Positions must be Mel=0 or an interior layer 1..9')
    for stage, position in zip(stages, positions):
        stage['after_layer'] = position
    expected['paths']['exp_dir'] = config['paths']['exp_dir']
    expected['paths']['from_checkpoint'] = config['paths'].get('from_checkpoint')
    expected['data']['max_token_limit'] = config['data']['max_token_limit']
    if config != expected:
        raise ValueError('Recipe changes settings beyond stage positions, output, resume, or token budget')
    if fresh and config['paths'].get('from_checkpoint') is not None:
        raise ValueError('Supplementary training must start from scratch')
    model = config['model']
    if (model['sequence_mode'], model['num_layers'], model['time_reduction_stride'],
            model['time_reduction_input_dim'], config['seed'], config['training']['epochs']) != (
            'offline', 10, 1, 128, 42, 50):
        raise ValueError('Expected the corresponding stride1, projection128, 10-layer, seed42, 50-epoch recipe')
    if model['compression'].get('execution_backend', 'reference') != 'reference':
        raise ValueError('Preserve the reference execution backend')
    if config['tokenizer']['type'] != 'phone69' or config['trainer'].get('fast_dev_run'):
        raise ValueError('Expected full Phone69 training')
    return config


def load_plan(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('The manifest must be inside the repository')
    plan = json.loads(path.read_text())
    local_path(plan['runtime_dir'])
    search = plan['batch_search']
    if not (0 < search['minimum'] <= search['initial'] <= search['maximum']
            and search['quantum'] > 0 and search['growth'] > 1
            and 0 < search['max_reserved_fraction'] <= 1):
        raise ValueError('Invalid memory calibration search limits')
    names, outputs = set(), set()
    if not plan['experiments']:
        raise ValueError('No experiments in the supplement manifest')
    for item in plan['experiments']:
        if not re.fullmatch(r'[a-zA-Z0-9_]+', item['name']) or item['name'] in names:
            raise ValueError('Experiment names must be safe and unique')
        names.add(item['name'])
        if item['action'] not in ('train', 'reuse'):
            raise ValueError('Experiment action must be train or reuse')
        cfg = yaml.safe_load(local_path(item['config']).read_text())
        baseline = yaml.safe_load(local_path(item['baseline_config']).read_text())
        validate_recipe(cfg, baseline, item['positions'], fresh=item['action'] == 'train')
        output = item.get('experiment', cfg['paths']['exp_dir'])
        if item['action'] == 'train' and output != cfg['paths']['exp_dir']:
            raise ValueError('Fresh experiment output must match its YAML')
        if output in outputs:
            raise ValueError('Experiment directories must be distinct')
        outputs.add(output)
        initial = item.get('initial_budget', search['initial'])
        if not search['minimum'] <= initial <= search['maximum']:
            raise ValueError('Initial budget outside the search limits')
    return path, plan


def live_jobs():
    """Return only job identities, never process commands or environment values."""
    jobs = []
    controllers = {'phone69_stride_sweep.py', 'phone69_placement_pipeline.py',
                   'phone69_placement_supplement.py', 'recover_phone69_stride_sweep.py',
                   'phone69_triple_compression.py'}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(path.name)
            if pid == os.getpid() or (path / 'stat').read_text().split()[2] == 'Z':
                continue
            arguments = (path / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            names = {Path(a).name for a in arguments if a}
            if 'train_phone69_460h_one.py' in names:
                jobs.append(dict(pid=pid, kind='training'))
            elif 'probe_phone69_training_memory.py' in names:
                jobs.append(dict(pid=pid, kind='calibration'))
            elif names & controllers:
                # Read-only status/inspection processes do not own the GPU queue.
                if '--mode' in arguments and arguments[arguments.index('--mode') + 1:][:1] != ['run']:
                    continue
                jobs.append(dict(pid=pid, kind='controller'))
        except (OSError, ValueError, IndexError):
            pass
    return jobs


def verify_completed(item, *, budget=None):
    """Read the actual final boundary; a status flag alone is insufficient."""
    import torch
    from conformer_rnnt.checkpoint_utils import phone69_resume_metadata

    requested = yaml.safe_load(local_path(item['config']).read_text())
    experiment = Path(item.get('experiment', requested['paths']['exp_dir']))
    run = json.loads((experiment / 'run.json').read_text())
    if run.get('status') != 'trained' or run.get('epoch') != 50:
        raise ValueError('Existing experiment is not a verified 50-epoch completion: ' + item['name'])
    actual_path = Path(run['config'])
    if not actual_path.is_absolute():
        actual_path = ROOT / actual_path
    if sha(actual_path) != run['config_sha256']:
        raise ValueError('Actual run configuration SHA changed')
    actual = yaml.safe_load(actual_path.read_text())
    validate_recipe(actual, requested, item['positions'], fresh=item['action'] == 'train')
    if run['max_token_limit'] != actual['data']['max_token_limit']:
        raise ValueError('Actual run budget differs from its recorded recipe')
    if actual['model']['compression'] != run['compression']:
        raise ValueError('Actual run compression differs from its recipe')
    if budget is not None and run['max_token_limit'] != budget:
        raise ValueError('Training budget differs from the selected budget')
    if run['internal_dim'] != 128 or run['time_reduction_stride'] != 1:
        raise ValueError('Actual model dimensions differ from the baseline')
    best = Path(run['best_checkpoint'])
    boundary = Path(run['epoch_boundary_checkpoint'])
    if not best.is_file() or not boundary.is_file():
        raise ValueError('Missing best or final epoch-boundary checkpoint')
    if (experiment / 'best_checkpoint.txt').read_text().strip() != str(best):
        raise ValueError('Best-checkpoint marker disagrees with the run')
    best_sha = sha(best)
    if item.get('expected_best_sha256') and best_sha != item['expected_best_sha256']:
        raise ValueError('Best checkpoint differs from the frozen completed experiment')
    definition = run.get('offline_native_definition')
    if not definition:
        raise ValueError('Staged run does not record its placement definition')
    checkpoint = torch.load(boundary, map_location='cpu', weights_only=False)
    metadata = phone69_resume_metadata(checkpoint, run['max_token_limit'], expected_definition=definition)
    if (metadata['resume_completed_epochs'] != 50 or not metadata['resume_at_epoch_boundary']
            or metadata['resume_global_step'] != run['global_step'] or not checkpoint.get('state_dict')):
        raise ValueError('Final checkpoint is not the full 50-epoch training boundary')
    del checkpoint
    return dict(name=item['name'], action=item['action'], positions=item['positions'],
                status=run['status'], epoch=50, global_step=run['global_step'],
                budget=run['max_token_limit'], training_commit=run['code_commit'],
                config=str(actual_path), config_sha256=run['config_sha256'],
                best_checkpoint=str(best), best_checkpoint_sha256=best_sha,
                epoch_boundary_checkpoint=str(boundary), epoch_boundary_metadata=metadata)


class PlacementQueue:
    def __init__(self, manifest):
        self.manifest, self.plan = load_plan(manifest)
        self.audit = local_path(self.plan['runtime_dir'])
        self.audit.mkdir(parents=True, exist_ok=False)
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        paths = list((ROOT / 'conformer_rnnt').rglob('*.py'))
        paths += [ROOT / name for name in ['common.py', 'train.py', 'phone69_tokenizer.py',
            'scripts/server/train_phone69_460h_one.py', 'scripts/server/phone69_epoch_checkpoint.py',
            'scripts/server/probe_phone69_training_memory.py', 'scripts/server/phone69_stride_sweep.py',
            'scripts/server/phone69_placement_supplement.py']]
        paths += [self.manifest]
        paths += [local_path(item[key]) for item in self.plan['experiments'] for key in ('config', 'baseline_config')]
        self.sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}
        self.completed = []
        self.trial = 0
        save(self.audit / 'deployment.json', dict(code_commit=self.commit, source_sha256=self.sources,
                                                manifest=self.plan, controller_pid=os.getpid(), server_utc=now()))
        self.state('starting')

    def state(self, status, **fields):
        value = dict(status=status, controller_pid=os.getpid(), code_commit=self.commit,
                     server_utc=now(), completed=self.completed, **fields)
        save(self.audit / 'queue_state.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def unchanged(self):
        changed = [p for p, h in self.sources.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
        if changed:
            raise RuntimeError('Deployed source/configuration changed: ' + ', '.join(changed))

    def wait_idle(self):
        while True:
            # run() holds the shared lock here. Another compliant controller may
            # be waiting for it; treating that waiter as GPU work would deadlock
            # the lock holder. Real workers still block even before CUDA starts.
            jobs = [job for job in live_jobs() if job['kind'] != 'controller']
            gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                           '--format=csv,noheader,nounits'], text=True).strip()
            if not jobs and not gpu:
                return
            self.state('waiting_for_gpu', live_jobs=jobs, gpu_pids=gpu.splitlines())
            time.sleep(10)

    def probe(self, item, budget, *, confirmation=False, rounds=1):
        self.unchanged()
        self.wait_idle()
        self.trial += 1
        tag = '%s_b%d_%s_%03d' % (item['name'], budget, 'confirm' if confirmation else 'search', self.trial)
        output = self.audit / (tag + '.json')
        command = [sys.executable, '-u', 'scripts/server/probe_phone69_training_memory.py', item['config'],
                   '--out', str(output), '--budget', str(budget), '--keep-cache',
                   '--max-reserved-fraction', str(self.plan['batch_search']['max_reserved_fraction']),
                   '--rounds', str(rounds)]
        if confirmation:
            command += ['--random-batches', '3']
        with (self.audit / (tag + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            self.state('calibrating', experiment=item['name'], positions=item['positions'],
                       budget=budget, confirmation=confirmation, calibration_pid=child.pid,
                       report=str(output.relative_to(ROOT)))
            rc = child.wait()
        if not output.is_file():
            raise RuntimeError('Calibration did not produce a report: ' + tag)
        report = json.loads(output.read_text())
        if report['budget'] != budget or report['config_sha256'] != self.sources[item['config']]:
            raise RuntimeError('Calibration report identity mismatch')
        passed = rc == 0 and report['status'] == 'passed'
        reducible = (report['status'] in ('oom', 'unsafe_joint')
                     or 'reserved-memory safety margin' in report.get('error', ''))
        if not passed and not reducible:
            raise RuntimeError('Non-memory calibration failure; preserve and inspect ' + str(output))
        peaks = [r['peak_reserved_GiB'] for r in report.get('results', [])]
        result = dict(passed=passed, status=report['status'], report=str(output.relative_to(ROOT)),
                      peak_reserved_GiB=max(peaks, default=None),
                      batch_structure_sha256=report.get('batch_structure_sha256'),
                      optimizer_steps=len(report.get('results', [])),
                      selected_batches=len(report.get('selected_batches', [])))
        if passed and confirmation and result['optimizer_steps'] < 24:
            required = math.ceil(24 / max(1, result['selected_batches']))
            return self.probe(item, budget, confirmation=True, rounds=required)
        return result

    def tune(self, item):
        search = self.plan['batch_search']
        budget, trials, limiter = maximize_budget(lambda b: self.probe(item, b),
            initial=item.get('initial_budget', search['initial']), minimum=search['minimum'],
            maximum=search['maximum'], quantum=search['quantum'], growth=search['growth'])
        confirmations = []
        while budget >= search['minimum']:
            passed = next((t for t in reversed(trials) if t['passed'] and t['budget'] == budget), None)
            rounds = max(2, math.ceil(24 / max(1, passed['selected_batches']))) if passed else 4
            result = self.probe(item, budget, confirmation=True, rounds=rounds)
            confirmations.append(dict(budget=budget, **result))
            if result['passed']:
                break
            budget -= search['quantum']
        else:
            raise RuntimeError('No budget passed at least 24 repeated real-data optimizer steps')
        cfg = yaml.safe_load(local_path(item['config']).read_text())
        cfg['data']['max_token_limit'] = budget
        effective = self.audit / (item['name'] + '_train.yaml')
        with effective.open('x') as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)
        gate = dict(valid=True, name=item['name'], positions=item['positions'], budget=budget,
                    base_config=item['config'], base_config_sha256=self.sources[item['config']],
                    effective_config=str(effective.relative_to(ROOT)), effective_config_sha256=sha(effective),
                    code_commit=self.commit, source_sha256=self.sources, trials=trials,
                    confirmations=confirmations, limiter=limiter, budget_resolution=search['quantum'],
                    budget_claim='highest observed confirmed budget within the unchanged RNNT guard, search cap and resolution')
        save(self.audit / (item['name'] + '_selected.json'), gate)
        return effective, cfg, gate

    def train(self, item, effective, cfg, gate):
        self.unchanged()
        self.wait_idle()
        if sha(effective) != gate['effective_config_sha256']:
            raise RuntimeError('Selected configuration changed')
        experiment = Path(cfg['paths']['exp_dir'])
        if experiment.exists():
            raise FileExistsError('Refusing to reuse or overwrite a fresh experiment directory')
        log_dir = experiment / 'logs'
        log_dir.mkdir(parents=True)
        training_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with (log_dir / 'train.log').open('x') as log, (log_dir / 'gpu_usage.csv').open('x') as gpu:
            child = subprocess.Popen([sys.executable, '-u', 'scripts/server/train_phone69_460h_one.py', str(effective)],
                cwd=ROOT, env=dict(os.environ, RESUME_LAST='0'), stdout=log, stderr=subprocess.STDOUT)
            self.state('training', experiment=item['name'], positions=item['positions'], budget=gate['budget'],
                       training_pid=child.pid, training_commit=training_commit,
                       effective_config=str(effective.relative_to(ROOT)),
                       train_log=str(log_dir / 'train.log'), gpu_log=str(log_dir / 'gpu_usage.csv'))
            while child.poll() is None:
                subprocess.run(['nvidia-smi', '-i', '0',
                    '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'],
                    stdout=gpu, stderr=subprocess.DEVNULL)
                gpu.flush()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        if child.returncode:
            raise RuntimeError(item['name'] + ' training failed; logs/checkpoints preserved; no automatic retry')
        self.unchanged()
        run = json.loads((experiment / 'run.json').read_text())
        if run['config_sha256'] != gate['effective_config_sha256'] or run['code_commit'] != training_commit:
            raise RuntimeError('Actual training configuration or startup commit mismatch')
        result = verify_completed(item, budget=gate['budget'])
        self.completed.append(result)
        save(self.audit / (item['name'] + '_complete.json'), result)

    def run(self):
        for item in self.plan['experiments']:
            if item['action'] == 'train':
                cfg = yaml.safe_load(local_path(item['config']).read_text())
                if Path(cfg['paths']['exp_dir']).exists():
                    raise FileExistsError('Fresh experiment already exists: ' + item['name'])
            else:
                self.state('verifying_existing', experiment=item['name'])
                result = verify_completed(item)
                self.completed.append(result)
                save(self.audit / (item['name'] + '_reused.json'), result)
        with open('/tmp/phone69_460h_serial.lock', 'a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.state('waiting_for_training_lock')
                    time.sleep(10)
            for item in self.plan['experiments']:
                if item['action'] == 'reuse':
                    continue
                effective, cfg, gate = self.tune(item)
                self.train(item, effective, cfg, gate)
            save(self.audit / 'completion.json', dict(server_utc=now(), code_commit=self.commit, models=self.completed))
            self.state('complete')


def status(path, plan):
    audit = local_path(plan['runtime_dir'])
    def read(name):
        p = audit / name
        return json.loads(p.read_text()) if p.is_file() else None
    deployment = read('deployment.json')
    changed = [p for p, h in (deployment or {}).get('source_sha256', {}).items()
               if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    runs = {}
    for item in plan['experiments']:
        cfg = yaml.safe_load(local_path(item['config']).read_text())
        exp = Path(item.get('experiment', cfg['paths']['exp_dir']))
        p = exp / 'run.json'
        runs[item['name']] = json.loads(p.read_text()) if p.is_file() else None
    return dict(server_utc=now(), state=read('queue_state.json'), deployment=deployment,
                completion=read('completion.json'), live_jobs=live_jobs(), runs=runs,
                integrity_ok=deployment is not None and not changed, integrity_issues=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--mode', choices=('validate', 'run', 'status'), default='validate')
    args = parser.parse_args()
    path, plan = load_plan(args.manifest)
    if args.mode == 'validate':
        print(json.dumps(dict(valid=True, train=[i['name'] for i in plan['experiments'] if i['action'] == 'train'],
                              reuse=[i['name'] for i in plan['experiments'] if i['action'] == 'reuse'])))
    elif args.mode == 'status':
        print(json.dumps(status(path, plan), ensure_ascii=False))
    else:
        os.chdir(ROOT)
        os.environ.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
        queue = PlacementQueue(path)
        try:
            queue.run()
        except BaseException as error:
            queue.state('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
