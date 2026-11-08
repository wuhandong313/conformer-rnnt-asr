#!/usr/bin/env python3
"""Tune real-batch memory and serially train the approved fixed-stride sweep.

Unit/synthetic checks belong on the development machine. Server probes below
are disposable, real-data optimizer steps authorized as training calibration.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 'configs/server_phone69_460h_offline_native_stride4_none_dim512_s42_e50.yaml'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def validate_recipe(config, baseline, stride):
    """Only stride and experiment path change in the tracked baseline recipe."""
    expected = json.loads(json.dumps(baseline))
    expected['model']['time_reduction_stride'] = stride
    expected['paths']['exp_dir'] = config['paths']['exp_dir']
    if config != expected:
        raise ValueError('Recipe differs from stride4 baseline beyond stride/output path')
    m = config['model']
    assert m['time_reduction_input_dim'] == 128
    assert m['num_layers_block1'] + m['num_layers_block2'] == 10
    assert m['compression'] == {'filter': 'none'}
    assert config['paths']['from_checkpoint'] is None
    assert config['training']['epochs'] == 50
    assert config['seed'] == 42


def maximize_budget(probe, *, initial, minimum, maximum, quantum, growth):
    """Bracket the stable boundary, then refine to one token-budget quantum.

    Reports retain all measurements. Dynamic packing may be nonmonotonic; the
    return value is the highest observed pass, not a mathematical global bound.
    """
    if not (0 < minimum <= initial <= maximum and quantum > 0 and growth > 1):
        raise ValueError('Invalid batch search limits')
    low, high, trials = 0, None, []

    def attempt(budget):
        result = probe(budget)
        trials.append(dict(budget=budget, **result))
        return result

    candidate = initial
    while True:
        result = attempt(candidate)
        if result['passed']:
            low = candidate
            break
        high = candidate
        if candidate == minimum:
            raise RuntimeError('No stable batch budget, including minimum')
        candidate = max(minimum, (candidate // 2 // quantum) * quantum)

    if high is None:
        while low < maximum:
            candidate = min(maximum, math.ceil(low * growth / quantum) * quantum)
            result = attempt(candidate)
            if not result['passed']:
                high = candidate
                break
            previous = trials[-2].get('batch_structure_sha256')
            low = candidate
            if previous and result.get('batch_structure_sha256') == previous:
                return low, trials, 'rnnt_guard_plateau'
        if high is None:
            return low, trials, 'search_cap'

    while high - low > quantum:
        candidate = ((low + high) // 2 // quantum) * quantum
        if candidate <= low:
            break
        if attempt(candidate)['passed']:
            low = candidate
        else:
            high = candidate
    return low, trials, 'memory_or_joint_boundary'


class StrideQueue:
    def __init__(self, plan_path):
        self.plan_path = Path(plan_path).resolve()
        self.plan = yaml.safe_load(self.plan_path.read_text())
        self.audit = ROOT / self.plan['runtime_dir']
        self.audit.mkdir(parents=True, exist_ok=False)
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        source = sorted((ROOT / 'conformer_rnnt').rglob('*.py')) + [ROOT / p for p in (
            'common.py', 'train.py', 'phone69_tokenizer.py',
            'scripts/server/train_phone69_460h_one.py', 'scripts/server/phone69_epoch_checkpoint.py',
            'scripts/server/probe_phone69_training_memory.py', 'scripts/server/phone69_stride_sweep.py',
            BASELINE,
        )] + [ROOT / x['config'] for x in self.plan['experiments']] + [self.plan_path]
        self.sources = {str(p.relative_to(ROOT)): sha(p) for p in source}
        self.completed = []
        save(self.audit / 'deployment.json', dict(code_commit=self.commit, source_sha256=self.sources,
                                                plan=self.plan, pid=os.getpid()))
        self.state('starting')

    def state(self, status, **fields):
        value = dict(status=status, pid=os.getpid(), code_commit=self.commit,
                     server_utc=datetime.now(timezone.utc).isoformat(), completed=self.completed, **fields)
        save(self.audit / 'queue_state.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def unchanged(self):
        assert all(sha(ROOT / path) == value for path, value in self.sources.items()), 'Source/config changed during queue'

    def wait_gpu(self):
        while True:
            pids = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                           '--format=csv,noheader,nounits'], text=True).strip()
            if not pids:
                return
            self.state('waiting_for_gpu', gpu_pids=pids.splitlines())
            time.sleep(10)

    def run_logged(self, command, path, **kwargs):
        with path.open('x') as stream:
            return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, **kwargs).returncode

    def probe(self, item, budget, *, confirmation=False):
        self.unchanged()
        self.wait_gpu()
        tag = item['name'] + '_b' + str(budget) + ('_confirm' if confirmation else '')
        output = self.audit / (tag + '.json')
        command = [sys.executable, '-u', 'scripts/server/probe_phone69_training_memory.py', item['config'],
                   '--out', str(output), '--budget', str(budget), '--keep-cache',
                   '--max-reserved-fraction', str(self.plan['batch_search']['max_reserved_fraction']),
                   '--rounds', '2' if confirmation else '1']
        if confirmation:
            command += ['--random-batches', '3']
        self.state('tuning_batch', experiment=item['name'], stride=item['stride'],
                   budget=budget, confirmation=confirmation, report=str(output.relative_to(ROOT)))
        rc = self.run_logged(command, self.audit / (tag + '.log'))
        if not output.exists():
            raise RuntimeError('Probe did not create its report: ' + tag)
        report = json.loads(output.read_text())
        assert report['budget'] == budget
        assert report['config_sha256'] == self.sources[item['config']]
        passed = rc == 0 and report['status'] == 'passed'
        reducible = report['status'] in ('oom', 'unsafe_joint') or 'reserved-memory safety margin' in report.get('error', '')
        if not passed and not reducible:
            raise RuntimeError('Non-memory probe error, inspect ' + str(output))
        peaks = [r['peak_reserved_GiB'] for r in report.get('results', [])]
        return dict(passed=passed, status=report['status'], report=str(output.relative_to(ROOT)),
                    peak_reserved_GiB=max(peaks, default=None),
                    batch_structure_sha256=report.get('batch_structure_sha256'))

    def tune(self, item):
        search = self.plan['batch_search']
        budget, trials, limiter = maximize_budget(lambda b: self.probe(item, b),
            initial=search['initial'], minimum=search['minimum'], maximum=search['maximum'],
            quantum=search['quantum'], growth=search['growth'])
        confirmations = []
        while budget >= search['minimum']:
            result = self.probe(item, budget, confirmation=True)
            confirmations.append(dict(budget=budget, **result))
            if result['passed']:
                break
            budget -= search['quantum']
        else:
            raise RuntimeError('No budget passed the repeated real-batch confirmation')
        config = yaml.safe_load((ROOT / item['config']).read_text())
        config['data']['max_token_limit'] = budget
        effective = self.audit / (item['name'] + '_train.yaml')
        effective.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        gate = dict(valid=True, name=item['name'], stride=item['stride'], projection_dim=128,
                    encoder_dim=128 * item['stride'], encoder_layers=10, budget=budget,
                    base_config=item['config'], base_config_sha256=self.sources[item['config']],
                    effective_config=str(effective.relative_to(ROOT)), effective_config_sha256=sha(effective),
                    code_commit=self.commit, source_sha256=self.sources, trials=trials,
                    confirmations=confirmations, limiter=limiter, budget_resolution=search['quantum'])
        save(self.audit / (item['name'] + '_selected.json'), gate)
        return effective, config, gate

    def train(self, item, effective, config, gate):
        self.unchanged()
        self.wait_gpu()
        assert sha(effective) == gate['effective_config_sha256']
        experiment = Path(config['paths']['exp_dir'])
        assert not experiment.exists(), 'Refusing to overwrite an existing experiment'
        log_dir = experiment / 'logs'
        log_dir.mkdir(parents=True)
        with (log_dir / 'train.log').open('x') as log, (log_dir / 'gpu_usage.csv').open('x') as gpu:
            child = subprocess.Popen([sys.executable, '-u', 'scripts/server/train_phone69_460h_one.py', str(effective)],
                                     cwd=ROOT, env=dict(os.environ, RESUME_LAST='0'), stdout=log, stderr=subprocess.STDOUT)
            self.state('training', experiment=item['name'], stride=item['stride'], budget=gate['budget'],
                       training_pid=child.pid, effective_config=str(effective.relative_to(ROOT)),
                       train_log=str(log_dir / 'train.log'), pending=[x['name'] for x in self.plan['experiments']
                           if x['name'] not in [y['name'] for y in self.completed] and x['name'] != item['name']])
            while child.poll() is None:
                subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu',
                                '--format=csv,noheader'], stdout=gpu, stderr=subprocess.DEVNULL)
                gpu.flush()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        if child.returncode:
            raise RuntimeError(item['name'] + ' training failed; preserved checkpoints and log; queue stopped')
        run = json.loads((experiment / 'run.json').read_text())
        assert run['status'] == 'trained' and run['epoch'] == 50
        assert run['max_token_limit'] == gate['budget'] and run['internal_dim'] == 128 * item['stride']
        assert run['config_sha256'] == gate['effective_config_sha256']
        assert Path(run['best_checkpoint']).is_file()
        result = dict(name=item['name'], stride=item['stride'], budget=gate['budget'],
                      best_checkpoint=run['best_checkpoint'], global_step=run['global_step'])
        self.completed.append(result)
        save(self.audit / (item['name'] + '_complete.json'), result)

    def main(self):
        baseline = yaml.safe_load((ROOT / BASELINE).read_text())
        assert [x['stride'] for x in self.plan['experiments']] == [1, 2, 8]
        for item in self.plan['experiments']:
            config = yaml.safe_load((ROOT / item['config']).read_text())
            validate_recipe(config, baseline, item['stride'])
            assert not Path(config['paths']['exp_dir']).exists(), 'Experiment exists before queue start'
        with open('/tmp/phone69_460h_serial.lock', 'a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.state('waiting_for_training_lock')
                    time.sleep(10)
            for item in self.plan['experiments']:
                effective, config, gate = self.tune(item)
                self.train(item, effective, config, gate)
            self.state('complete')
            save(self.audit / 'completion.json', dict(models=self.completed, code_commit=self.commit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
    queue = StrideQueue(args.plan)
    try:
        queue.main()
    except BaseException as error:
        queue.state('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
