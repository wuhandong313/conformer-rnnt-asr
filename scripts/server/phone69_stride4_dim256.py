#!/usr/bin/env python3
"""Train the 256-wide stride-4 baseline after the returned placement queue."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.server.append_phone69_placement_queue import NAMES, matches, predecessor_done, read
from scripts.server.phone69_placement_supplement import live_jobs, now, save, sha
from scripts.server.recover_phone69_stride_sweep import boundary_metadata


EXPERIMENT = 'exp_phone69_460h_offline_native_stride4_none_dim256_s42_e50'
CONFIG = Path(f'configs/server_{EXPERIMENT.removeprefix("exp_")}.yaml')
REFERENCE = Path('artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111'
                 '/stride2_proj128/training_config.yaml')
AUDIT = Path('output/phone69_460h_stride4_dim256_20261119')
PRIOR = Path('output/phone69_460h_placement_current_server_three_20261115')
PRIOR_SCRIPT = 'append_phone69_placement_queue.py'
PRIOR_STATE = Path('output/phone69_460h_placement_append_20261115/append_state.json')
LOCK = Path('/tmp/phone69_460h_serial.lock')
TRAIN = 'scripts/server/train_phone69_460h_one.py'
PROBE = 'scripts/server/probe_phone69_training_memory.py'


def recipe():
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    expected = yaml.safe_load((ROOT / REFERENCE).read_text())
    expected['model'].update(time_reduction_stride=4, time_reduction_input_dim=64)
    expected['paths']['exp_dir'] = str(Path(expected['paths']['exp_dir']).with_name(EXPERIMENT))
    if config != expected:
        raise ValueError('The recipe must match the trained stride-2 baseline apart from the stack and output path')
    return config


def verified_predecessor(queue, completion):
    models = (completion or {}).get('models', [])
    required_models = [row for row in models if row['name'] in NAMES]
    if ((queue or {}).get('status') != 'complete'
            or models != queue.get('completed') or [row['name'] for row in required_models] != NAMES):
        raise RuntimeError('The preceding placement queue lacks a matching completion record')
    for row in required_models:
        boundary = row['epoch_boundary_metadata']
        if (row['status'] != 'trained' or row['epoch'] != 50
                or boundary['resume_completed_epochs'] != 50 or not boundary['resume_at_epoch_boundary']
                or boundary['resume_global_step'] != row['global_step']
                or not all(Path(row[key]).is_file() for key in ('best_checkpoint', 'epoch_boundary_checkpoint'))):
            raise RuntimeError(f"Incomplete predecessor: {row['name']}")
    return [row['name'] for row in required_models]


class Stride4Queue:
    def __init__(self):
        config = recipe()
        self.exp = Path(config['paths']['exp_dir'])
        if self.exp.parent != ROOT or self.exp.exists():
            raise ValueError('Expected a new experiment directory in the training checkout')
        self.budget = config['data']['max_token_limit']
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip()
        paths = [CONFIG, REFERENCE, Path(config['tokenizer']['vocab_path']),
                 Path(__file__).resolve().relative_to(ROOT)]
        paths += [Path(p) for p in ('common.py', 'train.py', 'phone69_tokenizer.py', TRAIN, PROBE,
                                   'scripts/server/phone69_epoch_checkpoint.py',
                                   'scripts/server/phone69_placement_supplement.py',
                                   'scripts/server/phone69_stride_sweep.py',
                                   f'scripts/server/{PRIOR_SCRIPT}',
                                   'scripts/server/recover_phone69_stride_sweep.py')]
        paths += sorted(p.relative_to(ROOT) for p in (ROOT / 'conformer_rnnt').rglob('*.py'))
        self.sources = {str(path): sha(ROOT / path) for path in paths}
        self.audit = ROOT / AUDIT
        self.audit.mkdir(parents=True, exist_ok=False)
        save(self.audit / 'deployment.json', dict(config=str(CONFIG), after=str(PRIOR),
             code_commit=self.commit, source_sha256=self.sources, controller_pid=os.getpid()))

    def state(self, status, **details):
        value = dict(status=status, experiment='stride4_dim256', stride=4, projection_dim=64,
                     encoder_dim=256, budget=self.budget, controller_pid=os.getpid(),
                     code_commit=self.commit, server_utc=now(), **details)
        save(self.audit / 'queue_state.json', value)
        print(json.dumps(value), flush=True)

    def unchanged(self):
        if any(sha(ROOT / path) != digest for path, digest in self.sources.items()):
            raise RuntimeError('Training code or recipe changed since registration')

    def wait_previous(self):
        initial = read(ROOT / PRIOR_STATE)
        if not initial:
            raise RuntimeError('The preceding placement controller is missing')
        pid = initial['controller_pid']
        while True:
            alive = matches(pid, PRIOR_SCRIPT)
            state = read(ROOT / PRIOR_STATE)
            queue = read(ROOT / PRIOR / 'queue_state.json')
            if not state or state['controller_pid'] != pid or (queue and queue['controller_pid'] != pid):
                raise RuntimeError('The preceding placement controller identity changed')
            if predecessor_done(state, alive):
                completed = verified_predecessor(queue, read(ROOT / PRIOR / 'completion.json'))
                self.state('waiting_for_training_lock', completed_predecessor=completed)
                return
            self.state('waiting_for_predecessor', predecessor_pid=pid,
                       predecessor_experiment=(queue or {}).get('experiment'),
                       predecessor_training_pid=(queue or {}).get('training_pid'))
            time.sleep(30)

    def wait_idle(self):
        while True:
            workers = [job['pid'] for job in live_jobs() if job['kind'] != 'controller']
            gpu = subprocess.check_output(
                ['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                text=True, timeout=20).split()
            if not workers and not gpu:
                return
            self.state('waiting_for_gpu', worker_pids=workers, gpu_pids=gpu)
            time.sleep(15)

    def confirm_batch(self):
        report_path = self.audit / 'batch_confirmation.json'
        command = [sys.executable, '-u', PROBE, str(CONFIG), '--out', str(report_path),
                   '--keep-cache', '--max-reserved-fraction', '0.96', '--rounds', '4',
                   '--random-batches', '3']
        with (self.audit / 'batch_confirmation.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            self.state('calibrating', calibration_pid=child.pid, report=str(report_path))
            code = child.wait()
        report = read(report_path) or {}
        if (code or report.get('status') != 'passed' or report.get('budget') != self.budget
                or report.get('config_sha256') != self.sources[str(CONFIG)]
                or len(report.get('results', [])) < 24):
            raise RuntimeError(f'The matched batch did not pass confirmation. See {report_path}')

    def train(self):
        log_dir = self.exp / 'logs'
        self.exp.mkdir(exist_ok=False)
        log_dir.mkdir()
        train_log = log_dir / 'train.log'
        with train_log.open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', TRAIN, str(CONFIG)], cwd=ROOT,
                                     env=dict(os.environ, RESUME_LAST='0'),
                                     stdout=log, stderr=subprocess.STDOUT)
            self.state('training', training_pid=child.pid, train_log=str(train_log))
            if child.wait():
                raise RuntimeError(f'Training failed. See {train_log}')
        self.unchanged()
        run = read(self.exp / 'run.json') or {}
        if (run.get('status') != 'trained' or run.get('epoch') != 50 or run.get('internal_dim') != 256
                or run.get('time_reduction_stride') != 4 or run.get('max_token_limit') != self.budget
                or run.get('config_sha256') != self.sources[str(CONFIG)]):
            raise RuntimeError('The finished run does not match the recorded configuration')
        best = Path(run['best_checkpoint'])
        if not best.is_file() or (self.exp / 'best_checkpoint.txt').read_text().strip() != str(best):
            raise RuntimeError('Missing verified best validation checkpoint')
        boundary = Path(run['epoch_boundary_checkpoint'])
        metadata = boundary_metadata(boundary, self.budget)
        if metadata['resume_completed_epochs'] != 50 or metadata['resume_global_step'] != run['global_step']:
            raise RuntimeError('The final checkpoint does not contain the completed 50th epoch')
        result = dict(training_commit=run['code_commit'], global_step=run['global_step'],
                      epoch=50, best_checkpoint=str(best), best_checkpoint_sha256=sha(best),
                      epoch_boundary_checkpoint=str(boundary), epoch_boundary_metadata=metadata)
        save(self.audit / 'completion.json', result)
        self.state('complete', **result)

    def run(self):
        self.wait_previous()
        with LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.wait_idle()
            self.unchanged()
            if self.exp.exists():
                raise FileExistsError(self.exp)
            self.confirm_batch()
            self.wait_idle()
            self.unchanged()
            self.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('validate', 'status', 'run'), required=True)
    args = parser.parse_args()
    if args.mode == 'validate':
        config = recipe()
        print(json.dumps(dict(config=str(CONFIG), budget=config['data']['max_token_limit'])))
    elif args.mode == 'status':
        print(json.dumps(dict(state=read(ROOT / AUDIT / 'queue_state.json'),
                              predecessor=read(ROOT / PRIOR / 'queue_state.json'))))
    else:
        os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0',
                          OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
        queue = Stride4Queue()
        try:
            queue.run()
        except BaseException as error:
            queue.state('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
