#!/usr/bin/env python3
"""Train a compute-matched stride-4 baseline after a registered Phone69 job."""

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
from scripts.server.phone69_placement_supplement import live_jobs, local_path, now, save, sha
from scripts.server.recover_phone69_stride_sweep import boundary_metadata


LOCK = Path('/tmp/phone69_460h_serial.lock')
TRAIN = ROOT / 'scripts/server/train_phone69_460h_one.py'


def read(path):
    return json.loads(path.read_text())


def checkout():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def await_predecessor(previous):
    state_path = previous / 'queue_state.json'
    while True:
        state = read(state_path)
        if state['status'] == 'complete':
            complete = read(previous / 'completion.json')
            if (complete['global_step'] != state['global_step']
                    or complete['epoch_boundary_metadata']['resume_completed_epochs'] != state['epochs']
                    or not Path(complete['best_checkpoint']).is_file()
                    or not Path(complete['epoch_boundary_checkpoint']).is_file()):
                raise RuntimeError('Previous queue completion does not match its checkpoints')
            return
        pid = state['controller_pid']
        try:
            stat = (Path('/proc') / str(pid) / 'stat').read_text()
            alive = stat.rsplit(')', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            alive = False
        if state['status'] == 'failed' or not alive:
            final_status = read(state_path)['status']
            if final_status == 'complete':
                continue
            raise RuntimeError(f'Previous queue stopped at {final_status}: {previous}')
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--audit', required=True)
    parser.add_argument('--after', required=True)
    args = parser.parse_args()
    config, audit, previous = map(local_path, (args.config, args.audit, args.after))
    cfg = yaml.safe_load(config.read_text())
    model = cfg['model']
    stride = model['time_reduction_stride']
    width = stride * model['time_reduction_input_dim']
    exp = local_path(cfg['paths']['exp_dir'])
    if (stride != 4 or width % model['num_heads'] or model['compression'] != {'filter': 'none'}
            or cfg['paths']['from_checkpoint'] is not None or exp.exists() or previous == audit):
        raise ValueError('Expected a new native stride-4 experiment')
    budget = cfg['data']['max_token_limit']
    epochs = cfg['training']['epochs']
    sources = [config, TRAIN, Path(__file__).resolve(), local_path(cfg['tokenizer']['vocab_path'])]
    sources += [ROOT / p for p in ('train.py', 'common.py', 'phone69_tokenizer.py',
                                   'scripts/server/phone69_epoch_checkpoint.py')]
    sources += sorted((ROOT / 'conformer_rnnt').rglob('*.py'))
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    config_name = str(config.relative_to(ROOT))
    identity = dict(experiment=f'stride4_dim{width}', config=config_name,
                    after=str(previous.relative_to(ROOT)), code_commit=checkout(),
                    controller_pid=os.getpid(), stride=stride, encoder_dim=width,
                    budget=budget, epochs=epochs)
    audit.mkdir(parents=True, exist_ok=False)
    save(audit / 'deployment.json', dict(**identity, source_sha256=hashes))

    def state(status, **details):
        record = dict(**identity, status=status, server_utc=now(), **details)
        save(audit / 'queue_state.json', record)
        print(json.dumps(record), flush=True)

    try:
        state('waiting_for_predecessor')
        await_predecessor(previous)
        state('waiting_for_training_lock')
        with LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while True:
                workers = [job['pid'] for job in live_jobs() if job['kind'] != 'controller']
                gpu = subprocess.check_output(
                    ['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                    text=True, timeout=20).split()
                if not workers and not gpu:
                    break
                state('waiting_for_gpu', worker_pids=workers, gpu_pids=gpu)
                time.sleep(15)
            if any(sha(ROOT / path) != digest for path, digest in hashes.items()):
                raise RuntimeError('Training source or recipe changed while the job was queued')

            startup_commit = checkout()
            exp.mkdir(exist_ok=False)
            log_dir = exp / 'logs'
            log_dir.mkdir()
            train_log = log_dir / 'train.log'
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0',
                       OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', RESUME_LAST='0')
            with train_log.open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', str(TRAIN), str(config)],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                state('training', training_pid=child.pid, training_commit=startup_commit,
                      train_log=str(train_log))
                if child.wait():
                    raise RuntimeError(f'Training exited with code {child.returncode}. See {train_log}')

            run = read(exp / 'run.json')
            actual = tuple(run.get(key) for key in (
                'status', 'epoch', 'internal_dim', 'time_reduction_stride',
                'max_token_limit', 'config_sha256', 'code_commit',
            ))
            if actual != ('trained', epochs, width, stride, budget, hashes[config_name], startup_commit):
                raise RuntimeError('The finished training run differs from the recorded configuration')
            best = Path(run['best_checkpoint'])
            if not best.is_file() or (exp / 'best_checkpoint.txt').read_text().strip() != str(best):
                raise RuntimeError('Missing verified best validation checkpoint')
            boundary = Path(run['epoch_boundary_checkpoint'])
            metadata = boundary_metadata(boundary, budget)
            if (metadata['resume_completed_epochs'] != epochs
                    or metadata['resume_global_step'] != run['global_step']):
                raise RuntimeError('The final checkpoint does not contain the completed training boundary')
            complete = dict(training_commit=startup_commit, epoch=epochs, global_step=run['global_step'],
                            best_checkpoint=str(best), best_checkpoint_sha256=sha(best),
                            epoch_boundary_checkpoint=str(boundary), epoch_boundary_metadata=metadata)
            save(audit / 'completion.json', complete)
            state('complete', **complete)
    except BaseException as error:
        state('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
