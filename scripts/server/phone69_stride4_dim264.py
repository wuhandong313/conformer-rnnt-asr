#!/usr/bin/env python3
"""Queue the L1 compute-matched stride-4 recipe on the Phone69 GPU."""

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
from scripts.server.phone69_placement_supplement import live_jobs, now, save, sha
from scripts.server.recover_phone69_stride_sweep import boundary_metadata


CONFIG = ROOT / 'configs/server_phone69_460h_offline_native_stride4_none_dim264_s42_e50.yaml'
AUDIT = ROOT / 'output/phone69_460h_stride4_dim264_l1_match'
LOCK = Path('/tmp/phone69_460h_serial.lock')
TRAIN = ROOT / 'scripts/server/train_phone69_460h_one.py'


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    exp = Path(cfg['paths']['exp_dir'])
    if exp.exists():
        raise FileExistsError(exp)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    sources = [CONFIG, TRAIN, Path(__file__).resolve(), ROOT / cfg['tokenizer']['vocab_path']]
    sources += [ROOT / p for p in (
        'train.py', 'common.py', 'phone69_tokenizer.py', 'scripts/server/phone69_epoch_checkpoint.py',
    )]
    sources += sorted((ROOT / 'conformer_rnnt').rglob('*.py'))
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    budget = cfg['data']['max_token_limit']
    identity = dict(experiment='stride4_dim264', config=str(CONFIG.relative_to(ROOT)),
                    code_commit=commit, controller_pid=os.getpid(), stride=4, encoder_dim=264,
                    budget=budget, epochs=cfg['training']['epochs'])
    AUDIT.mkdir(parents=True, exist_ok=False)
    save(AUDIT / 'deployment.json', dict(**identity, source_sha256=hashes))

    def state(status, **details):
        record = dict(**identity, status=status, server_utc=now(), **details)
        save(AUDIT / 'queue_state.json', record)
        print(json.dumps(record), flush=True)

    try:
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
            startup_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            exp.mkdir(exist_ok=False)
            log_dir = exp / 'logs'
            log_dir.mkdir()
            train_log = log_dir / 'train.log'
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0',
                       OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                       RESUME_LAST='0')
            with train_log.open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', str(TRAIN), str(CONFIG)], cwd=ROOT,
                                         env=env, stdout=log, stderr=subprocess.STDOUT)
                state('training', training_pid=child.pid, training_commit=startup_commit,
                      train_log=str(train_log))
                if child.wait():
                    raise RuntimeError(f'Training exited with code {child.returncode}. See {train_log}')

            run = json.loads((exp / 'run.json').read_text())
            expected = ('trained', identity['epochs'], 264, 4, budget, hashes[identity['config']],
                        startup_commit)
            actual = tuple(run.get(key) for key in (
                'status', 'epoch', 'internal_dim', 'time_reduction_stride', 'max_token_limit',
                'config_sha256', 'code_commit',
            ))
            if actual != expected:
                raise RuntimeError('The finished training run differs from the recorded configuration')
            best = Path(run['best_checkpoint'])
            if not best.is_file() or (exp / 'best_checkpoint.txt').read_text().strip() != str(best):
                raise RuntimeError('Missing verified best validation checkpoint')
            boundary = Path(run['epoch_boundary_checkpoint'])
            metadata = boundary_metadata(boundary, budget)
            if (metadata['resume_completed_epochs'] != identity['epochs']
                    or metadata['resume_global_step'] != run['global_step']):
                raise RuntimeError('The final checkpoint does not contain the completed training boundary')
            completed = dict(training_commit=startup_commit, epoch=run['epoch'],
                             global_step=run['global_step'], best_checkpoint=str(best),
                             best_checkpoint_sha256=sha(best), epoch_boundary_checkpoint=str(boundary),
                             epoch_boundary_metadata=metadata)
            save(AUDIT / 'completion.json', completed)
            state('complete', **completed)
    except BaseException as error:
        state('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
