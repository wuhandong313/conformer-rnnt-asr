#!/usr/bin/env python3
"""Launch or inspect the fixed Mel/layer5/layer8 experiment; never run tests."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = 'configs/server_phone69_460h_placement_mel_layer5_layer8_stride1_s42_b11050_e50.yaml'
BASE_CONFIG = 'configs/server_phone69_460h_placement_mel_layer5_stride1_s42_resume_b11050_e50.yaml'
EXPERIMENT = 'exp_phone69_460h_offline_native_placement_mel_layer5_layer8_stride1_s42_b11050_e50'
LOCK = Path('/tmp/phone69_460h_serial.lock')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_config():
    actual = yaml.safe_load((REPO / CONFIG).read_text())
    expected = yaml.safe_load((REPO / BASE_CONFIG).read_text())
    expected['paths']['exp_dir'] = str(REPO / EXPERIMENT)
    expected['paths']['from_checkpoint'] = None
    third = deepcopy(expected['model']['compression']['stages'][-1])
    third['after_layer'] = 8
    expected['model']['compression']['stages'].append(third)
    if actual != expected:
        raise RuntimeError('Triple-stage configuration differs from the recorded layout/budget')
    return actual


def output(arguments):
    return subprocess.check_output(arguments, cwd=REPO, text=True).strip()


def launch(expected_commit, expected_config_sha):
    config = REPO / CONFIG
    experiment = REPO / EXPERIMENT
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        head = output(['git', 'rev-parse', 'HEAD'])
        branch = output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        dirty = output(['git', 'status', '--porcelain=v1', '--untracked-files=all'])
        if head != expected_commit or branch != 'main' or dirty:
            raise RuntimeError('Server Git must match the expected clean main commit')
        if sha(config) != expected_config_sha:
            raise RuntimeError('Configuration SHA256 mismatch')
        check_config()
        if experiment.exists():
            raise RuntimeError('Experiment already exists; refusing duplicate launch or restart')
        apps = output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
        memory = output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
        if apps or not memory or any(int(x.strip()) > 2048 for x in memory.splitlines()):
            raise RuntimeError('GPU is occupied; no other process will be interrupted')
        experiment.mkdir(exist_ok=False)
        logs = experiment / 'logs'
        logs.mkdir()
        env = os.environ.copy()
        env.update(RESUME_LAST='0', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
        with (logs / 'train.log').open('xb', buffering=0) as log:
            child = subprocess.Popen(
                [sys.executable, '-u', 'scripts/server/train_phone69_460h_one.py', str(config)],
                cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),),
            )
        record = {
            'pid': child.pid, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'config': str(config), 'config_sha256': expected_config_sha, 'git_commit': head,
            'experiment': str(experiment), 'train_log': str(logs / 'train.log'),
            'resume_last': False, 'resume_checkpoint': None, 'max_token_limit': 11050,
            'stages': [0, 5, 8], 'lock': str(LOCK), 'tests': 'local only; no server memory probe',
        }
        temporary = experiment / 'launch.json.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(experiment / 'launch.json')
    return record


def status():
    experiment = REPO / EXPERIMENT
    config = REPO / CONFIG
    record = read_json(experiment / 'launch.json')
    run = read_json(experiment / 'run.json')
    pid = (record or {}).get('pid')
    process = subprocess.run(['ps', '-p', str(pid or 0), '-o', 'pid=,stat=,args='],
                             capture_output=True, text=True).stdout.strip()
    alive = bool(process and 'train_phone69_460h_one.py' in process and str(config) in process)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True)
    log = experiment / 'logs/train.log'
    tail = []
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 18000))
            tail = stream.read().decode('utf-8', 'replace').replace('\r', '\n').splitlines()[-18:]
    checkpoints = sorted((experiment / 'checkpoints').glob('*.ckpt'),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    keys = ('status', 'epoch', 'epochs', 'global_step', 'pid', 'error', 'updated_at',
            'config_sha256', 'code_commit', 'max_token_limit', 'internal_dim', 'resume_checkpoint',
            'best_checkpoint', 'epoch_boundary_checkpoint', 'epoch_boundary_metadata',
            'offline_native_definition', 'observed_compression_lengths')
    current_sha = sha(config) if config.exists() else None
    integrity_ok = bool(record and current_sha == record.get('config_sha256'))
    if run:
        integrity_ok = integrity_ok and run.get('config_sha256') == current_sha
        integrity_ok = integrity_ok and run.get('code_commit') == record.get('git_commit')
        integrity_ok = integrity_ok and run.get('max_token_limit') == 11050
        integrity_ok = integrity_ok and run.get('resume_checkpoint') is None
        definition = run.get('offline_native_definition') or {}
        integrity_ok = integrity_ok and definition.get('layers') == [5, 3, 2]
        integrity_ok = integrity_ok and definition.get('stride') == 1
        integrity_ok = integrity_ok and definition.get('input_projection') == 128
        integrity_ok = integrity_ok and [s.get('after_layer') for s in definition.get('compression_stages', [])] == [0, 5, 8]
    best_exists = bool(run and run.get('best_checkpoint') and Path(run['best_checkpoint']).is_file())
    completed = bool(integrity_ok and run and run.get('status') == 'trained' and run.get('epoch') == 50
                     and best_exists and (run.get('epoch_boundary_metadata') or {}).get('resume_completed_epochs') == 50
                     and Path(run.get('epoch_boundary_checkpoint', '')).is_file())
    return {
        'server_utc': datetime.now(timezone.utc).isoformat(), 'launch': record,
        'run': {key: run.get(key) for key in keys} if run else None,
        'process_alive': alive, 'process': process, 'gpu': gpu.stdout.strip(),
        'config_sha256': current_sha, 'integrity_ok': integrity_ok,
        'git_head': output(['git', 'rev-parse', 'HEAD']),
        'git_status': output(['git', 'status', '--porcelain=v1', '--untracked-files=all']),
        'log_bytes': log.stat().st_size if log.exists() else 0, 'log_tail': tail,
        'checkpoint_count': len(checkpoints),
        'recent_checkpoints': [{'name': p.name, 'bytes': p.stat().st_size} for p in checkpoints[:5]],
        'best_exists': best_exists, 'complete': completed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'status'))
    parser.add_argument('--expected-commit')
    parser.add_argument('--expected-config-sha')
    args = parser.parse_args()
    if args.action == 'launch':
        if not args.expected_commit or not args.expected_config_sha:
            parser.error('launch requires --expected-commit and --expected-config-sha')
        result = launch(args.expected_commit, args.expected_config_sha)
    else:
        result = status()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
