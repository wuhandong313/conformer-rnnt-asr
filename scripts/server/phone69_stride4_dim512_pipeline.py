#!/usr/bin/env python3
"""Run the 512-wide stride-4 no-compression baseline from scratch."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['PATH'] = '/root/anaconda3/envs/whd/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
AUDIT = ROOT / 'artifacts/runs/phone69_460h_stride4_dim512_20261103'
CONFIG = ROOT / 'configs/server_phone69_460h_offline_native_stride4_none_dim512_s42_e50.yaml'
EXP = ROOT / 'exp_phone69_460h_offline_native_stride4_none_dim512_s42_e50'
BUDGET = 7000
AUDIT.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(status, **fields):
    data = dict(status=status, server_utc=datetime.now(timezone.utc).isoformat(),
                pid=os.getpid(), **fields)
    tmp = AUDIT / 'queue_state.json.tmp'
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(AUDIT / 'queue_state.json')
    print(json.dumps(data), flush=True)


def run(args, name):
    with (AUDIT / name).open('x') as output:
        return subprocess.run([sys.executable, '-u', *map(str, args)], cwd=ROOT,
                              stdout=output, stderr=subprocess.STDOUT).returncode


def gpu_busy():
    return subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True).strip()


def wait_for_gpu_and_lock(lock):
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state('waiting', reason='training_lock_held')
            time.sleep(60)
            continue
        processes = gpu_busy()
        if not processes:
            return
        fcntl.flock(lock, fcntl.LOCK_UN)
        state('waiting', reason='gpu_busy', gpu_pids=processes.splitlines())
        time.sleep(60)


def main():
    with open('/tmp/phone69_460h_serial.lock', 'a') as lock:
        wait_for_gpu_and_lock(lock)
        assert not EXP.exists(), 'Experiment already exists; refusing to overwrite or restart'
        cfg = yaml.safe_load(CONFIG.read_text())
        model, comp = cfg['model'], cfg['model']['compression']
        assert cfg['paths']['exp_dir'] == str(EXP) and cfg['paths']['from_checkpoint'] is None
        assert (model['time_reduction_input_dim'], model['time_reduction_stride'],
                model['encoding_dim'], model['ffn_dim'], model['num_heads']) == (128, 4, 256, 512, 8)
        assert (model['num_layers_block1'], model['num_layers_block2']) == (2, 8)
        assert model['sequence_mode'] == 'offline' and 'right_context_length' not in model
        assert comp == {'filter': 'none'}
        assert cfg['data']['max_token_limit'] == BUDGET and cfg['training']['epochs'] == 50
        assert cfg['seed'] == 42 and cfg['trainer']['accumulate_grad_batches'] == 1
        state('preflight', stage='regression_tests')
        assert run(['-m', 'unittest', 'discover', '-s', 'tests'], 'tests.log') == 0, \
            'Regression tests failed'
        assert not gpu_busy(), 'GPU became occupied during preflight'
        state('preflight', stage='synthetic_rnnt')
        assert run(['scripts/server/validate_phone69_stride4_dim512.py', CONFIG],
                   'synthetic.log') == 0, 'Synthetic RNNT test failed'
        assert not gpu_busy(), 'GPU became occupied during preflight'
        state('preflight', stage='real_batch_memory', budget=BUDGET)
        report = AUDIT / 'memory_probe.json'
        assert run(['scripts/server/probe_phone69_training_memory.py', CONFIG,
                    '--out', report, '--keep-cache', '--max-reserved-fraction', '0.9'],
                   'memory_probe.log') == 0, 'Memory preflight failed'
        result = json.loads(report.read_text())
        assert result['status'] == 'passed' and result['config_sha256'] == sha(CONFIG)
        assert result['results'] and all(
            r['device_used_GiB'] <= result['gpu_total_GiB'] * 0.90
            for r in result['results']), 'Device memory exceeded 90% preflight threshold'
        assert all(sha(ROOT / p) == digest for p, digest in result['source_sha256'].items())
        sources = [*sorted((ROOT / 'conformer_rnnt').rglob('*.py')),
                   *[ROOT / p for p in [
                       'common.py', 'train.py', 'scripts/server/train_phone69_460h_one.py',
                       'scripts/server/phone69_epoch_checkpoint.py',
                       'scripts/server/probe_phone69_training_memory.py',
                       'scripts/server/validate_phone69_stride4_dim512.py',
                       'scripts/server/phone69_stride4_dim512_pipeline.py']]]
        gate = dict(valid=True, config=str(CONFIG), config_sha256=sha(CONFIG),
                    sources={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    probe=str(report), budget=BUDGET,
                    server_utc=datetime.now(timezone.utc).isoformat())
        (AUDIT / 'prelaunch_validation.json').write_text(json.dumps(gate, indent=2) + '\n')
        assert all(sha(ROOT / p) == h for p, h in gate['sources'].items())
        assert sha(CONFIG) == gate['config_sha256']
        assert not gpu_busy() and not EXP.exists()
        (EXP / 'logs').mkdir(parents=True)
        with (EXP / 'logs/train.log').open('x') as log, \
                (EXP / 'logs/gpu_usage.csv').open('x') as gpu:
            child = subprocess.Popen(
                [sys.executable, '-u', 'scripts/server/train_phone69_460h_one.py', str(CONFIG)],
                cwd=ROOT, env=dict(os.environ, RESUME_LAST='0'),
                stdout=log, stderr=subprocess.STDOUT)
            state('training', training_pid=child.pid, experiment=str(EXP), budget=BUDGET)
            while child.poll() is None:
                subprocess.run(
                    ['nvidia-smi', '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu',
                     '--format=csv,noheader', '-i', '0'],
                    stdout=gpu, stderr=subprocess.DEVNULL)
                gpu.flush()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
        assert child.returncode == 0, 'Training exited unsuccessfully; logs preserved'
        import torch
        from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
        runstate = json.loads((EXP / 'run.json').read_text())
        assert runstate['status'] == 'trained' and runstate['epoch'] == 50
        definition = LibriSpeechRNNTModule.from_config(cfg).offline_native_definition
        for key in ['best_checkpoint', 'epoch_boundary_checkpoint']:
            ck = torch.load(runstate[key], map_location='cpu', weights_only=False)
            assert ck.get('state_dict') and ck.get('optimizer_states') and ck.get('lr_schedulers')
            assert ck.get('offline_native_definition') == definition
            if key == 'epoch_boundary_checkpoint':
                cur = ck['loops']['fit_loop']['epoch_progress']['current']
                assert cur['ready'] == cur['completed']
                assert ck['phone69_epoch_boundary']['resume_completed_epochs'] == 50
        (EXP / 'TRAINING_COMPLETE').touch(exist_ok=False)
        state('complete', experiment=str(EXP), global_step=runstate['global_step'],
              best_checkpoint=runstate['best_checkpoint'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state('failed', error=repr(error),
              message='Logs and checkpoints preserved; no automatic restart')
        raise
