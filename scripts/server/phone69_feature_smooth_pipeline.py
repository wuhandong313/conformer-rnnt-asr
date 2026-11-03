"""Preflight, train and monitor the approved fresh feature-filter experiment."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'artifacts/runs/phone69_460h_feature_smooth_20261102'
CONFIG = ROOT / 'configs/server_phone69_460h_offline_native_stride1_cosd_learned_feature_smooth_a025_s42_e50.yaml'
EXP = ROOT / 'exp_phone69_460h_offline_native_stride1_cosd_learned_feature_smooth_a025_s42_e50'
os.chdir(ROOT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def state(status, **updates):
    previous = json.loads((AUDIT / 'queue_state.json').read_text()) if (AUDIT / 'queue_state.json').exists() else {}
    previous.update(status=status, pipeline_pid=os.getpid(), server_utc=datetime.now(timezone.utc).isoformat(), **updates)
    save(AUDIT / 'queue_state.json', previous)
    print(json.dumps(previous, ensure_ascii=False), flush=True)


def idle():
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).strip()
    if processes:
        raise RuntimeError('GPU is occupied; will not preempt existing work: ' + processes)


def protected_unchanged():
    records = json.loads((AUDIT / 'protected_experiments.json').read_text())
    for relative, record in records.items():
        path = ROOT / relative
        stat = path.stat()
        assert stat.st_size == record['bytes'] and stat.st_mtime_ns == record['mtime_ns'], relative
        if 'sha256' in record:
            assert sha(path) == record['sha256'], relative
    return len(records)


def run(arguments, log_name):
    with (AUDIT / log_name).open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', *map(str, arguments)], cwd=ROOT,
            env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'),
            stdout=log, stderr=subprocess.STDOUT)
        state('preflight', preflight_pid=process.pid, preflight_log=log_name)
        code = process.wait()
        if code:
            raise RuntimeError(f'{log_name} failed with exit status {code}')


def main():
    import yaml

    AUDIT.mkdir(parents=True, exist_ok=True)
    inherited = os.environ.pop('PHONE69_LAUNCH_LOCK_FD', None)
    lock = os.fdopen(int(inherited), 'a') if inherited else open('/tmp/phone69_460h_serial.lock', 'a')
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not EXP.exists(), 'Experiment exists; refusing to overwrite or restart'
        manifest = json.loads((AUDIT / 'deployment.json').read_text())
        assert all(sha(ROOT / p) == h for p, h in manifest['deployed_sha256'].items())
        config = yaml.safe_load(CONFIG.read_text())
        model, compression = config['model'], config['model']['compression']
        assert config['paths']['from_checkpoint'] is None and config['paths']['exp_dir'] == str(EXP)
        assert model['sequence_mode'] == 'offline' and model['time_reduction_stride'] == 1
        assert model['time_reduction_input_dim'] == 128 and model['num_heads'] == 8
        assert (model['num_layers_block1'], model['num_layers_block2']) == (2, 8)
        assert compression['detector'] == {'type': 'cosine_diff', 'params': {'feature_smoothing_alpha': 0.25}}
        assert compression['weight_predictor_mode'] == 'five_frames'
        assert compression.get('execution_backend', 'reference') == 'reference'
        assert 'keep_rate' not in compression and 'right_context_length' not in model
        assert config['data']['max_token_limit'] == 6500 and config['training']['epochs'] == 50
        assert config['seed'] == 42 and config['trainer']['accumulate_grad_batches'] == 1
        idle()
        state('preflight', stage='regression_tests')
        run(['-m', 'unittest', 'discover', '-s', 'tests'], 'tests.log')
        idle()
        state('preflight', stage='synthetic_cuda_rnnt')
        run(['scripts/server/validate_phone69_feature_smoothing.py', CONFIG], 'synthetic.log')
        idle()
        state('preflight', stage='real_batch_memory', budget=6500)
        report_path = AUDIT / 'memory_probe.json'
        run(['scripts/server/probe_phone69_training_memory.py', CONFIG, '--out', report_path,
             '--keep-cache', '--max-reserved-fraction', '0.9'], 'memory_probe.log')
        report = json.loads(report_path.read_text())
        assert report['status'] == 'passed' and report['budget'] == 6500
        assert report['config_sha256'] == sha(CONFIG)
        assert all(sha(ROOT / p) == h for p, h in report['source_sha256'].items())
        sources = [*sorted((ROOT / 'conformer_rnnt').rglob('*.py')),
            *[ROOT / p for p in ['common.py', 'train.py', 'phone69_tokenizer.py',
                'scripts/server/train_phone69_460h_one.py', 'scripts/server/phone69_epoch_checkpoint.py',
                'scripts/server/probe_phone69_training_memory.py',
                'scripts/server/phone69_feature_smooth_pipeline.py',
                'scripts/server/validate_phone69_feature_smoothing.py']]]
        gate = {'valid': True, 'server_utc': datetime.now(timezone.utc).isoformat(),
            'config': str(CONFIG), 'config_sha256': sha(CONFIG), 'memory_probe': str(report_path),
            'sources': {str(p.relative_to(ROOT)): sha(p) for p in sources},
            'protected_files_verified': protected_unchanged()}
        save(AUDIT / 'prelaunch_validation.json', gate)
        for p in sources:
            destination = AUDIT / 'source_snapshot' / p.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(p.read_bytes())
        assert all(sha(ROOT / p) == h for p, h in gate['sources'].items())
        assert sha(CONFIG) == gate['config_sha256']
        idle()
        assert not EXP.exists()
        (EXP / 'logs').mkdir(parents=True)
        with (EXP / 'logs/train.log').open('x') as log, (EXP / 'logs/gpu_usage.csv').open('x') as gpu:
            child = subprocess.Popen([sys.executable, '-u', 'scripts/server/train_phone69_460h_one.py',
                str(CONFIG)], cwd=ROOT, env=dict(os.environ, RESUME_LAST='0',
                OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), stdout=log, stderr=subprocess.STDOUT)
            state('training', stage='training', training_pid=child.pid, experiment=str(EXP),
                config_sha256=gate['config_sha256'], budget=6500)
            while child.poll() is None:
                subprocess.run(['nvidia-smi', '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader', '-i', '0'], stdout=gpu, stderr=subprocess.DEVNULL)
                gpu.flush()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
        assert child.returncode == 0, 'Training failed; logs and checkpoints preserved'
        state('validating', stage='completion_validation')
        import torch
        sys.path.insert(0, str(ROOT))
        from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
        run_state = json.loads((EXP / 'run.json').read_text())
        assert run_state['status'] == 'trained' and run_state['epoch'] == 50
        definition = LibriSpeechRNNTModule.from_config(config).offline_native_definition
        validation = {'valid': True, 'server_utc': datetime.now(timezone.utc).isoformat(),
            'experiment': str(EXP), 'config_sha256': sha(CONFIG), 'checkpoints': {}}
        for key in ['best_checkpoint', 'epoch_boundary_checkpoint']:
            path = Path(run_state[key])
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            assert checkpoint.get('state_dict') and checkpoint.get('optimizer_states') and checkpoint.get('lr_schedulers')
            assert checkpoint['offline_native_definition'] == definition
            assert all(torch.isfinite(value).all() for value in checkpoint['state_dict'].values()
                       if isinstance(value, torch.Tensor) and value.is_floating_point())
            if key == 'epoch_boundary_checkpoint':
                progress = checkpoint['loops']['fit_loop']['epoch_progress']['current']
                assert progress['ready'] == progress['completed'] == 50
                assert checkpoint['phone69_epoch_boundary']['resume_completed_epochs'] == 50
            validation['checkpoints'][key] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
            del checkpoint
        validation['protected_files_verified'] = protected_unchanged()
        save(AUDIT / 'completion_validation.json', validation)
        (EXP / 'TRAINING_COMPLETE').touch(exist_ok=False)
        state('complete', stage='complete', global_step=run_state['global_step'],
            best_checkpoint=run_state['best_checkpoint'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state('failed', error=repr(error), message='Logs and checkpoints preserved; no automatic restart')
        raise
