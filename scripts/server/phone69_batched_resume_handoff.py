#!/usr/bin/env python3
"""Switch the running nine-frame experiment at a complete epoch boundary.

The existing trainer remains untouched until a fully checked boundary is
available. The new runner appends to the existing log and keeps the old model,
optimizer, scheduler, budget and epoch count. It does not rewrite checkpoints.
"""
from __future__ import annotations
import fcntl, hashlib, json, os, re, shutil, signal, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/test21/conformer-rnnt-asr')
EXP = ROOT / 'exp_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_s42_e50'
AUDIT = ROOT / 'artifacts/runs/phone69_460h_nine_frames_20261029'
WORK = AUDIT / 'batched_handoff_20261029'
QUEUE = AUDIT / 'queue_state.json'
CONFIG = ROOT / 'configs/server_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_batched_resume_s42_e50.yaml'
ORIGINAL = ROOT / 'configs/server_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_s42_e50.yaml'
TRAINER = ROOT / 'scripts/server/train_phone69_460h_one.py'
LOCK = '/tmp/phone69_460h_serial.lock'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def state(status, **kw):
    data = {'status': status, 'server_utc': now(), 'pid': os.getpid(), **kw}
    save(WORK / 'state.json', data)
    print(json.dumps(data), flush=True)


def proc_state(pid):
    path = Path('/proc') / str(pid) / 'status'
    return path.read_text().split('State:', 1)[1].strip()[0] if path.exists() else None


def children(pid):
    p = Path('/proc') / str(pid) / 'task' / str(pid) / 'children'
    return [int(x) for x in p.read_text().split()] if p.exists() else []


def queue(status, **kw):
    save(QUEUE, {'status': status, 'server_utc': now(), 'pid': os.getpid(),
                 'experiment': str(EXP), **kw})


def progress_after_boundary(completed):
    p = EXP / 'logs/train.log'
    with p.open('rb') as f:
        f.seek(max(0, p.stat().st_size - 160000))
        text = f.read().decode('utf-8', 'replace')
    matches = re.findall(r'Epoch\s+(\d+):[^\r\n]*?(\d+)/2531', text)
    return any(int(epoch) >= completed and int(batch) > 0 for epoch, batch in matches)


def source_fingerprints():
    paths = sorted((ROOT / 'conformer_rnnt').rglob('*.py')) + [
        ROOT / 'train.py', TRAINER, ROOT / 'scripts/server/phone69_epoch_checkpoint.py',
        CONFIG, ORIGINAL,
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    import torch, yaml
    sys.path.insert(0, str(ROOT))
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from conformer_rnnt.checkpoint_utils import phone69_resume_metadata

    os.chdir(ROOT)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == EXPECTED_HEAD
    fingerprints = source_fingerprints()
    config = yaml.safe_load(CONFIG.read_text())
    original = yaml.safe_load(ORIGINAL.read_text())
    check = yaml.safe_load(CONFIG.read_text())
    assert check['model']['compression'].pop('execution_backend') == 'batched' and check == original
    model = LibriSpeechRNNTModule.from_config(config)
    assert model.model.transcriber.compression.execution_backend == 'batched'
    initial = json.loads((EXP / 'run.json').read_text())
    q = json.loads(QUEUE.read_text())
    trainer_pid, controller_pid = int(q['training_pid']), int(q['pid'])
    assert q['status'] == 'training' and initial['status'] == 'training' and initial['pid'] == trainer_pid
    original_checkpoint = Path(initial['epoch_boundary_checkpoint'])
    checkpoint = torch.load(original_checkpoint, map_location='cpu', weights_only=False)
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    initial_metadata = phone69_resume_metadata(checkpoint, config['data']['max_token_limit'], model.offline_native_definition)
    assert initial_metadata['resume_at_epoch_boundary']
    completed = initial_metadata['resume_completed_epochs']
    del checkpoint, model
    state('waiting_for_epoch_boundary', training_pid=trainer_pid, after_epoch=completed)
    frozen = False
    while True:
        assert proc_state(trainer_pid) is not None and proc_state(controller_pid) is not None
        options = []
        for path in (EXP / 'checkpoints').glob('epoch-boundary-epoch=*-step=*.ckpt'):
            m = re.match(r'epoch-boundary-epoch=(\d+)-step=(\d+)-.*\.ckpt$', path.name)
            if m and int(m.group(1)) > completed:
                options.append((int(m.group(1)), int(m.group(2)), path))
        if not options:
            time.sleep(0.03)
            continue
        epoch, step, path = min(options)
        os.kill(trainer_pid, signal.SIGSTOP)
        frozen = True
        try:
            for _ in range(100):
                if proc_state(trainer_pid) == 'T':
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError('Trainer did not pause')
            assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == EXPECTED_HEAD
            assert source_fingerprints() == fingerprints, 'Training source changed while waiting'
            if progress_after_boundary(epoch):
                completed = epoch
                os.kill(trainer_pid, signal.SIGCONT)
                frozen = False
                state('waiting_for_next_boundary', training_pid=trainer_pid, after_epoch=epoch)
                continue
            model = LibriSpeechRNNTModule.from_config(config)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model.on_load_checkpoint(checkpoint)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            meta = phone69_resume_metadata(checkpoint, config['data']['max_token_limit'], model.offline_native_definition)
            assert meta['resume_at_epoch_boundary'] and meta['resume_completed_epochs'] == epoch
            assert meta['resume_global_step'] == step and checkpoint.get('state_dict')
            assert checkpoint.get('optimizer_states') and checkpoint.get('lr_schedulers')
            assert not progress_after_boundary(epoch), 'Trainer advanced past boundary'
            del checkpoint, model
            state('boundary_validated', training_pid=trainer_pid, checkpoint=str(path), epoch=epoch, step=step)
            shutil.copy2(EXP / 'run.json', WORK / 'run_before_handoff.json')
            shutil.copy2(QUEUE, WORK / 'queue_before_handoff.json')
            worker_pids = children(trainer_pid)
            os.kill(controller_pid, signal.SIGSTOP)
            os.kill(trainer_pid, signal.SIGKILL)
            frozen = False
            os.kill(controller_pid, signal.SIGKILL)
            for child in worker_pids:
                try: os.kill(child, signal.SIGTERM)
                except ProcessLookupError: pass
            break
        except BaseException:
            if frozen and proc_state(trainer_pid) == 'T':
                os.kill(trainer_pid, signal.SIGCONT)
            raise

    with open(LOCK, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(120):
            p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            if not p.stdout.strip(): break
            time.sleep(0.5)
        else: raise RuntimeError('GPU did not become idle; no duplicate trainer started')

        runtime = yaml.safe_load(CONFIG.read_text())
        runtime['paths']['from_checkpoint'] = str(path)
        runtime_path = WORK / 'batched_runtime.yaml'
        runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False))
        state('memory_preflight', checkpoint=str(path), epoch=epoch, step=step)
        # Preflight exercises worst actual batches in a disposable model, never
        # updating the checkpoint. Failure falls back to the original backend.
        with (WORK / 'memory_preflight.log').open('x') as log:
            try:
                p = subprocess.run([sys.executable, '-u', 'scripts/server/probe_phone69_training_memory.py', str(runtime_path), '--out', str(WORK / 'memory_preflight.json'), '--keep-cache', '--max-reserved-fraction', '0.90'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                preflight_exit = p.returncode
            except subprocess.TimeoutExpired:
                preflight_exit = 124
        backend = 'batched' if preflight_exit == 0 else 'reference'
        if backend == 'reference':
            fallback = yaml.safe_load(ORIGINAL.read_text())
            fallback['paths']['from_checkpoint'] = str(path)
            runtime_path = WORK / 'reference_fallback_runtime.yaml'
            runtime_path.write_text(yaml.safe_dump(fallback, sort_keys=False))
        state('starting', backend=backend, checkpoint=str(path), epoch=epoch, step=step, preflight_exit=preflight_exit)
        assert source_fingerprints() == fingerprints, 'Training source changed during preflight'
        with (EXP / 'logs/train.log').open('a') as log, (EXP / 'logs/gpu_usage.csv').open('a') as gpu:
            log.write('\n--- '+now()+' resume backend='+backend+' checkpoint='+str(path)+' ---\n');log.flush()
            child = subprocess.Popen([sys.executable, '-u', str(TRAINER), str(runtime_path)], cwd=ROOT, env=dict(os.environ, RESUME_LAST='0'), stdout=log, stderr=subprocess.STDOUT)
            queue('training', training_pid=child.pid, backend=backend, resume_checkpoint=str(path), resume_global_step=step, budget=config['data']['max_token_limit'])
            state('training', backend=backend, training_pid=child.pid, checkpoint=str(path), epoch=epoch, step=step, preflight_exit=preflight_exit)
            while child.poll() is None:
                subprocess.run(['nvidia-smi', '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader', '-i', '0'], stdout=gpu, stderr=subprocess.DEVNULL)
                gpu.flush()
                try: child.wait(timeout=60)
                except subprocess.TimeoutExpired: pass
        if child.returncode:
            queue('failed', training_pid=child.pid, backend=backend, exit_code=child.returncode)
            raise RuntimeError('Resumed trainer failed, see train.log')
        run = json.loads((EXP / 'run.json').read_text())
        assert run['status'] == 'trained' and run['epoch'] == 50
        model = LibriSpeechRNNTModule.from_config(runtime if backend == 'batched' else fallback)
        for key in ('best_checkpoint', 'epoch_boundary_checkpoint'):
            ck = torch.load(run[key], map_location='cpu', weights_only=False)
            model.on_load_checkpoint(ck)
            assert ck.get('state_dict') and ck.get('optimizer_states') and ck.get('lr_schedulers')
        (EXP / 'TRAINING_COMPLETE').touch(exist_ok=False)
        queue('complete', training_pid=child.pid, backend=backend, global_step=run['global_step'])
        state('complete', backend=backend, global_step=run['global_step'])


EXPECTED_HEAD = os.environ['PHONE69_EXPECTED_HEAD']
if __name__ == '__main__':
    WORK.mkdir(parents=True, exist_ok=False)
    try:
        main()
    except BaseException as error:
        (WORK / 'error.log').write_text(traceback.format_exc())
        state('failed', error=repr(error))
        raise
