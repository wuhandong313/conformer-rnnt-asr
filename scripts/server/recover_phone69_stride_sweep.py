#!/usr/bin/env python3
"""Restore the recorded runtime and resume the authorized fixed-stride queue.

This entry point deliberately uses only the standard library until runtime
installation finishes. It never calibrates a batch, runs a software test, or
deletes an experiment. Completed groups are verified and skipped.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'output/phone69_460h_stride_sweep_proj128_20261108'
OUT = ROOT / 'output/phone69_460h_stride_sweep_recovery_20261111'
PYTHON = Path('/root/anaconda3/envs/whd/bin/python')
REQUIREMENTS = ROOT / 'artifacts/recoveries/phone69_post_restart_20261021/requirements-restored.txt'
STRIDES = (1, 2, 8)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), universal_newlines=True).strip()


def source_identity():
    deployment = read(ORIGINAL / 'deployment.json')
    if not deployment or not deployment.get('source_sha256'):
        raise ValueError('Original deployment fingerprints are missing')
    changed = [name for name, digest in deployment['source_sha256'].items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    if changed:
        raise ValueError('Original training source/configuration changed: ' + ', '.join(changed))
    return dict(original_controller_commit=deployment['code_commit'],
                recovery_commit=commit(), source_sha256=deployment['source_sha256'],
                original_source_files_unchanged=True)


def job_kind(arguments):
    names = {Path(value).name for value in arguments if value}
    if 'train_phone69_460h_one.py' in names:
        return 'training'
    if 'phone69_stride_sweep.py' in names:
        return 'original_controller'
    if Path(__file__).name in names and '--mode' in arguments:
        mode = arguments[arguments.index('--mode') + 1:][:1]
        if mode == ['run']:
            return 'recovery_controller'
        if mode == ['restore']:
            return 'runtime_installer'
    return None


def live_jobs():
    result = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(path.name)
            if pid == os.getpid() or (path / 'stat').read_text().split()[2] == 'Z':
                continue
            arguments = (path / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            kind = job_kind(arguments)
            if kind:
                result.append(dict(pid=pid, kind=kind))
        except (OSError, ValueError, IndexError):
            continue
    return result


def gpu_info():
    result = subprocess.run(['nvidia-smi', '-i', '0',
                             '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu',
                             '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    return result.stdout.strip() if result.returncode == 0 else None


def experiment(stride):
    return ROOT / ('exp_phone69_460h_offline_native_stride%d_none_proj128_s42_e50' % stride)


def boundary_key(path):
    found = re.fullmatch(r'epoch-boundary-epoch=(\d+)-step=(\d+)-[a-f0-9]+\.ckpt', Path(path).name)
    return tuple(map(int, found.groups())) if found else (-1, -1)


def selected(stride):
    gate = read(ORIGINAL / ('stride%d_proj128_selected.json' % stride))
    if not gate or not gate.get('valid'):
        raise ValueError('Missing verified batch selection for stride%d' % stride)
    config = ROOT / gate['effective_config']
    if sha(config) != gate['effective_config_sha256']:
        raise ValueError('Effective recipe fingerprint changed for stride%d' % stride)
    return gate, config


def inspect():
    identity = source_identity()
    groups = []
    for stride in STRIDES:
        gate, config = selected(stride)
        exp = experiment(stride)
        run = read(exp / 'run.json', {})
        boundaries = sorted((exp / 'checkpoints').glob('epoch-boundary-*.ckpt'), key=boundary_key)
        best = run.get('best_checkpoint')
        groups.append(dict(stride=stride, budget=gate['budget'], effective_config=str(config),
                           run=run, best_exists=bool(best and Path(best).is_file()),
                           latest_boundaries=[dict(path=str(p), bytes=p.stat().st_size)
                                              for p in boundaries[-3:]]))
    return dict(server_utc=now(), identity=identity, groups=groups, live_jobs=live_jobs(),
                gpu=gpu_info(), runtime_python_exists=PYTHON.is_file(),
                conda_exists=Path('/root/anaconda3/bin/conda').is_file(),
                persistent_free_bytes=shutil.disk_usage(ROOT).free)


def update(**fields):
    state = read(OUT / 'recovery_state.json', {})
    state.update(fields, updated_at_server_utc=now(), controller_pid=os.getpid())
    save(OUT / 'recovery_state.json', state)
    return state


def checked(command, log, env=None, timeout=7200):
    with (OUT / log).open('a') as stream:
        stream.write(now() + '\n')
        stream.flush()
        result = subprocess.run(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
                                env=env, timeout=timeout)
    if result.returncode:
        raise RuntimeError('%s exited %d; preserved log in %s' % (command[0], result.returncode, log))


def restore():
    source_identity()
    if live_jobs():
        raise RuntimeError('Training/controller process is already alive; refusing environment replacement')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        update(status='installing_runtime', identity=source_identity(), requirements_sha256=sha(REQUIREMENTS))
        if not PYTHON.is_file():
            checked(['/root/anaconda3/bin/conda', 'create', '-n', 'whd', 'python=3.10.18', '-y',
                     '--override-channels', '-c', 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main'],
                    'conda_create.log', timeout=1800)
        env = dict(os.environ, PYTHONNOUSERSITE='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                   PIP_CACHE_DIR='/test21/.cache/phone69_runtime_pip')
        checked([str(PYTHON), '-m', 'pip', 'install', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple',
                 '--progress-bar', 'off', '--timeout', '60', '--retries', '3', '-r', str(REQUIREMENTS)],
                'pip_install.log', env=env)
        checked([str(PYTHON), '-m', 'pip', 'check'], 'pip_check.log', env=env, timeout=120)
        checked([str(PYTHON), '-m', 'pip', 'freeze'], 'pip_freeze.txt', env=env, timeout=120)
        query = ('import json,sys,torch,torchaudio,pytorch_lightning,numpy;'
                 'print(json.dumps(dict(python=sys.version,torch=torch.__version__,'
                 'torchaudio=torchaudio.__version__,lightning=pytorch_lightning.__version__,'
                 'numpy=numpy.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available())))')
        runtime = json.loads(subprocess.check_output([str(PYTHON), '-c', query], cwd=str(ROOT), env=env,
                                                   universal_newlines=True))
        if runtime['torch'].split('+')[0] != '2.6.0' or not runtime['cuda_available']:
            raise RuntimeError('The recorded CUDA training runtime is unavailable')
        save(OUT / 'runtime.json', runtime)
        update(status='runtime_ready', runtime=runtime)


def boundary_metadata(path, budget):
    import torch
    from conformer_rnnt.checkpoint_utils import phone69_resume_metadata
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    metadata = phone69_resume_metadata(checkpoint, budget)
    if not metadata['resume_at_epoch_boundary'] or metadata['previous_max_token_limit'] != budget:
        raise ValueError('Checkpoint is not a complete epoch at the original batch budget')
    if not checkpoint.get('state_dict'):
        raise ValueError('Checkpoint model state is missing')
    del checkpoint
    return metadata


def latest_boundary(stride, budget):
    errors = []
    for path in sorted((experiment(stride) / 'checkpoints').glob('epoch-boundary-*.ckpt'),
                       key=boundary_key, reverse=True):
        try:
            return path, boundary_metadata(path, budget)
        except Exception as error:
            errors.append(dict(path=str(path), error=str(error)))
            save(OUT / ('stride%d_preserved_checkpoint_errors.json' % stride), errors)
    raise ValueError('No complete, restorable boundary for stride%d' % stride)


def preserve_checkpoints(stride):
    """Keep existing bytes even if Lightning later rotates a top-k pathname."""
    destination = OUT / 'original_checkpoints' / ('stride%d' % stride)
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    for source in (experiment(stride) / 'checkpoints').glob('*.ckpt'):
        target = destination / source.name
        if not target.exists():
            if source.name.startswith('last'):
                shutil.copy2(str(source), str(target))
            else:
                try:
                    os.link(str(source), str(target))
                except OSError:
                    shutil.copy2(str(source), str(target))
        if source.stat().st_size != target.stat().st_size:
            raise ValueError('Preserved checkpoint size mismatch')
        inventory.append(dict(original=str(source), preserved=str(target), bytes=target.stat().st_size))
    save(destination / 'inventory.json', inventory)


def protect_mutable_archives():
    protected = []
    for path in (OUT / 'original_checkpoints').glob('stride*/last*.ckpt'):
        before = path.stat()
        if before.st_nlink > 1:
            temporary = path.with_suffix('.independent.tmp')
            shutil.copy2(str(path), str(temporary))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Archived checkpoint changed during preservation')
            temporary.replace(path)
        protected.append(dict(path=str(path), bytes=path.stat().st_size,
                              independent_inode=path.stat().st_nlink == 1))
    save(OUT / 'mutable_archive_protection.json', protected)
    return protected


def finalize_boundary(stride, boundary, metadata):
    """Recover the completion marker if the restart followed the last epoch."""
    import torch
    checkpoint = torch.load(boundary, map_location='cpu', weights_only=False)
    candidates = []
    for key, callback in checkpoint.get('callbacks', {}).items():
        if 'val_loss' in key and callback.get('best_model_path'):
            path = Path(callback['best_model_path'])
            score = callback.get('best_model_score')
            if path.is_file() and score is not None:
                candidates.append((float(score), str(path)))
    if not candidates or metadata['resume_completed_epochs'] != 50:
        raise ValueError('The final boundary does not preserve a best validation checkpoint')
    best = min(candidates)[1]
    del checkpoint
    run_path = experiment(stride) / 'run.json'
    run = read(run_path)
    run.update(status='trained', epoch=50, global_step=metadata['resume_global_step'],
               best_checkpoint=best, epoch_boundary_checkpoint=str(boundary),
               epoch_boundary_metadata=metadata, updated_at=now(),
               recovered_completion_after_restart=True)
    marker = experiment(stride) / 'best_checkpoint.txt'
    if marker.exists() and marker.read_text().strip() != best:
        raise ValueError('Existing best-checkpoint marker disagrees with final boundary')
    if not marker.exists():
        marker.write_text(best + '\n')
    save(run_path, run)


def completed(stride, gate, allowed_config_sha):
    run = read(experiment(stride) / 'run.json', {})
    if run.get('status') != 'trained':
        return None
    if (run.get('epoch'), run.get('max_token_limit'), run.get('internal_dim')) != (50, gate['budget'], 128 * stride):
        raise ValueError('Completed run metadata does not match stride%d' % stride)
    if run.get('config_sha256') not in allowed_config_sha:
        raise ValueError('Completed run configuration fingerprint is unexpected')
    path, metadata = latest_boundary(stride, gate['budget'])
    best = Path(run['best_checkpoint'])
    marker = (experiment(stride) / 'best_checkpoint.txt').read_text().strip()
    if metadata['resume_completed_epochs'] != 50 or not best.is_file() or marker != str(best):
        raise ValueError('Final epoch boundary or best checkpoint is incomplete')
    return dict(name='stride%d_proj128' % stride, stride=stride, budget=gate['budget'],
                status='trained', epoch=50, global_step=run['global_step'],
                training_commit=run['code_commit'], original_config=str(ROOT / gate['effective_config']),
                original_config_sha256=gate['effective_config_sha256'],
                best_checkpoint=str(best), best_checkpoint_sha256=sha(best),
                epoch_boundary_checkpoint=str(path), epoch_boundary_metadata=metadata)


def run_queue():
    import yaml
    OUT.mkdir(parents=True, exist_ok=True)
    with open('/tmp/phone69_460h_serial.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if live_jobs():
            raise RuntimeError('A training/controller process already exists; refusing duplicate launch')
        identity = source_identity()
        initial = inspect()
        snapshot = OUT / 'before_recovery.json'
        if not snapshot.exists():
            save(snapshot, initial)
        results = []
        for stride in STRIDES:
            source_identity()
            gate, original_config = selected(stride)
            cfg = yaml.safe_load(original_config.read_text())
            if (cfg['seed'], cfg['training']['epochs'], cfg['model']['time_reduction_stride'],
                    cfg['data']['max_token_limit']) != (42, 50, stride, gate['budget']):
                raise ValueError('Original stride recipe changed')
            resumed_config = OUT / ('stride%d_resume.yaml' % stride)
            allowed = {gate['effective_config_sha256']}
            if resumed_config.exists():
                allowed.add(sha(resumed_config))
            done = completed(stride, gate, allowed)
            if done:
                results.append(done)
                update(status='checking_groups', completed=results, identity=identity)
                continue
            boundary, metadata = latest_boundary(stride, gate['budget'])
            if metadata['resume_completed_epochs'] > 50:
                raise ValueError('Checkpoint exceeds the authorized fifty epochs')
            run_before = experiment(stride) / 'run.json'
            backup = OUT / ('stride%d_run_before_resume.json' % stride)
            if backup.exists():
                raise RuntimeError('A recovery was already launched; inspect its preserved state before another attempt')
            shutil.copyfile(run_before, backup)
            preserve_checkpoints(stride)
            if metadata['resume_completed_epochs'] == 50:
                finalize_boundary(stride, boundary, metadata)
                results.append(completed(stride, gate, allowed))
                update(status='checking_groups', completed=results, identity=identity)
                continue
            cfg['paths']['from_checkpoint'] = str(boundary)
            resumed_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            allowed.add(sha(resumed_config))
            while subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                           '--format=csv,noheader,nounits'], universal_newlines=True).strip():
                update(status='waiting_for_gpu', completed=results, active_stride=stride)
                import time
                time.sleep(30)
            log = OUT / ('stride%d_train_resume.log' % stride)
            gpu = OUT / ('stride%d_gpu_usage.csv' % stride)
            with log.open('x') as stream, gpu.open('x') as gpu_stream:
                child = subprocess.Popen([str(PYTHON), '-u', 'scripts/server/train_phone69_460h_one.py',
                                          str(resumed_config)], cwd=str(ROOT),
                                         env=dict(os.environ, RESUME_LAST='0'), stdout=stream, stderr=subprocess.STDOUT)
                update(status='training', identity=identity, completed=results, active_stride=stride,
                       training_pid=child.pid, train_log=str(log), gpu_log=str(gpu),
                       config=str(resumed_config), config_sha256=sha(resumed_config),
                       resume_checkpoint=str(boundary), resume_metadata=metadata)
                while child.poll() is None:
                    gpu_stream.write((gpu_info() or 'GPU sample unavailable') + '\n')
                    gpu_stream.flush()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            if child.returncode:
                raise RuntimeError('stride%d training exited %d; logs and checkpoints preserved' % (stride, child.returncode))
            done = completed(stride, gate, allowed)
            if not done:
                raise RuntimeError('Training exited without verified completion')
            results.append(done)
            update(completed=results)
        save(OUT / 'completion.json', dict(server_utc=now(), identity=identity, models=results))
        update(status='complete', completed=results, active_stride=None, training_pid=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inspect', 'restore', 'run', 'status', 'protect-archive'), required=True)
    args = parser.parse_args()
    os.chdir(str(ROOT))
    sys.path.insert(0, str(ROOT))
    try:
        if args.mode == 'inspect':
            print(json.dumps(inspect(), ensure_ascii=False))
        elif args.mode == 'restore':
            restore()
        elif args.mode == 'run':
            run_queue()
        elif args.mode == 'protect-archive':
            print(json.dumps(protect_mutable_archives()))
        else:
            print(json.dumps(dict(server_utc=now(), state=read(OUT / 'recovery_state.json'),
                                  live_jobs=live_jobs(), gpu=gpu_info(),
                                  runs={str(s): read(experiment(s) / 'run.json') for s in STRIDES}),
                             ensure_ascii=False))
    except Exception as error:
        if args.mode in ('run', 'restore'):
            update(status='failed', failed_mode=args.mode, error=repr(error))
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
