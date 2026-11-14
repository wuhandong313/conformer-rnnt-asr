#!/usr/bin/env python3
"""Resume the authorized placement queue without repeating completed runs.

The recorded runtime is restored separately. This controller preserves the
original audit, configurations, logs and independent checkpoint copies; resumed
training changes only paths.from_checkpoint and retains the selected budget.
"""

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from conformer_rnnt.checkpoint_utils import phone69_resume_metadata
from scripts.server.phone69_placement_supplement import (
    PlacementQueue, live_jobs as original_live_jobs, load_plan, local_path,
    sha, save, validate_recipe, verify_completed,
)

ORIGINAL = ROOT / 'output/phone69_460h_placement_supplement_20261112'
OUT = ROOT / 'output/phone69_460h_placement_recovery_20261114'
MANIFEST = ROOT / 'configs/phone69_placement_supplement_20261112.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def live_jobs():
    jobs = original_live_jobs()
    for path in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(path.name)
            if pid == os.getpid() or (path / 'stat').read_text().split()[2] == 'Z':
                continue
            args = (path / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if Path(__file__).name in {Path(a).name for a in args if a} and '--mode' in args:
                if args[args.index('--mode') + 1:][:1] == ['run']:
                    jobs.append(dict(pid=pid, kind='recovery_controller'))
        except (OSError, ValueError, IndexError):
            pass
    return jobs


def source_identity():
    deployment = read(ORIGINAL / 'deployment.json')
    if not deployment or not deployment.get('source_sha256'):
        raise ValueError('Original placement deployment fingerprints are missing')
    changed = [name for name, digest in deployment['source_sha256'].items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    if changed:
        raise ValueError('Original source/configuration changed: ' + ', '.join(changed))
    return deployment


def preserve_checkpoints(experiment, destination):
    """Copy every old checkpoint to an independent inode before any resume."""
    experiment, destination = Path(experiment), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sorted((experiment / 'checkpoints').glob('*.ckpt')):
        target = destination / source.name
        digest = sha(source)
        if not target.exists():
            shutil.copy2(source, target)
        if sha(target) != digest or source.stat().st_ino == target.stat().st_ino:
            raise ValueError('Checkpoint preservation failed: ' + source.name)
        records.append(dict(original=str(source), preserved=str(target), sha256=digest,
                            bytes=target.stat().st_size, independent_copy=True))
    save(destination / 'inventory.json', records)
    return records


def boundary_key(path):
    found = re.fullmatch(r'epoch-boundary-epoch=(\d+)-step=(\d+)-[a-f0-9]+\.ckpt', Path(path).name)
    return tuple(map(int, found.groups())) if found else (-1, -1)


def choose_boundary(paths, budget, definition, loader=None):
    if not definition:
        raise ValueError('Original run has no placement definition')
    if loader is None:
        import torch
        loader = lambda path: torch.load(path, map_location='cpu', weights_only=False)
    errors = []
    for path in sorted(paths, key=boundary_key, reverse=True):
        if boundary_key(path)[0] < 0:
            continue
        try:
            checkpoint = loader(path)
            metadata = phone69_resume_metadata(checkpoint, budget, expected_definition=definition)
            if (not metadata['resume_at_epoch_boundary'] or not checkpoint.get('state_dict')
                    or metadata['previous_max_token_limit'] != budget):
                raise ValueError('Not a complete boundary at the original token budget')
            if (metadata['resume_completed_epochs'], metadata['resume_global_step']) != boundary_key(path):
                raise ValueError('Checkpoint filename and actual loop progress disagree')
            del checkpoint
            return Path(path), metadata, errors
        except Exception as error:
            errors.append(dict(path=str(path), error=str(error)))
    raise ValueError('No intact full epoch-boundary checkpoint: ' + repr(errors))


class RecoveryQueue(PlacementQueue):
    def __init__(self):
        self.manifest, self.plan = load_plan(MANIFEST)
        original = source_identity()
        self.audit = OUT
        self.audit.mkdir(parents=True, exist_ok=True)
        if (self.audit / 'deployment.json').exists():
            raise FileExistsError('Recovery was already launched; inspect rather than relaunch')
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        self.sources = dict(original['source_sha256'])
        for relative in ['scripts/server/recover_phone69_placement_supplement.py',
                         'artifacts/recoveries/phone69_post_restart_20261021/requirements-restored.txt']:
            self.sources[relative] = sha(ROOT / relative)
        self.completed, self.trial = [], 0
        old_state = self.audit / 'recovery_state.json'
        if old_state.exists():
            shutil.copy2(old_state, self.audit / 'environment_install_state.json')
        if (ORIGINAL / 'queue_state.json').exists():
            shutil.copy2(ORIGINAL / 'queue_state.json', self.audit / 'original_queue_state.json')
        save(self.audit / 'deployment.json', dict(code_commit=self.commit, server_utc=now(),
            controller_pid=os.getpid(), original_deployment_commit=original['code_commit'],
            original_source_files_verified=len(original['source_sha256']),
            source_sha256=self.sources, manifest=self.plan))
        self.state('starting')

    def state(self, status, **fields):
        value = dict(status=status, controller_pid=os.getpid(), code_commit=self.commit,
                     server_utc=now(), completed=self.completed, **fields)
        save(self.audit / 'recovery_state.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def selected(self, item):
        gate = read(ORIGINAL / (item['name'] + '_selected.json'))
        if gate is None:
            return None
        config_path = local_path(gate['effective_config'])
        if not gate.get('valid') or sha(config_path) != gate['effective_config_sha256']:
            raise ValueError('Original selected configuration changed: ' + item['name'])
        if gate['base_config_sha256'] != self.sources[item['config']]:
            raise ValueError('Selected recipe does not match the original base configuration')
        config = yaml.safe_load(config_path.read_text())
        baseline = yaml.safe_load(local_path(item['config']).read_text())
        validate_recipe(config, baseline, item['positions'], fresh=True)
        if config['data']['max_token_limit'] != gate['budget']:
            raise ValueError('Selected batch budget does not match its configuration')
        if not any(c['passed'] and c['budget'] == gate['budget'] and c['optimizer_steps'] >= 24
                   for c in gate['confirmations']):
            raise ValueError('Original budget lacks the full real-data confirmation')
        return config_path, config, gate

    def verify(self, item, selected=None):
        actual = copy.deepcopy(item)
        actual['action'] = 'reuse'  # Resume YAML may differ only in from_checkpoint.
        result = verify_completed(actual, budget=selected[2]['budget'] if selected else None)
        if selected and result['config_sha256'] != selected[2]['effective_config_sha256']:
            raise ValueError('Completed original run does not use its selected configuration')
        result['action'] = item['action']
        result['skipped_completed'] = True
        return result

    def prepare_resume(self, item, selected, run):
        original_path, config, gate = selected
        if (run['config_sha256'] != gate['effective_config_sha256']
                or run['max_token_limit'] != gate['budget']):
            raise ValueError('Interrupted run differs from its selected recipe/budget')
        if (run.get('compression') != config['model']['compression']
                or run.get('internal_dim') != 128 or run.get('time_reduction_stride') != 1):
            raise ValueError('Interrupted run model differs from its original configuration')
        experiment = Path(config['paths']['exp_dir'])
        destination = self.audit / 'original_checkpoints' / item['name']
        inventory = preserve_checkpoints(experiment, destination)
        before = self.audit / (item['name'] + '_run_before_resume.json')
        if before.exists():
            raise FileExistsError('A prior resume already exists; do not overwrite it')
        shutil.copy2(experiment / 'run.json', before)
        boundary, metadata, errors = choose_boundary(
            (experiment / 'checkpoints').glob('epoch-boundary-*.ckpt'),
            gate['budget'], run.get('offline_native_definition'))
        if metadata['resume_completed_epochs'] >= 50:
            raise ValueError('Final boundary exists but run needs completion-marker audit, not more training')
        resumed = copy.deepcopy(config)
        resumed['paths']['from_checkpoint'] = str(boundary)
        effective = self.audit / (item['name'] + '_resume.yaml')
        with effective.open('x') as stream:
            yaml.safe_dump(resumed, stream, sort_keys=False, allow_unicode=True)
        record = dict(name=item['name'],original_config=str(original_path),
            original_config_sha256=gate['effective_config_sha256'],
            resume_config=str(effective),resume_config_sha256=sha(effective),budget=gate['budget'],
            checkpoint=str(boundary),checkpoint_sha256=sha(boundary),metadata=metadata,
            original_run_commit=run['code_commit'],boundary_read_errors=errors,
            preserved_checkpoint_count=len(inventory),only_from_checkpoint_changed=True)
        save(self.audit / (item['name'] + '_resume.json'), record)
        return effective, resumed, record

    def train_resume(self, item, selected, run):
        self.unchanged()
        self.wait_idle()
        effective, config, record = self.prepare_resume(item, selected, run)
        logs = self.audit / 'logs'
        logs.mkdir(exist_ok=True)
        log_path, gpu_path = logs / (item['name'] + '_train_resume.log'), logs / (item['name'] + '_gpu_usage.csv')
        training_commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        with log_path.open('x') as log, gpu_path.open('x') as gpu:
            child = subprocess.Popen([sys.executable,'-u','scripts/server/train_phone69_460h_one.py',str(effective)],
                cwd=ROOT,env=dict(os.environ,RESUME_LAST='0'),stdout=log,stderr=subprocess.STDOUT)
            self.state('training',experiment=item['name'],positions=item['positions'],budget=record['budget'],
                training_pid=child.pid,training_commit=training_commit,effective_config=str(effective),
                resume_record=record,train_log=str(log_path),gpu_log=str(gpu_path))
            while child.poll() is None:
                subprocess.run(['nvidia-smi','-i','0','--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader'],stdout=gpu,stderr=subprocess.DEVNULL)
                gpu.flush()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        if child.returncode:
            raise RuntimeError('Resumed training failed; original and resumed checkpoints/logs preserved; no retry')
        self.unchanged()
        actual = copy.deepcopy(item)
        actual['action'] = 'reuse'
        result = verify_completed(actual,budget=record['budget'])
        if result['config_sha256'] != record['resume_config_sha256'] or result['training_commit'] != training_commit:
            raise ValueError('Resumed run identity changed')
        result.update(action='train',resumed=True,original_run_commit=record['original_run_commit'])
        self.completed.append(result)
        save(self.audit / (item['name'] + '_complete.json'),result)

    def run(self):
        with open('/tmp/phone69_460h_serial.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            if live_jobs():
                raise RuntimeError('Another queue/worker is alive; recovery will not start duplicate work')
            pending = []
            for item in self.plan['experiments']:
                config = yaml.safe_load(local_path(item['config']).read_text())
                experiment = Path(item.get('experiment',config['paths']['exp_dir']))
                run = read(experiment / 'run.json')
                selected = self.selected(item) if item['action'] == 'train' else None
                if item['action'] == 'train' and run and selected is None:
                    raise ValueError('Original run is missing its verified selected budget')
                if item['action'] == 'reuse' or (run and run.get('status') == 'trained'):
                    self.state('verifying_completed',experiment=item['name'])
                    result = self.verify(item,selected)
                    self.completed.append(result)
                    save(self.audit / (item['name'] + '_skipped.json'),result)
                elif run:
                    if selected is None:
                        raise ValueError('Interrupted run is missing its selected budget')
                    pending.append((item,selected,run))
                else:
                    if experiment.exists():
                        raise ValueError('Unexpected pre-existing experiment without run.json')
                    pending.append((item,selected,None))
            for item,selected,run in pending:
                if run:
                    self.train_resume(item,selected,run)
                else:
                    selected = selected or self.tune(item)
                    self.train(item,*selected)
            save(self.audit / 'completion.json',dict(server_utc=now(),code_commit=self.commit,models=self.completed))
            self.state('complete')


def status():
    deployment = read(OUT / 'deployment.json')
    changed = [name for name,digest in (deployment or {}).get('source_sha256',{}).items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    return dict(server_utc=now(),state=read(OUT / 'recovery_state.json'),deployment=deployment,
        completion=read(OUT / 'completion.json'),live_jobs=live_jobs(),
        integrity_ok=deployment is not None and not changed,integrity_issues=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['validate','run','status'],required=True)
    args = parser.parse_args()
    if args.mode == 'validate':
        load_plan(MANIFEST)
        print(json.dumps(dict(valid=True,manifest=str(MANIFEST))))
    elif args.mode == 'status':
        print(json.dumps(status(),ensure_ascii=False))
    else:
        os.chdir(ROOT)
        os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
        queue = RecoveryQueue()
        try:
            queue.run()
        except BaseException as error:
            queue.state('failed',error=repr(error))
            raise


if __name__ == '__main__':
    main()
