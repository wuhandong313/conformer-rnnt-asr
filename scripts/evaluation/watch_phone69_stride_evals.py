#!/usr/bin/env python3
"""Fetch verified completed stride models and evaluate them serially locally.

All server access goes through the existing credential-reading wrapper. The
server is queried and read only; this process never starts server training.
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from eval_phone69_stride_checkpoint import now, sha, verified, write_json, THREAD_ENV

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111'
OPS = ROOT / 'output/phone69_460h/stride_recovery_20261111'
MODELS = ROOT / 'output/local_cpu_20261111/models'
ARCHIVE = ROOT / 'papers/phone69_460h_cpu_authoritative_20261108/supporting/original_archive'
SERVER_ROOT = '/test21/conformer-rnnt-asr'
SERVER_STATE = SERVER_ROOT + '/output/phone69_460h_stride_sweep_recovery_20261111/recovery_state.json'
PARAMETERS = {1: 4215622, 2: 10378310, 8: 95523398}
CHUNK = 24 * 1024 * 1024


def state(**fields):
    path = OUT / 'controller_state.json'
    value = json.loads(path.read_text()) if path.is_file() else {}
    value.update(fields, updated_at_utc=now(), controller_pid=os.getpid())
    write_json(path, value)


def remote(script, timeout=120):
    for attempt in range(3):
        result = subprocess.run([sys.executable, str(ROOT / 'output/phone69_460h/remote.py'),
                                 str(script), '--runtime', 'system', '--timeout', str(timeout)],
                                cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        if attempt < 2:
            time.sleep(2)
    raise ConnectionError('Read-only server operation failed: ' + (result.stderr + result.stdout)[-1500:])


def remote_state():
    script = OPS / 'eval_read_state_server.py'
    script.write_text('import json\nfrom pathlib import Path\np=Path(%r)\n'
                      'print(p.read_text() if p.is_file() else "{}")\n' % SERVER_STATE)
    return json.loads(remote(script, 60))


def metadata(group):
    stride = group['stride']
    script = OPS / ('eval_stride%d_metadata_server.py' % stride)
    script.write_text('''import base64, hashlib, json
from pathlib import Path
root=Path(%r)
stride=%d
state=json.loads((root/'output/phone69_460h_stride_sweep_recovery_20261111/recovery_state.json').read_text())
group=next(g for g in state['completed'] if g['stride']==stride)
assert group['status']=='trained' and group['epoch']==50
assert group['epoch_boundary_metadata']['resume_completed_epochs']==50
assert Path(group['epoch_boundary_checkpoint']).is_file()
checkpoint=Path(group['best_checkpoint'])
assert checkpoint.parent==root/('exp_phone69_460h_offline_native_stride%%d_none_proj128_s42_e50'%%stride)/'checkpoints'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert sha(checkpoint)==group['best_checkpoint_sha256']
config=Path(group['original_config'])
assert sha(config)==group['original_config_sha256']
sources=state['identity']['source_sha256']
assert all(sha(root/p)==h for p,h in sources.items())
stats=Path('/test21/librispeech/global_stats.json')
vocab=root/'configs/phone69_vocab.json'
print(json.dumps(dict(group=group,bytes=checkpoint.stat().st_size,training_config=config.read_text(),
 source_sha256=sources,global_stats_b64=base64.b64encode(stats.read_bytes()).decode(),
 global_stats_sha256=sha(stats),vocab_b64=base64.b64encode(vocab.read_bytes()).decode(),
 vocab_sha256=sha(vocab))))
''' % (SERVER_ROOT, stride))
    value = json.loads(remote(script))
    write_json(OPS / ('stride%d_server_metadata.json' % stride), value)
    return value


def download(meta):
    group = meta['group']
    folder = MODELS / group['name']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(group['best_checkpoint']).name
    if path.is_file():
        verified(path, group['best_checkpoint_sha256'])
        return path
    partial = path.with_suffix(path.suffix + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > meta['bytes'] or (offset != meta['bytes'] and offset % CHUNK):
        raise ValueError('Inspect the preserved partial checkpoint before resuming its transfer')
    script = OPS / 'eval_download_chunk_server.py'
    with partial.open('ab') as stream:
        while offset < meta['bytes']:
            length = min(CHUNK, meta['bytes'] - offset)
            script.write_text('import base64,sys\nfrom pathlib import Path\np=Path(%r)\n'
                              'assert p.stat().st_size==%d\nwith p.open("rb") as f:\n'
                              ' f.seek(%d)\n b=f.read(%d)\n assert len(b)==%d\n'
                              ' sys.stdout.write(base64.b64encode(b).decode())\n'
                              % (group['best_checkpoint'], meta['bytes'], offset, length, length))
            data = base64.b64decode(remote(script, 180), validate=True)
            if len(data) != length:
                raise ValueError('Checkpoint transfer returned an incomplete chunk')
            stream.write(data)
            stream.flush()
            offset += length
            state(status='downloading_checkpoint', active_model=group['name'],
                  transferred_bytes=offset, checkpoint_bytes=meta['bytes'])
    verified(partial, group['best_checkpoint_sha256'])
    partial.replace(path)
    write_json(folder / 'transfer.json', dict(server_path=group['best_checkpoint'], local_path=str(path),
               bytes=meta['bytes'], sha256=sha(path), completed_at_utc=now()))
    return path


def freeze_sources(meta):
    source = OUT / 'model_source'
    for relative, digest in meta['source_sha256'].items():
        verified(ROOT / relative, digest)
        if relative.startswith('conformer_rnnt/') or relative in ('common.py', 'phone69_tokenizer.py'):
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(ROOT / relative, target)
            verified(target, digest)
    for name in ('measure_phone69_460h_compute.py', 'run_phone69_460h_cpu_batch.py'):
        relative = Path('scripts/evaluation') / name
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(ROOT / relative, target)
    return source


def prepare(meta, checkpoint):
    import yaml
    group = meta['group']
    folder = OUT / group['name']
    folder.mkdir(exist_ok=True)
    source = freeze_sources(meta)
    inputs = OUT / 'inputs'
    inputs.mkdir(exist_ok=True)
    for name, key in (('global_stats.json', 'global_stats'), ('phone69_vocab.json', 'vocab')):
        path = inputs / name
        data = base64.b64decode(meta[key + '_b64'], validate=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError('Server preprocessing inputs differ across stride groups')
        path.write_bytes(data)
        verified(path, meta[key + '_sha256'])
    original = folder / 'training_config.yaml'
    original.write_text(meta['training_config'])
    verified(original, group['original_config_sha256'])
    training = yaml.safe_load(meta['training_config'])
    baseline = yaml.safe_load((ARCHIVE / 'E09/config.yaml').read_text())
    cfg = copy.deepcopy(training)
    cfg['paths'].update(dataset_path=baseline['paths']['dataset_path'],
                        global_stats_path=str(inputs/'global_stats.json'),
                        exp_dir=str(checkpoint.parent), from_checkpoint=None)
    cfg['tokenizer']['vocab_path'] = str(inputs/'phone69_vocab.json')
    cfg['data']['num_workers'] = 0
    cfg['trainer']['accelerator'] = 'cpu'
    if cfg['decoding']['beam_width'] != 20:
        raise ValueError('Training recipe does not preserve the beam-20 protocol')
    config = folder / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    frozen = json.loads((ARCHIVE/'manifest.json').read_text())
    baseline_entry = next(e for e in frozen['models'] if e['id'] == 'E09')
    if meta['global_stats_sha256'] != baseline_entry['global_stats_sha256']:
        raise ValueError('Training feature normalization differs from the comparison corpus')
    baseline_result = json.loads((ARCHIVE/'E09/result.json').read_text())
    reference = ARCHIVE / baseline_result['csv']
    manifest = dict(schema=1, model=dict(name=group['name'], stride=group['stride'],
                    expected_parameters=PARAMETERS[group['stride']], training_commit=group['training_commit'],
                    checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                    training_config=str(original), training_config_sha256=sha(original),
                    config=str(config), config_sha256=sha(config), server_training=group),
                    source=str(source), code_sha256={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
                    evaluator_sha256=sha(ROOT/'scripts/evaluation/eval_phone69_stride_checkpoint.py'),
                    runtime=frozen['runtime'], global_stats_sha256=meta['global_stats_sha256'],
                    vocab_sha256=meta['vocab_sha256'], audio=frozen['audio'], chapter_labels=frozen['chapter_labels'],
                    reference_csv=str(reference), reference_csv_sha256=sha(reference),
                    reference_manifest_sha256=sha(ARCHIVE/'manifest.json'),
                    protocol=dict(device='cpu',dtype='float32',batch=1,beam=20,threads=4,interop_threads=1,
                                  data_workers=0,openmp_max_active_levels=1,warmup_utterances=10,subset='test-clean'))
    path = folder/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Preserve the previous manifest; evaluation inputs have changed')
    write_json(path, manifest)
    return folder, path, manifest


def evaluate(meta, checkpoint):
    folder, manifest_path, manifest = prepare(meta, checkpoint)
    attempt = folder/'attempt_01'
    if (attempt/'result.json').is_file():
        result = json.loads((attempt/'result.json').read_text())
        if not result.get('verified_full_corpus') or result['manifest_sha256'] != sha(manifest_path):
            raise ValueError('Existing evaluation result is not verified for this manifest')
        verified(attempt/'utterances.csv', result['csv_sha256'])
        return result
    attempt.mkdir(exist_ok=False)
    runtime_dir = OUT/'runtime_lib'
    runtime_dir.mkdir(exist_ok=True)
    library = Path(manifest['runtime']['openmp']['path'])
    verified(library, manifest['runtime']['openmp']['sha256'])
    link = runtime_dir/'libomp.dylib'
    if not link.exists():
        link.symlink_to(library)
    env = dict(os.environ, **{key:'4' for key in THREAD_ENV}, OMP_MAX_ACTIVE_LEVELS='1',
               DYLD_LIBRARY_PATH=str(runtime_dir))
    with (attempt/'run.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/evaluation/eval_phone69_stride_checkpoint.py'),
                                  '--manifest', str(manifest_path), '--out', str(attempt)],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state(status='evaluating', active_model=meta['group']['name'], evaluation_pid=child.pid,
              progress=str(attempt/'progress.json'), log=str(attempt/'run.log'))
        rc = child.wait()
    if rc:
        raise RuntimeError('Local evaluation exited %d; inspect preserved %s' % (rc, attempt/'run.log'))
    result = json.loads((attempt/'result.json').read_text())
    if not result.get('verified_full_corpus') or result['utterances'] != 2620:
        raise ValueError('Full-corpus evaluation did not finish')
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    OPS.mkdir(parents=True, exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        state(status='starting', completed_models=[])
        while len(results) < 3:
            try:
                server = remote_state()
            except ConnectionError as error:
                state(status='waiting_for_server', connection_error=str(error), completed_models=list(results))
                time.sleep(300)
                continue
            available = {g['stride']:g for g in server.get('completed', [])}
            for stride in (1, 2, 8):
                if stride in results or stride not in available:
                    continue
                meta = metadata(available[stride])
                checkpoint = download(meta)
                result = evaluate(meta, checkpoint)
                results[stride] = result
                write_json(OUT/'summary.json', dict(complete=len(results)==3, models=results,
                                                   protocol=result['protocol'], updated_at_utc=now()))
                state(status='waiting_for_training', completed_models=list(results), evaluation_pid=None)
            if server.get('status') == 'failed':
                raise RuntimeError('Server recovery/training failed: ' + str(server.get('error')))
            if len(results) < 3:
                state(status='waiting_for_training', completed_models=list(results),
                      pending_strides=[s for s in (1,2,8) if s not in results])
                time.sleep(60 if not available else 300)
        (OUT/'EVALUATION_COMPLETE').write_text(now()+'\n')
        state(status='complete', completed_models=list(results), active_model=None)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.is_dir():
            state(status='failed', error=repr(error))
        traceback.print_exc()
        raise SystemExit(1)
