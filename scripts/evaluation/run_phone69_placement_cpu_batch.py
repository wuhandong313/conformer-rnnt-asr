#!/usr/bin/env python3
"""Download completed L1/L3/L9 weights and evaluate them serially at home."""
from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from eval_phone69_placement_checkpoint import THREAD_ENV, now, sha, verified, write_json

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'papers/phone69_460h_cpu_authoritative_20261108/supporting/original_archive'
OUT = ROOT / 'artifacts/evaluations/phone69_placement_l1_l3_l9_local_cpu_20261117'
MODELS = ROOT / 'output/local_cpu_20261117/models'
HELPERS = ('theoretical_phone69_macs.py', 'measure_phone69_460h_compute.py',
           'run_phone69_460h_cpu_batch.py')


def state(**fields):
    path = OUT / 'controller_state.json'
    value = json.loads(path.read_text()) if path.exists() else {}
    value.update(fields, controller_pid=os.getpid(), updated_at_utc=now())
    write_json(path, value)
    print(json.dumps(value), flush=True)


def download(entry):
    folder = MODELS / entry['name']
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(entry['best_checkpoint']).name
    if target.exists():
        verified(target, entry['checkpoint_sha256'])
        return target
    partial = target.with_suffix('.ckpt.part')
    if partial.exists():
        if partial.stat().st_size == entry['checkpoint_bytes'] and sha(partial) == entry['checkpoint_sha256']:
            partial.replace(target)
            return target
        raise FileExistsError('Inspect the preserved partial checkpoint: ' + str(partial))
    private = ROOT / 'remote-access.md'
    line = next(line.strip() for line in private.read_text().splitlines()
                if line.strip().startswith('sshpass ') and ' ssh internal' in line)
    args = shlex.split(line)
    secret = args[args.index('-p') + 1]
    env = dict(os.environ, SSHPASS=secret)
    state(status='downloading_checkpoint', active_model=entry['name'], checkpoint_bytes=entry['checkpoint_bytes'])
    result = subprocess.run(['sshpass', '-e', 'scp', '-O', '-o', 'ConnectTimeout=15',
                             'internal:' + entry['best_checkpoint'], str(partial)],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=600)
    if result.returncode:
        raise ConnectionError('Checkpoint download failed: ' + result.stderr.replace(secret, '[redacted]')[-800:])
    if partial.stat().st_size != entry['checkpoint_bytes']:
        raise ValueError('Incomplete checkpoint transfer')
    verified(partial, entry['checkpoint_sha256'])
    partial.replace(target)
    write_json(folder / 'transfer.json', dict(server_path=entry['best_checkpoint'],
               local_path=str(target), bytes=target.stat().st_size, sha256=sha(target), completed_at_utc=now()))
    return target


def freeze_sources(entry, folder):
    source = folder / 'model_source'
    commit = entry['run']['code_commit']
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, '--',
                                    'conformer_rnnt', 'common.py', 'phone69_tokenizer.py'], cwd=ROOT, text=True).splitlines()
    names = [name for name in names if name.endswith('.py')]
    if not names or 'conformer_rnnt/staged_encoder.py' not in names:
        raise ValueError('Training commit does not contain the placement model')
    for name in names:
        data = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Frozen model source changed: ' + name)
        target.write_bytes(data)
    return source, {str(p.relative_to(source)): sha(p) for p in source.rglob('*.py')}


def prepare(metadata, entry, checkpoint):
    import yaml

    run = entry['run']
    if (run['status'], run['epoch'], run['epochs'], run['tokenizer'], run['val_subset']) != (
            'trained', 50, 50, 'phone69', 'dev-clean'):
        raise ValueError('Expected a completed dev-selected Phone69 training run')
    if entry['training_config_sha256'] != run['config_sha256']:
        raise ValueError('Actual training configuration identity differs')
    folder = OUT / entry['name']
    folder.mkdir(exist_ok=True)
    source, fingerprints = freeze_sources(entry, folder)
    inputs = OUT / 'inputs'
    inputs.mkdir(exist_ok=True)
    for name, key in (('global_stats.json', 'stats'), ('phone69_vocab.json', 'vocab')):
        path = inputs / name
        data = base64.b64decode(metadata[key + '_b64'], validate=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError('Preprocessing inputs differ between models')
        path.write_bytes(data)
        verified(path, metadata[key + '_sha256'])
    original = folder / 'training_config.yaml'
    original.write_text(entry['training_config_text'])
    verified(original, entry['training_config_sha256'])
    cfg = yaml.safe_load(original.read_text())
    baseline = yaml.safe_load((ARCHIVE / 'E09/config.yaml').read_text())
    cfg = copy.deepcopy(cfg)
    cfg['paths'].update(dataset_path=baseline['paths']['dataset_path'],
                        global_stats_path=str(inputs / 'global_stats.json'),
                        exp_dir=str(checkpoint.parent), from_checkpoint=None)
    cfg['tokenizer']['vocab_path'] = str(inputs / 'phone69_vocab.json')
    cfg['data']['num_workers'] = 0
    cfg['trainer']['accelerator'] = 'cpu'
    cfg['trainer']['strategy'] = 'auto'
    cfg['trainer']['devices'] = 1
    if cfg['decoding']['beam_width'] != 20 or cfg['model']['compression'].get('execution_backend', 'reference') != 'reference':
        raise ValueError('Preserve beam20 and the reference compression backend')
    if [s['after_layer'] for s in cfg['model']['compression']['stages']] != [entry['position']]:
        raise ValueError('Wrong compression placement')
    config = folder / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    frozen = json.loads((ARCHIVE / 'manifest.json').read_text())
    baseline_entry = next(item for item in frozen['models'] if item['id'] == 'E09')
    if metadata['stats_sha256'] != baseline_entry['global_stats_sha256']:
        raise ValueError('Training statistics differ from the comparison protocol')
    result = json.loads((ARCHIVE / 'E09/result.json').read_text())
    reference = ARCHIVE / result['csv']
    manifest = dict(schema=1, model=dict(name=entry['name'], position=entry['position'],
                    expected_parameters=4236299, training_commit=run['code_commit'],
                    checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                    config=str(config), config_sha256=sha(config),
                    training_config=str(original), training_config_sha256=sha(original),
                    offline_native_definition=run['offline_native_definition'], server_training=run),
                    source=str(source), code_sha256=fingerprints,
                    evaluator_sha256=sha(ROOT / 'scripts/evaluation/eval_phone69_placement_checkpoint.py'),
                    helper_sha256={name: sha(ROOT / 'scripts/evaluation' / name) for name in HELPERS},
                    runtime=frozen['runtime'], audio=frozen['audio'], chapter_labels=frozen['chapter_labels'],
                    global_stats_sha256=metadata['stats_sha256'], vocab_sha256=metadata['vocab_sha256'],
                    reference_csv=str(reference), reference_csv_sha256=sha(reference),
                    reference_manifest_sha256=sha(ARCHIVE / 'manifest.json'),
                    protocol=dict(device='cpu', dtype='float32', batch=1, beam=20, threads=4,
                                  interop_threads=1, data_workers=0, openmp_max_active_levels=1,
                                  warmup_utterances=10, subset='test-clean'))
    path = folder / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Preserve the existing manifest: input identity changed')
    write_json(path, manifest)
    return path, manifest


def evaluate(entry, path, manifest):
    folder = path.parent
    attempt = folder / 'attempt_01'
    if (attempt / 'result.json').is_file():
        result = json.loads((attempt / 'result.json').read_text())
        if not result.get('verified_full_corpus') or result['manifest_sha256'] != sha(path):
            raise ValueError('Existing evaluation does not match the frozen manifest')
        verified(attempt / 'utterances.csv', result['csv_sha256'])
        return result
    attempt.mkdir(exist_ok=False)
    runtime_dir = OUT / 'runtime_lib'
    runtime_dir.mkdir(exist_ok=True)
    library = Path(manifest['runtime']['openmp']['path'])
    verified(library, manifest['runtime']['openmp']['sha256'])
    link = runtime_dir / 'libomp.dylib'
    if not link.exists():
        link.symlink_to(library)
    env = dict(os.environ, **{key: '4' for key in THREAD_ENV}, OMP_MAX_ACTIVE_LEVELS='1',
               DYLD_LIBRARY_PATH=str(runtime_dir))
    evaluator = ROOT / 'scripts/evaluation/eval_phone69_placement_checkpoint.py'
    with (attempt / 'run.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(evaluator), '--manifest', str(path),
                                  '--out', str(attempt)], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        state(status='evaluating', active_model=entry['name'], evaluation_pid=child.pid,
              progress=str(attempt / 'progress.json'), log=str(attempt / 'run.log'))
        rc = child.wait()
    if rc:
        raise RuntimeError('Evaluation failed; preserve and inspect ' + str(attempt / 'run.log'))
    result = json.loads((attempt / 'result.json').read_text())
    if not result.get('verified_full_corpus') or result['utterances'] != 2620:
        raise ValueError('Full-corpus evaluation did not finish')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    if [entry['position'] for entry in metadata['models']] != [1, 3, 9]:
        raise ValueError('Expected the three requested placements in order')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        state(status='preparing', completed_models=[])
        prepared = []
        for entry in metadata['models']:
            checkpoint = download(entry)
            path, manifest = prepare(metadata, entry, checkpoint)
            prepared.append((entry, path, manifest))
        for entry, path, manifest in prepared:
            result = evaluate(entry, path, manifest)
            results[entry['name']] = result
            write_json(OUT / 'summary.json', dict(complete=len(results) == 3, models=results,
                                                  updated_at_utc=now()))
            state(completed_models=list(results), evaluation_pid=None)
        (OUT / 'EVALUATION_COMPLETE').write_text(now() + '\n')
        state(status='complete', completed_models=list(results), active_model=None)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            state(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
