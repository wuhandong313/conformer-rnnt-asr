"""Default: show the plan. Execute only after explicit user approval."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'plan.json').read_text())
WORK = Path(PLAN['work'])
RESULTS = HERE / 'results'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_file(path, record):
    path = Path(path)
    if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
        raise ValueError('File does not match approved manifest: ' + str(path))


def pinned_inputs():
    for group in PLAN['sources'].values():
        for rel, expected in group['sha256'].items():
            assert digest(Path(group['root']) / rel) == expected, rel
    for item in PLAN['models']:
        assert digest(item['config']) == item['config_sha256'], item['name']
        assert digest(item['source_config']) == item['source_config_sha256'], item['name']
    expected_files = ''.join(f['relative_path'] + '\n' for f in PLAN['dataset']['files'])
    assert (HERE / 'dataset_files.txt').read_text() == expected_files


def runtime_env():
    env = os.environ.copy()
    env.update(DYLD_LIBRARY_PATH=PLAN['dyld_library_path'], CUDA_VISIBLE_DEVICES='',
               LOCAL_CPU_THREADS=str(PLAN['cpu_threads']), OMP_NUM_THREADS=str(PLAN['cpu_threads']),
               MKL_NUM_THREADS=str(PLAN['cpu_threads']), OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    return env


def transfer():
    if shutil.disk_usage(WORK).free < PLAN['transfer_bytes'] + 1024 ** 3:
        raise RuntimeError('Insufficient free disk space')
    text = (Path(PLAN['repo']) / 'remote-access.md').read_text()
    env = os.environ.copy()
    env['SSHPASS'] = re.search(r"sshpass -p '([^']+)' ssh internal", text).group(1)
    transfers = [(item['server']['best'], Path(item['checkpoint'])) for item in PLAN['models']]
    transfers.append((PLAN['models'][0]['server']['stats'], WORK / 'assets/global_stats.json'))
    for record, destination in transfers:
        if destination.exists():
            verify_file(destination, record)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + '.partial')
        subprocess.run(['sshpass', '-e', 'scp', '-q', '-o', 'ConnectTimeout=20',
                        'internal:' + record['path'], str(partial)], check=True, env=env)
        verify_file(partial, record)
        partial.replace(destination)
        print('Verified download:', destination.name, flush=True)
    local_data = WORK / 'librispeech'
    local_data.mkdir(exist_ok=True)
    subprocess.run(['sshpass', '-e', 'rsync', '-rt', '--partial',
                    '--files-from=' + str(HERE / 'dataset_files.txt'),
                    '-e', 'ssh -o ConnectTimeout=20 -o ServerAliveInterval=15',
                    'internal:' + PLAN['dataset']['root'] + '/', str(local_data) + '/'],
                   check=True, env=env)
    verify_downloads()


def verify_downloads():
    for item in PLAN['models']:
        verify_file(item['checkpoint'], item['server']['best'])
    verify_file(WORK / 'assets/global_stats.json', PLAN['models'][0]['server']['stats'])
    for record in PLAN['dataset']['files']:
        verify_file(WORK / 'librispeech' / record['relative_path'], record)
    print('Verified all checkpoint, statistics and dataset SHA-256 values.', flush=True)


def preflight():
    verify_downloads()
    for group in PLAN['sources']:
        subprocess.run([PLAN['python'], str(HERE / 'check_local.py'), group, '--with-checkpoints'],
                       check=True, env=runtime_env(), cwd=PLAN['repo'])


def validate_csv(path, expected_count):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected_count, path
    for idx, row in enumerate(rows):
        assert int(row['idx']) == idx
        assert math.isfinite(float(row['RTF'])) and float(row['RTF']) >= 0
        errors, phones = int(row['phone_errors']), int(row['reference_phones'])
        assert errors >= 0 and phones > 0
        assert phones == len(row['ref'].split())
        assert math.isclose(float(row['PER']), errors / phones, rel_tol=1e-10, abs_tol=1e-12)
    errors = sum(int(row['phone_errors']) for row in rows)
    phones = sum(int(row['reference_phones']) for row in rows)
    if expected_count == PLAN['expected_utterances']:
        assert phones == PLAN['expected_reference_phones']
    ref_digest = hashlib.sha256(json.dumps([row['ref'] for row in rows]).encode()).hexdigest()
    return dict(utterances=len(rows), phone_errors=errors, reference_phones=phones,
                corpus_per=errors / phones, mean_utterance_rtf=sum(float(r['RTF']) for r in rows) / len(rows),
                reference_order_sha256=ref_digest, csv_sha256=digest(path))


def run_one(item, phase, max_utts):
    directory = RESULTS / phase
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (item['name'] + '.csv')
    log = directory / (item['name'] + '.log')
    if output.exists() or log.exists():
        raise FileExistsError('Existing result/log needs review before a retry: ' + str(output))
    command = [PLAN['python'], '-u', str(Path(PLAN['sources'][item['group']]['root']) / 'eval_cpu.py'),
               item['config'], '--checkpoint', item['checkpoint'], '--cpu', '--metric', 'per', '--out', str(output)]
    if max_utts:
        command += ['--max-utts', str(max_utts)]
    print('Starting', phase, item['name'], flush=True)
    started = time.perf_counter()
    with log.open('x') as stream:
        subprocess.run(command, check=True, env=runtime_env(), cwd=PLAN['sources'][item['group']]['root'],
                       stdout=stream, stderr=subprocess.STDOUT)
    result = validate_csv(output, max_utts or PLAN['expected_utterances'])
    result.update(name=item['name'], phase=phase, process_wall_seconds=time.perf_counter() - started)
    return result


def evaluate():
    preflight()
    smoke = [run_one(item, 'smoke', PLAN['expected_smoke_utterances']) for item in PLAN['models']]
    (RESULTS / 'smoke_summary.json').write_text(json.dumps(smoke, indent=2) + '\n')
    full = []
    for item in PLAN['models']:
        full.append(run_one(item, 'full', 0))
        assert len({row['reference_order_sha256'] for row in full}) == 1
        (RESULTS / 'summary.json').write_text(json.dumps(full, indent=2) + '\n')
    (RESULTS / 'COMPLETE').touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-after-approval', action='store_true')
    parser.add_argument('--stage', choices=['all', 'transfer', 'preflight', 'eval'], default='all')
    args = parser.parse_args()
    pinned_inputs()
    print('Transfer MiB:', round(PLAN['transfer_bytes'] / 1024 ** 2, 3))
    print('Serial CPU order:', ', '.join(item['name'] for item in PLAN['models']))
    if not args.execute_after_approval:
        print('Dry run only. User approval is required before transfer/eval.')
        return
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'serial.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage in ['all', 'transfer']:
            transfer()
        if args.stage == 'preflight':
            preflight()
        if args.stage in ['all', 'eval']:
            evaluate()


if __name__ == '__main__':
    main()
