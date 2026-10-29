#!/usr/bin/env python3
from datetime import datetime, timezone
from pathlib import Path
import copy, fcntl, hashlib, json, os, re, runpy, subprocess, sys
import yaml

repo = Path('/test21/conformer-rnnt-asr')
audit = repo / 'artifacts/runs/phone69_460h_placement_layer7_stride1_20261027'
config_path = repo / 'configs/server_phone69_460h_placement_layer7_stride1_s42_e50.yaml'
root = repo / 'exp_phone69_460h_offline_native_placement_layer7_stride1_s42_e50'
base_path = repo / 'configs/server_phone69_460h_placement_layer5_stride1_s42_e50.yaml'
os.chdir(repo)
sys.path.insert(0, str(repo))
os.environ.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
helper = runpy.run_path(str(repo / 'scripts/server/phone69_placement_pipeline.py'))
now = lambda: datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)

def state(status, **values):
    record = dict(status=status, server_utc=now(), controller_pid=os.getpid(), run_phase='placement_layer7_stride1_20261027', **values)
    save(audit / 'queue_state.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)

def capture_protected():
    existing = ['mel', 'layer5', 'layers2_5_n3']
    result = {'experiments': {}, 'configs': {}, 'legacy_manifest': str(repo / 'artifacts/runs/phone69_460h_placement_stride1_20261022/protected_manifest.json')}
    helper['protected_unchanged']()
    for name in existing:
        exp = repo / ('exp_phone69_460h_offline_native_placement_' + name + '_stride1_s42_e50')
        run_path = exp / 'run.json'
        run = json.loads(run_path.read_text())
        assert run['status'] == 'trained' and run['epoch'] == 50 and (exp / 'TRAINING_COMPLETE').is_file()
        critical = {str(run_path), str(exp / 'best_checkpoint.txt'), run['best_checkpoint'], run['epoch_boundary_checkpoint']}
        files = []
        for p in sorted(exp.rglob('*')):
            if not p.is_file():
                continue
            stat = p.stat()
            row = {'path': str(p.relative_to(repo)), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if str(p) in critical:
                row['sha256'] = sha(p)
            files.append(row)
        assert all(Path(p).is_file() for p in critical)
        result['experiments'][str(exp.relative_to(repo))] = files
        cfg = repo / run['config']
        result['configs'][str(cfg.relative_to(repo))] = sha(cfg)
    return result

def check_protected(manifest):
    helper['protected_unchanged']()
    for relative, rows in manifest['experiments'].items():
        actual = {str(p.relative_to(repo)) for p in (repo / relative).rglob('*') if p.is_file()}
        assert actual == {x['path'] for x in rows}, relative
        for row in rows:
            p = repo / row['path']
            stat = p.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (row['bytes'], row['mtime_ns']), row['path']
            if 'sha256' in row:
                assert sha(p) == row['sha256'], row['path']
    for path, digest in manifest['configs'].items():
        assert sha(repo / path) == digest, path

def run_logged(arguments, filename):
    with (audit / filename).open('x') as log:
        return subprocess.run([sys.executable, '-u', *arguments], cwd=repo, stdout=log, stderr=subprocess.STDOUT).returncode

def source_hashes():
    paths = sorted((repo / 'conformer_rnnt').rglob('*.py'))
    paths += [repo / p for p in ('common.py', 'train.py', 'scripts/server/train_phone69_460h_one.py', 'scripts/server/phone69_epoch_checkpoint.py', 'scripts/server/probe_phone69_training_memory.py', 'scripts/server/validate_phone69_placement.py')]
    return {str(p.relative_to(repo)): sha(p) for p in paths}

try:
    with open('/tmp/phone69_460h_serial.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not config_path.exists() and not root.exists(), 'layer7 output already exists'
        helper['gpu_idle']()
        state('preflight', stage='protecting_existing_runs')
        protected = capture_protected()
        save(audit / 'protected_manifest.json', protected)
        source_before = source_hashes()
        cache = repo / 'output/phone69_460h/audio_samples_cache.json'
        cache_sha256 = json.loads((audit / 'launch_state.json').read_text())['audio_cache_sha256']
        assert cache.is_file() and sha(cache) == cache_sha256, 'Existing audio cache changed before preflight'
        base = yaml.safe_load(base_path.read_text())
        assert base['model']['sequence_mode'] == 'offline'
        assert base['model']['time_reduction_stride'] == 1 and base['model']['time_reduction_input_dim'] == 128
        assert base['model']['encoding_dim'] == 256 and base['model']['num_heads'] == 8 and base['model']['ffn_dim'] == 512
        assert base['model']['num_layers'] == 10 and base['training']['epochs'] == 50 and not base['training'].get('early_stopping')
        assert base['paths']['from_checkpoint'] is None and base['seed'] == 42
        assert base['data']['train_subset'] == ['train-clean-100', 'train-clean-360'] and base['data']['val_subset'] == 'dev-clean'
        assert base['training']['learning_rate'] == 0.0005 and base['training']['warmup_steps'] == 10000
        assert len(base['model']['compression']['stages']) == 1 and base['model']['compression']['stages'][0]['after_layer'] == 5
        candidate = copy.deepcopy(base)
        candidate['paths']['exp_dir'] = str(root)
        candidate['model']['compression']['stages'][0]['after_layer'] = 7
        candidate['data']['max_token_limit'] = 6500
        stage = candidate['model']['compression']['stages'][0]
        assert stage['detector']['type'] == 'cosine_diff' and stage['mixer']['type'] == 'learned'
        assert stage['weight_predictor_mode'] == 'five_frames' and stage['window_size'] == 5
        assert 'keep_rate' not in candidate['model']['compression'] and 'right_context_length' not in candidate['model']
        first_config = audit / 'config_layer7_b6500.yaml'
        first_config.write_text(yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True))
        state('preflight', stage='regression_tests')
        assert run_logged(['-m', 'unittest', 'discover', '-s', 'tests'], 'tests.log') == 0, 'Regression tests failed'
        state('preflight', stage='synthetic_rnnt')
        assert run_logged(['scripts/server/validate_phone69_placement.py', '--configs', str(first_config), '--out', str(audit / 'synthetic_validation.json')], 'synthetic_validation.log') == 0, 'Synthetic RNNT validation failed'
        synth = json.loads((audit / 'synthetic_validation.json').read_text())
        assert synth['valid'] and len(synth['models']) == 1
        accepted = None
        for budget in (6500, 5000, 4000, 3000, 2000):
            helper['gpu_idle']()
            assert sha(cache) == cache_sha256, 'Existing audio cache changed during preflight'
            check_protected(protected)
            assert source_before == source_hashes(), 'Source changed during preflight'
            config = copy.deepcopy(candidate)
            config['data']['max_token_limit'] = budget
            candidate_path = audit / ('config_layer7_b' + str(budget) + '.yaml')
            if budget != 6500:
                candidate_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            state('preflight', stage='real_batch_memory', budget=budget)
            report_path = audit / ('probe_layer7_b' + str(budget) + '.json')
            rc = run_logged(['scripts/server/probe_phone69_training_memory.py', str(candidate_path), '--out', str(report_path), '--keep-cache', '--max-reserved-fraction', '0.9'], 'probe_layer7_b' + str(budget) + '.log')
            assert sha(cache) == cache_sha256, 'Probe modified existing audio cache'
            assert report_path.is_file(), 'Probe did not produce a report'
            report = json.loads(report_path.read_text())
            if rc == 0 and report['status'] == 'passed':
                assert report['budget'] == budget and report['config_sha256'] == sha(candidate_path) and report['results']
                peak = max(x['peak_reserved_GiB'] for x in report['results'])
                assert peak <= 0.9 * report['gpu_total_GiB']
                accepted = dict(budget=budget, candidate=str(candidate_path), config_sha256=sha(candidate_path), probe=str(report_path), peak_reserved_GiB=peak, gpu_total_GiB=report['gpu_total_GiB'])
                break
            reducible = report.get('status') in ('oom', 'unsafe_joint') or 'reserved-memory safety margin' in report.get('error', '')
            assert reducible, 'Probe failed for a reason other than OOM or a configured safety boundary: ' + str(report_path)
        assert accepted is not None, 'No allowed budget passed the memory preflight'
        helper['gpu_idle']()
        check_protected(protected)
        assert source_before == source_hashes(), 'Source changed after preflight'
        assert not config_path.exists() and not root.exists(), 'layer7 output appeared during preflight'
        chosen = Path(accepted['candidate']).read_bytes()
        with config_path.open('xb') as f:
            f.write(chosen)
        assert sha(config_path) == accepted['config_sha256']
        config = yaml.safe_load(config_path.read_text())
        definition = helper['expected_definition'](config)
        assert definition['version'] == 2 and definition['stride'] == 1 and definition['num_layers'] == 10
        assert definition['layers'] == [7, 3] and len(definition['compression_stages']) == 1
        desc = definition['compression_stages'][0]
        assert desc['after_layer'] == 7 and desc['feature_dim'] == 128 and desc['input_domain'] == 'encoder'
        assert desc['detector']['type'] == 'cosine_diff' and desc['mixer']['type'] == 'learned'
        assert desc['weight_predictor_mode'] == 'five_frames' and desc['window_size'] == 5
        gate = dict(valid=True, server_utc=now(), run_phase='placement_layer7_stride1_20261027', config=str(config_path), experiment=str(root), accepted=accepted, offline_native_definition=definition, source_sha256=source_before, protected_artifacts_unchanged=True, regression_tests='passed', synthetic_rnnt='passed')
        save(audit / 'prelaunch_validation.json', gate)
        helper['gpu_idle']()
        check_protected(protected)
        assert not root.exists() and sha(config_path) == accepted['config_sha256'] and source_before == source_hashes()
        (root / 'logs').mkdir(parents=True, exist_ok=False)
        state('training_starting', budget=accepted['budget'], config=str(config_path), config_sha256=sha(config_path))
        with (root / 'logs/train.log').open('x') as log, (root / 'logs/gpu_usage.csv').open('x') as gpu_log:
            env = dict(os.environ, RESUME_LAST='0')
            child = subprocess.Popen([sys.executable, '-u', 'scripts/server/train_phone69_460h_one.py', str(config_path)], cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state('training', budget=accepted['budget'], config=str(config_path), config_sha256=sha(config_path), training_pid=child.pid)
            while child.poll() is None:
                subprocess.run(['nvidia-smi', '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader', '-i', '0'], stdout=gpu_log, stderr=subprocess.DEVNULL)
                gpu_log.flush()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
        assert child.returncode == 0, 'Training failed; logs and checkpoints preserved'
        run = json.loads((root / 'run.json').read_text())
        assert run['config_sha256'] == sha(config_path) and run['resume_checkpoint'] is None
        assert (root / 'best_checkpoint.txt').read_text().strip() == run['best_checkpoint']
        result = helper['check_complete']({'scheme':'layer7', 'experiment':str(root.relative_to(repo))}, definition)
        check_protected(protected)
        save(audit / 'complete_layer7.json', result)
        (root / 'TRAINING_COMPLETE').touch(exist_ok=False)
        save(audit / 'completion_validation.json', dict(valid=True, server_utc=now(), model=result, protected_artifacts_unchanged=True))
        (audit / 'TRAINING_COMPLETE').touch(exist_ok=False)
        state('complete', budget=accepted['budget'], completed=result, completion_validation=str(audit / 'completion_validation.json'))
except BaseException as exc:
    try:
        state('failed', error=repr(exc), message='Preserved logs and checkpoints; no automatic restart')
    except Exception:
        pass
    raise
