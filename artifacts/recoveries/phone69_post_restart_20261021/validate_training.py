import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import torch
import yaml

ROOT = Path('/test21/conformer-rnnt-asr')
AUDIT = ROOT / 'artifacts/runs/phone69_460h_native_offline_five_frames_20261018'
QUEUE = AUDIT / 'queue_state.json'
SCHEMES = ['stride1_cosd_learned', 'stride2_cosd_learned', 'stride4_cosd_learned']
BASE = 'stride4_none'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def finite_tree(value):
    if torch.is_tensor(value):
        return (not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all()))
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return True

def verify_baseline():
    before = json.loads((AUDIT / 'baseline_preserved_before.json').read_text())
    after = json.loads((AUDIT / 'baseline_preserved_after.json').read_text())
    assert after.get('valid') is True and after.get('all_sha256_size_mtime_unchanged') is True
    rows = before.get('files') or []
    assert rows, 'baseline manifest has no file fingerprints'
    for row in rows:
        path = ROOT / row['path']
        st = path.stat()
        assert st.st_size == row['bytes'], f'baseline size changed: {row["path"]}'
        assert st.st_mtime_ns == row['mtime_ns'], f'baseline mtime changed: {row["path"]}'
        assert digest(path) == row['sha256'], f'baseline hash changed: {row["path"]}'
    cfg = ROOT / 'configs/server_phone69_460h_offline_native_stride4_none_s42_e50.yaml'
    run = json.loads((ROOT / 'exp_phone69_460h_offline_native_stride4_none_s42_e50/run.json').read_text())
    assert digest(cfg) == before['config_sha256'] == run['config_sha256']
    return {'files_fingerprinted': len(rows), 'bytes': sum(row['bytes'] for row in rows), 'manifest_unchanged': True}

def verify_model(scheme, experiment, config_rel, expected_epochs=50):
    cfg_path = ROOT / config_rel
    exp_path = ROOT / experiment
    run = json.loads((exp_path / 'run.json').read_text())
    assert run['status'] == 'trained' and run['epochs'] == expected_epochs and run['epoch'] == expected_epochs
    assert run['global_step'] > 0 and run['test_subset'] == 'test-clean'
    assert run['tokenizer'] == 'phone69' and run['sequence_mode'] == 'offline'
    assert (exp_path / 'TRAINING_COMPLETE').is_file()
    best = Path(run['best_checkpoint'])
    assert best.is_file()
    best_pointer = (exp_path / 'best_checkpoint.txt').read_text().strip()
    assert best_pointer == str(best)
    boundary = Path(run['epoch_boundary_checkpoint'])
    assert boundary.is_file()
    if scheme != BASE:
        comp = (yaml.safe_load(cfg_path.read_text())['model'].get('compression') or {})
        assert comp.get('weight_predictor_mode') == 'five_frames'
    cfg = yaml.safe_load(cfg_path.read_text())
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    model = LibriSpeechRNNTModule.from_config(cfg)
    expected_definition = model.offline_native_definition
    assert expected_definition is not None
    if scheme != BASE:
        assert expected_definition.get('weight_predictor_mode') == 'five_frames'
    model_state = model.state_dict()
    files = []
    for kind, path in [('best', best), ('boundary', boundary)]:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        assert isinstance(ckpt.get('state_dict'), dict) and ckpt['state_dict']
        state = ckpt['state_dict']
        assert state.keys() == model_state.keys(), f'{scheme}/{kind}: model key set mismatch'
        for key, tensor in state.items():
            assert torch.is_tensor(tensor), f'{scheme}/{kind}: non-tensor model state {key}'
            assert tuple(tensor.shape) == tuple(model_state[key].shape), f'{scheme}/{kind}: shape mismatch {key}'
            assert finite_tree(tensor), f'{scheme}/{kind}: non-finite model tensor {key}'
        assert ckpt.get('offline_native_definition') == expected_definition, f'{scheme}/{kind}: native definition mismatch'
        assert ckpt.get('phone69_tokenizer', {}).get('fingerprint') == model.tokenizer.fingerprint
        optimizers = ckpt.get('optimizer_states')
        schedulers = ckpt.get('lr_schedulers')
        assert isinstance(optimizers, list) and optimizers and optimizers[0].get('state'), f'{scheme}/{kind}: missing optimizer state'
        assert isinstance(schedulers, list) and schedulers and schedulers[0], f'{scheme}/{kind}: missing scheduler state'
        assert finite_tree(optimizers), f'{scheme}/{kind}: non-finite optimizer state'
        if kind == 'boundary':
            meta = ckpt.get('phone69_epoch_boundary') or {}
            assert meta.get('resume_at_epoch_boundary') is True
            assert meta.get('resume_completed_epochs') == expected_epochs
            assert ckpt.get('epoch') == expected_epochs
        files.append({'kind': kind, 'path': str(path), 'bytes': path.stat().st_size,
                      'model_tensors': len(state), 'optimizer_states': len(optimizers),
                      'scheduler_states': len(schedulers), 'definition': expected_definition})
        del ckpt
    return {'scheme': scheme, 'status': run['status'], 'epoch': run['epoch'],
            'step': run['global_step'], 'best_checkpoint': str(best),
            'boundary_checkpoint': str(boundary), 'checkpoints': files}

os.chdir(ROOT)
queue = json.loads(QUEUE.read_text())
assert queue['status'] == 'complete' and not queue.get('error')
active = subprocess.run(['pgrep', '-af', '^/root/anaconda3/envs/whd/bin/python -u scripts/server/train_phone69_460h_one.py'], capture_output=True, text=True)
assert active.returncode == 1, f'training process remains: {active.stdout.strip()}'
baseline = verify_baseline()
models = [verify_model(BASE, 'exp_phone69_460h_offline_native_stride4_none_s42_e50',
                       'configs/server_phone69_460h_offline_native_stride4_none_s42_e50.yaml')]
for scheme in SCHEMES:
    models.append(verify_model(scheme,
        f'exp_phone69_460h_offline_native_{scheme}_five_frames_s42_e50',
        f'configs/server_phone69_460h_offline_native_{scheme}_s42_e50.yaml'))
definitions = [row['checkpoints'][0]['definition'] for row in models[1:]]
for definition in definitions:
    common = {k:v for k,v in definition.items() if k not in {'stride','input_projection'}}
    assert common == {k:v for k,v in definitions[0].items() if k not in {'stride','input_projection'}}
print(json.dumps({'server_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                  'queue_complete': True, 'active_training_processes': 0,
                  'baseline': baseline, 'models': models,
                  'five_frames_definitions_consistent': True}, ensure_ascii=False))
