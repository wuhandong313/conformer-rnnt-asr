#!/usr/bin/env python3
"""Validate the approved three-model definition and preserved baseline gate."""
import hashlib
import json
from pathlib import Path
import sys

import yaml

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / 'artifacts/runs/phone69_460h_native_offline_five_frames_20261018'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gate = json.loads((AUDIT / 'prelaunch_validation.json').read_text())
    assert gate['valid'] is True
    assert gate['approved_audit_item'] == 'A05'
    for name, expected in gate['source_sha256'].items():
        assert sha(REPO / name) == expected, name + ' changed after validation'
    assert sha(REPO / gate['report']) == gate['report_sha256']
    approval = json.loads((AUDIT / 'audit_approval_state.json').read_text())
    assert next(x for x in approval['items'] if x['id'] == 'A05')['status'] == 'approved'
    for row in gate['models']:
        config_path = REPO / row['config']
        assert sha(config_path) == row['config_sha256']
        config = yaml.safe_load(config_path.read_text())
        assert config['model']['sequence_mode'] == 'offline'
        assert config['model']['compression'] == dict(
            detector=dict(type='cosine_diff'), mixer=dict(type='learned'),
            weight_predictor_mode='five_frames')
        assert config['paths']['from_checkpoint'] is None
        assert config['paths']['exp_dir'] == str(REPO / row['experiment'])
        assert config['training']['epochs'] == 50 and config['seed'] == 42
        assert config['data']['max_token_limit'] == row['budget']
    baseline = json.loads((AUDIT / 'baseline_preserved_before.json').read_text())
    assert sha(REPO / baseline['config']) == baseline['config_sha256']
    root = Path(baseline['experiment'])
    actual = {str(p.relative_to(REPO)) for p in root.rglob('*') if p.is_file()}
    assert actual == {x['path'] for x in baseline['files']}
    for row in baseline['files']:
        stat = (REPO / row['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (row['bytes'], row['mtime_ns']), row['path']
    run = json.loads((root / 'run.json').read_text())
    assert run['status'] == 'trained' and run['epoch'] == 50
    assert (root / 'TRAINING_COMPLETE').is_file() and Path(run['best_checkpoint']).is_file()
    print('Validated five_frames launch gate; completed baseline files unchanged', flush=True)


if __name__ == '__main__':
    main()
