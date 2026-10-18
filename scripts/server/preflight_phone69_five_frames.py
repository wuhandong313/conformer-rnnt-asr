#!/usr/bin/env python3
"""Validate the approved recipes without changing budgets or training state."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / 'artifacts/runs/phone69_460h_native_offline_five_frames_20261018'
REPORT = 'docs/phone69_460h_five_frames_computation_20261018.md'


def save(name, data):
    path = AUDIT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.chdir(REPO)
    os.environ.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    plan = json.loads((AUDIT / 'queue_plan.json').read_text())
    state = dict(status='validating', pid=os.getpid(), run_phase=plan['run_phase'])

    def progress(**updates):
        state.update(updates, server_utc=datetime.now(timezone.utc).isoformat())
        save('preflight_state.json', state)

    def run(args, log):
        progress(command=args, log=log)
        with (AUDIT / log).open('x') as stream:
            result = subprocess.run([sys.executable, '-u', *args], stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{args[0]} failed ({result.returncode}); see {log}')

    progress(status='checking_config_scope')
    import yaml

    for item in plan['experiments']:
        old = yaml.safe_load((AUDIT / 'obsolete_diff5_作废' / 'configs' / Path(item['config']).name).read_text())
        new = yaml.safe_load((REPO / item['config']).read_text())
        expected = dict(old)
        expected['paths'] = dict(old['paths'], exp_dir=str(REPO / item['experiment']))
        expected['model'] = dict(old['model'], compression=dict(old['model']['compression'], weight_predictor_mode='five_frames'))
        assert expected == new, item['config'] + ' contains an unapproved config change'
        assert new['data']['max_token_limit'] == item['budget']
    save('config_scope_validation.json', dict(valid=True, changed_fields=[
        'paths.exp_dir', 'model.compression.weight_predictor_mode'], models=plan['experiments']))
    try:
        run(['-m','unittest','discover','-s','tests'], 'preflight_tests.log')
        run(['scripts/diagnostics/check_phone69_offline_native.py','--out',
             str(AUDIT / 'dimension_audit.json')], 'dimension_audit.log')
        progress(status='real_batch_memory_checks')
        accepted = []
        for item in plan['experiments']:
            stem = 'probe_' + item['scheme'] + '_b' + str(item['budget'])
            run(['scripts/server/probe_phone69_training_memory.py', item['config'],
                 '--out',str(AUDIT / (stem + '.json')), '--keep-cache',
                 '--max-reserved-fraction','0.90'], stem + '.log')
            report = json.loads((AUDIT / (stem + '.json')).read_text())
            assert report['status'] == 'passed' and report['results']
            assert report['config_sha256'] == sha(REPO / item['config'])
            assert report['budget'] == item['budget']
            accepted.append(dict(scheme=item['scheme'], budget=item['budget'],
                probe=str((AUDIT / (stem + '.json')).relative_to(REPO)),
                peak_reserved_GiB=max(x['peak_reserved_GiB'] for x in report['results']),
                peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in report['results']),
                cases=len(report['results'])))
            progress(accepted=accepted)
        dimensions = json.loads((AUDIT / 'dimension_audit.json').read_text())['models']
        models = []
        for item in plan['experiments']:
            row = next(x for x in dimensions if x['scheme'] == item['scheme'])
            assert row['weight_predictor_mode'] == 'five_frames'
            assert row['compression_parameters'] == 20677
            row.update(item)
            row.update(next(x for x in accepted if x['scheme'] == item['scheme']))
            models.append(row)
        save('memory_validation.json', dict(valid=True, models=models,
             baseline='not rerun; original trained files retained', budget_changed=False))
        sources = list((REPO / 'conformer_rnnt').rglob('*.py')) + [REPO / x for x in (
            'common.py','train.py','scripts/server/train_phone69_460h_one.py',
            'scripts/server/phone69_epoch_checkpoint.py',
            'scripts/server/phone69_460h_five_frames_pipeline.sh',
            'scripts/server/validate_phone69_five_frames_launch.py',
            'scripts/server/status_phone69_460h_five_frames.py',
            'scripts/server/preflight_phone69_five_frames.py',
            'scripts/diagnostics/check_phone69_offline_native.py')]
        gate = dict(valid=True, approved_audit_item='A05', models=models,
            source_sha256={str(p.relative_to(REPO)):sha(p) for p in sources},
            report=REPORT, report_sha256=sha(REPO / REPORT),
            server_utc=datetime.now(timezone.utc).isoformat(),
            baseline_manifest_sha256=sha(AUDIT / 'baseline_preserved_before.json'),
            training_model_code_changed=False)
        save('prelaunch_validation.json', gate)
        run(['scripts/server/validate_phone69_five_frames_launch.py'], 'launch_gate_validation.log')
        progress(status='passed', accepted=accepted)
    except BaseException as error:
        progress(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
