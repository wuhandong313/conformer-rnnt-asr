#!/usr/bin/env python3
"""Server-clock status probe for the new offline natural-compression queue."""
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

repo = Path('/test21/conformer-rnnt-asr')
audit = repo / 'artifacts/runs/phone69_460h_native_offline_20261018'
now = time.time()


def file_info(path, tail=0):
    if not path.is_file():
        return None
    stat = path.stat()
    record = dict(path=str(path), bytes=stat.st_size, mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), age_seconds=now-stat.st_mtime)
    if path.suffix == '.json':
        try:
            record['data'] = json.loads(path.read_text())
        except json.JSONDecodeError:
            record['parse_error'] = True
    if tail:
        with path.open('rb') as stream:
            stream.seek(max(0, stat.st_size-20000))
            record['tail'] = stream.read().decode(errors='replace').replace('\r', '\n').splitlines()[-tail:]
    return record


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return dict(returncode=result.returncode, stdout=result.stdout[-10000:], stderr=result.stderr[-1000:])


experiments = []
for scheme in ('stride4_none', 'stride1_cosd_learned', 'stride2_cosd_learned', 'stride4_cosd_learned'):
    path = repo / f'exp_phone69_460h_offline_native_{scheme}_s42_e50'
    run = file_info(path/'run.json')
    data = (run or {}).get('data', {})
    pid = data.get('pid')
    checkpoints = sorted((path/'checkpoints').glob('*.ckpt'), key=lambda p:p.stat().st_mtime)
    experiments.append(dict(
        scheme=scheme, exists=path.exists(), run=run,
        main_process=command(['ps', '-p', str(pid), '-o', 'pid,ppid,stat,etimes,args']) if pid else None,
        log=file_info(path/'logs/train.log', 8), gpu_log=file_info(path/'logs/gpu_usage.csv', 2),
        checkpoints=[file_info(p) for p in checkpoints[-6:]],
        training_complete=(path/'TRAINING_COMPLETE').is_file(),
        best_checkpoint_exists=Path(data['best_checkpoint']).is_file() if data.get('best_checkpoint') else None,
    ))
print(json.dumps(dict(
    server_utc=datetime.now(timezone.utc).isoformat(),
    run_phase='offline_native_20261018', queue=file_info(audit/'queue_state.json'),
    preflight=file_info(audit/'preflight_state.json'), gate=file_info(audit/'prelaunch_validation.json'),
    pipeline_log=file_info(repo/'artifacts/logs/phone69_460h_offline_native_pipeline.log', 10),
    experiments=experiments,
    probes={p.name:file_info(p) for p in sorted(audit.glob('probe_*.json'))},
    processes=command(['pgrep','-af','(train_phone69|probe_phone69_training_memory|phone69_460h_offline_native_pipeline)']),
    gpu=command(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader']),
    tmux=command(['tmux','list-sessions']),
    queue_complete=(audit/'TRAINING_QUEUE_COMPLETE').is_file(),
), indent=2))
