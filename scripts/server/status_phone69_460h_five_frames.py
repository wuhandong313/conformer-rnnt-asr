#!/usr/bin/env python3
"""Read-only server-clock status for the three approved five_frames runs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

REPO = Path('/test21/conformer-rnnt-asr')
AUDIT = REPO / 'artifacts/runs/phone69_460h_native_offline_five_frames_20261018'
now = time.time()


def file_info(path, tail=0):
    if not path.is_file():
        return None
    stat = path.stat()
    row = dict(path=str(path), bytes=stat.st_size,
               mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
               age_seconds=now - stat.st_mtime)
    if path.suffix == '.json':
        try:
            row['data'] = json.loads(path.read_text())
        except json.JSONDecodeError:
            row['parse_error'] = True
    if tail:
        with path.open('rb') as stream:
            stream.seek(max(0, stat.st_size - 20000))
            row['tail'] = stream.read().decode(errors='replace').replace('\r', '\n').splitlines()[-tail:]
    return row


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return dict(returncode=result.returncode, stdout=result.stdout[-12000:], stderr=result.stderr[-1000:])


plan = json.loads((AUDIT / 'queue_plan.json').read_text())
experiments = []
for item in [plan['preserved_baseline'], *plan['experiments']]:
    root = REPO / item['experiment']
    run = file_info(root / 'run.json')
    data = (run or {}).get('data', {})
    pid = data.get('pid')
    checkpoints = sorted((root / 'checkpoints').glob('*.ckpt'), key=lambda p: p.stat().st_mtime)
    experiments.append(dict(**item, exists=root.exists(), run=run,
        main_process=command(['ps','-p',str(pid),'-o','pid,ppid,stat,etimes,args']) if pid else None,
        log=file_info(root / 'logs/train.log', 8), gpu_log=file_info(root / 'logs/gpu_usage.csv', 2),
        checkpoints=[file_info(p) for p in checkpoints[-4:]],
        training_complete=(root / 'TRAINING_COMPLETE').is_file(),
        best_checkpoint_exists=Path(data['best_checkpoint']).is_file() if data.get('best_checkpoint') else None))
print(json.dumps(dict(server_utc=datetime.now(timezone.utc).isoformat(),
    run_phase=plan['run_phase'], queue=file_info(AUDIT / 'queue_state.json'),
    preflight=file_info(AUDIT / 'preflight_state.json'), maintenance=file_info(AUDIT / 'maintenance_state.json'),
    gate=file_info(AUDIT / 'prelaunch_validation.json'),
    pipeline_log=file_info(REPO / plan['pipeline_log'], 10), experiments=experiments,
    processes=command(['pgrep','-af','(train_phone69|probe_phone69_training_memory|phone69_460h_five_frames|preflight_phone69_five_frames)']),
    gpu=command(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader']),
    tmux=command(['tmux','list-sessions']), queue_complete=(AUDIT / 'TRAINING_QUEUE_COMPLETE').is_file()), indent=2))
