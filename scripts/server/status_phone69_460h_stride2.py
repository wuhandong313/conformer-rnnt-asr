from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import time

repo = Path('/test21/conformer-rnnt-asr')
experiment = repo / 'exp_phone69_460h_stride2_cosd_learned_s42_e50'
audit = repo / 'artifacts/runs/phone69_460h_stride2'
now = time.time()

def inspect(path, tail=0):
    if not path.is_file():
        return None
    row = dict(bytes=path.stat().st_size, age_seconds=now-path.stat().st_mtime)
    if path.suffix == '.json':
        try:
            row['data'] = json.loads(path.read_text())
        except json.JSONDecodeError:
            row['parse_error'] = True
    if tail:
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 24000))
            row['tail'] = f.read().decode(errors='replace').replace('\r', '\n').splitlines()[-tail:]
    return row

def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return dict(returncode=p.returncode, stdout=p.stdout[-16000:], stderr=p.stderr[-1000:])

run_info = inspect(experiment / 'run.json')
run_data = (run_info or {}).get('data', {})
best_path = run_data.get('best_checkpoint')
best_marker = experiment / 'best_checkpoint.txt'

print(json.dumps(dict(
    server_utc=datetime.now(timezone.utc).isoformat(),
    dimension_audit=inspect(audit / 'dimension_audit.json'),
    probes={p.name: inspect(p) for p in sorted(audit.glob('probe_*.json'))},
    probe_logs={p.name: inspect(p, 8) for p in sorted(audit.glob('probe_*.log'))},
    run=run_info,
    main_process=command(['ps', '-p', str(run_data.get('pid', 0)), '-o', 'pid,ppid,stat,etimes,args']),
    training_log=inspect(experiment / 'logs/train.log', 10),
    gpu_log=inspect(experiment / 'logs/gpu_usage.csv', 3),
    checkpoints=[dict(name=p.name, **inspect(p)) for p in sorted((experiment / 'checkpoints').glob('*.ckpt'), key=lambda p:p.stat().st_mtime)[-5:]],
    training_complete=(experiment / 'TRAINING_COMPLETE').is_file(),
    best_checkpoint=dict(path=best_path, metadata=inspect(Path(best_path))) if best_path else None,
    best_checkpoint_marker=best_marker.read_text().strip() if best_marker.is_file() else None,
    gpu=command(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader']),
    gpu_processes=command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader']),
    processes=command(['pgrep', '-af', '(train_phone69|eval.py|probe_phone69_training_memory)']),
    tmux=command(['tmux', 'list-sessions']),
), indent=2))
