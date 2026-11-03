"""Read-only status of the approved 128-dimensional feature-filter run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/test21/conformer-rnnt-asr')
AUDIT = ROOT / 'artifacts/runs/phone69_460h_feature_smooth_20261102'
EXP = ROOT / 'exp_phone69_460h_offline_native_stride1_cosd_learned_feature_smooth_a025_s42_e50'
CONFIG = ROOT / 'configs/server_phone69_460h_offline_native_stride1_cosd_learned_feature_smooth_a025_s42_e50.yaml'
now = time.time()


def info(path, tail=0):
    if not path.is_file():
        return None
    stat = path.stat()
    row = {'path': str(path), 'bytes': stat.st_size, 'mtime_utc': datetime.fromtimestamp(
        stat.st_mtime, timezone.utc).isoformat(), 'age_seconds': now - stat.st_mtime}
    if path.suffix == '.json':
        try:
            row['data'] = json.loads(path.read_text())
        except ValueError:
            row['parse_error'] = True
    if tail:
        with path.open('rb') as stream:
            stream.seek(max(0, stat.st_size - 20000))
            row['tail'] = stream.read().decode(errors='replace').replace('\r', '\n').splitlines()[-tail:]
    return row


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=20)
    return {'returncode': p.returncode, 'stdout': p.stdout[-8000:], 'stderr': p.stderr[-1000:]}


queue = info(AUDIT / 'queue_state.json')
run = info(EXP / 'run.json')
q, r = (queue or {}).get('data', {}), (run or {}).get('data', {})
pids = {x for x in [q.get('pipeline_pid'), q.get('training_pid'), q.get('preflight_pid'), r.get('pid')] if x}
checkpoints = sorted((EXP / 'checkpoints').glob('*.ckpt'), key=lambda p: p.stat().st_mtime)
gate = info(AUDIT / 'prelaunch_validation.json')
expected = ((gate or {}).get('data') or {}).get('config_sha256')
actual = hashlib.sha256(CONFIG.read_bytes()).hexdigest() if CONFIG.is_file() else None
print(json.dumps({'server_utc': datetime.now(timezone.utc).isoformat(), 'queue': queue, 'run': run,
    'config_sha256': actual, 'config_matches_gate': actual == expected if expected else None,
    'prelaunch_gate': gate, 'memory_probe': info(AUDIT / 'memory_probe.json'),
    'completion_validation': info(AUDIT / 'completion_validation.json'),
    'training_complete': (EXP / 'TRAINING_COMPLETE').is_file(),
    'processes': command(['ps', '-p', ','.join(map(str, sorted(pids))), '-o', 'pid,ppid,stat,etimes,args']) if pids else None,
    'training_log': info(EXP / 'logs/train.log', 12), 'gpu_log': info(EXP / 'logs/gpu_usage.csv', 2),
    'pipeline_log': info(AUDIT / 'pipeline.log', 8),
    'preflight_log': info(AUDIT / q.get('preflight_log', 'tests.log'), 10),
    'checkpoints': [info(p) for p in checkpoints[-3:]],
    'gpu': command(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader']),
    'gpu_processes': command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])
}, ensure_ascii=False, indent=2))
