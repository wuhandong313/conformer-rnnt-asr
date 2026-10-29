#!/usr/bin/env python3
"""Read-only, server-clock status for the approved placement training queue."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

REPO = Path('/test21/conformer-rnnt-asr')
AUDIT = REPO / 'artifacts/runs/phone69_460h_placement_stride1_20261022'
NOW = time.time()


def file_info(path, lines=0):
    if not path.is_file():
        return None
    stat = path.stat()
    value = dict(path=str(path), bytes=stat.st_size,
                 mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                 age_seconds=NOW - stat.st_mtime)
    if path.suffix == '.json':
        try:
            value['data'] = json.loads(path.read_text())
        except (ValueError, OSError):
            value['parse_error'] = True
    if lines:
        with path.open('rb') as stream:
            stream.seek(max(0, stat.st_size - 16000))
            value['tail'] = stream.read().decode(errors='replace').replace('\r', '\n').splitlines()[-lines:]
    return value


def command(args):
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
        return dict(returncode=p.returncode, stdout=p.stdout[-6000:], stderr=p.stderr[-500:])
    except OSError as error:
        return dict(error=str(error))


def main():
    plan = json.loads((AUDIT / 'queue_plan.json').read_text())
    experiments = []
    for item in plan['experiments']:
        root = REPO / item['experiment']
        checkpoints = sorted((root / 'checkpoints').glob('*.ckpt'), key=lambda p: p.stat().st_mtime)
        experiments.append(dict(**item, run=file_info(root / 'run.json'),
                                training_complete=(root / 'TRAINING_COMPLETE').is_file(),
                                log=file_info(root / 'logs/train.log', 8),
                                gpu_log=file_info(root / 'logs/gpu_usage.csv', 2),
                                latest_checkpoint=file_info(checkpoints[-1]) if checkpoints else None,
                                completion_validation=file_info(AUDIT / ('complete_' + item['scheme'] + '.json'))))
    probes = sorted(AUDIT.glob('probe_*.json'), key=lambda p: p.stat().st_mtime)
    queue = file_info(AUDIT / 'queue_state.json')
    pipeline_pid = (queue or {}).get('data', {}).get('pid')
    processes = command(['ps', '-eo', 'pid,ppid,etime,args'])
    processes['stdout'] = '\n'.join(line for line in processes.get('stdout', '').splitlines()
                                     if any(x in line for x in ('phone69_placement_pipeline.py', 'train_phone69_460h_one.py', 'probe_phone69_training_memory.py')))
    print(json.dumps(dict(server_utc=datetime.now(timezone.utc).isoformat(), run_phase=plan['run_phase'],
                          queue=queue, launch=file_info(AUDIT / 'launch_state.json'),
                          gate=file_info(AUDIT / 'prelaunch_validation.json'),
                          latest_probe=file_info(probes[-1]) if probes else None,
                          pipeline_log=file_info(AUDIT / 'pipeline.log', 10), experiments=experiments,
                          pipeline_process=command(['ps', '-p', str(pipeline_pid), '-o', 'pid,ppid,stat,etime,args']) if pipeline_pid else None,
                          processes=processes,
                          gpu=command(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader']),
                          queue_complete=(AUDIT / 'TRAINING_QUEUE_COMPLETE').is_file()), ensure_ascii=False))


if __name__ == '__main__':
    main()
