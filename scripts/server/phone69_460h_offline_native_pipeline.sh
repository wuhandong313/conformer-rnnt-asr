#!/usr/bin/env bash
# Four fresh offline runs; resume only this new definition after interruptions.
set -euo pipefail
cd /test21/conformer-rnnt-asr
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python_bin=/root/anaconda3/envs/whd/bin/python
audit=artifacts/runs/phone69_460h_native_offline_20261018
mkdir -p "$audit" artifacts/logs
exec 9>/tmp/phone69_460h_serial.lock
flock -n 9 || { echo 'Another phone69 training pipeline holds the lock'; exit 75; }
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
state() {
  "$python_bin" - "$audit/queue_state.json" "$1" "${2:-}" "${3:-}" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_suffix('.tmp')
t.write_text(json.dumps(dict(status=sys.argv[2], experiment=sys.argv[3], error=sys.argv[4],
                            pid=os.getppid(), server_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
t.replace(p)
PY
}
"$python_bin" - "$audit/prelaunch_validation.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
gate=json.loads(Path(sys.argv[1]).read_text())
assert gate['valid'] is True
for record in gate['models']:
    p=Path(record['config'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==record['config_sha256'], str(p)+' changed after validation'
for name, expected in gate['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, name+' changed after validation'
assert hashlib.sha256(Path(gate['report']).read_bytes()).hexdigest()==gate['report_sha256']
PY
schemes=(stride4_none stride1_cosd_learned stride2_cosd_learned stride4_cosd_learned)
if [[ "${PHONE69_FRESH_LAUNCH:-0}" == 1 ]]; then
  for scheme in "${schemes[@]}"; do
    exp_dir="exp_phone69_460h_offline_native_${scheme}_s42_e50"
    if compgen -G "$exp_dir/checkpoints/*.ckpt" >/dev/null || [[ -e "$exp_dir/TRAINING_COMPLETE" ]]; then
      log "Fresh launch rejected: existing checkpoint/completion in $exp_dir"
      exit 2
    fi
  done
fi
train_pid='' monitor_pid=''
stop() {
  [[ -z "$train_pid" ]] || kill -INT "$train_pid" 2>/dev/null || true
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
  state interrupted "${exp_dir:-}" 'Pipeline interrupted'
  exit 130
}
trap stop INT TERM
for scheme in "${schemes[@]}"; do
  exp_dir="exp_phone69_460h_offline_native_${scheme}_s42_e50"
  config="configs/server_phone69_460h_offline_native_${scheme}_s42_e50.yaml"
  if [[ -f "$exp_dir/TRAINING_COMPLETE" ]]; then
    "$python_bin" - "$exp_dir" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); run=json.loads((p/'run.json').read_text())
assert run['status']=='trained' and run['epoch']==50
assert Path(run['best_checkpoint']).is_file()
PY
    log "Already completed: $exp_dir"
    continue
  fi
  mkdir -p "$exp_dir/logs" "$exp_dir/checkpoints"
  state training "$exp_dir"
  log "Starting/resuming $exp_dir"
  RESUME_LAST=1 "$python_bin" -u scripts/server/train_phone69_460h_one.py "$config" >> "$exp_dir/logs/train.log" 2>&1 &
  train_pid=$!
  (
    while kill -0 "$train_pid" 2>/dev/null; do
      nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader -i 0 >> "$exp_dir/logs/gpu_usage.csv" || true
      sleep 60
    done
  ) &
  monitor_pid=$!
  set +e
  wait "$train_pid"
  result=$?
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  train_pid='' monitor_pid=''
  if [[ "$result" != 0 ]]; then
    state failed "$exp_dir" "Training exited with code $result"
    log "ERROR $exp_dir exited $result; retained logs/checkpoints"
    exit "$result"
  fi
  touch "$exp_dir/TRAINING_COMPLETE"
  state experiment_complete "$exp_dir"
  log "Completed $exp_dir"
done
state complete
touch "$audit/TRAINING_QUEUE_COMPLETE"
log 'All four offline natural-compression runs completed'
