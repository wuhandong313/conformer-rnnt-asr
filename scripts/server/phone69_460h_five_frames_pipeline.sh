#!/usr/bin/env bash
# Three fresh five-frame dynamic models; the completed baseline is read-only.
set -euo pipefail
cd /test21/conformer-rnnt-asr
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python_bin=/root/anaconda3/envs/whd/bin/python
audit=artifacts/runs/phone69_460h_native_offline_five_frames_20261018
exec 9>/tmp/phone69_460h_serial.lock
flock -n 9 || { echo 'Another phone69 pipeline holds the lock'; exit 75; }
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
state() {
  "$python_bin" - "$audit/queue_state.json" "$1" "${2:-}" "${3:-}" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]);t=p.with_suffix('.tmp')
t.write_text(json.dumps(dict(status=sys.argv[2],experiment=sys.argv[3],error=sys.argv[4],
    pid=os.getppid(),run_phase='offline_native_five_frames_20261018',
    server_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
t.replace(p)
PY
}
"$python_bin" scripts/server/validate_phone69_five_frames_launch.py
schemes=(stride1_cosd_learned stride2_cosd_learned stride4_cosd_learned)
if [[ "${PHONE69_FRESH_LAUNCH:-0}" == 1 ]]; then
  for scheme in "${schemes[@]}"; do
    exp_dir="exp_phone69_460h_offline_native_${scheme}_five_frames_s42_e50"
    if [[ -e "$exp_dir/run.json" || -e "$exp_dir/TRAINING_COMPLETE" ]] || compgen -G "$exp_dir/checkpoints/*.ckpt" >/dev/null; then
      log "Fresh launch rejected: existing state in $exp_dir"
      exit 2
    fi
  done
fi
train_pid='' monitor_pid='' exp_dir=''
stop() {
  [[ -z "$train_pid" ]] || kill -INT "$train_pid" 2>/dev/null || true
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
  state interrupted "$exp_dir" 'Pipeline interrupted'
  exit 130
}
trap stop INT TERM
for scheme in "${schemes[@]}"; do
  exp_dir="exp_phone69_460h_offline_native_${scheme}_five_frames_s42_e50"
  config="configs/server_phone69_460h_offline_native_${scheme}_s42_e50.yaml"
  if [[ -f "$exp_dir/TRAINING_COMPLETE" ]]; then
    "$python_bin" - "$exp_dir/run.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['status']=='trained' and r['epoch']==50
assert r['compression']['weight_predictor_mode']=='five_frames'
assert Path(r['best_checkpoint']).is_file()
PY
    log "Already completed: $exp_dir"
    continue
  fi
  mkdir -p "$exp_dir/logs" "$exp_dir/checkpoints"
  state training "$exp_dir"
  log "Starting/resuming $exp_dir"
  RESUME_LAST=1 "$python_bin" -u scripts/server/train_phone69_460h_one.py "$config" 9>&- >> "$exp_dir/logs/train.log" 2>&1 &
  train_pid=$!
  (
    exec 9>&-
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
log 'All three five_frames runs completed; original trained baseline preserved'
