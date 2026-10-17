#!/usr/bin/env bash
# Train the dimension-matched stride2 comparison in an independent tmux session.
set -Eeuo pipefail

REPO=/test21/conformer-rnnt-asr
PYTHON="${PYTHON:-/root/anaconda3/envs/whd/bin/python}"
CONFIG=configs/server_phone69_460h_stride2_cosd_learned_s42_e50.yaml
MASTER_LOG="$REPO/artifacts/logs/phone69_460h_stride2_pipeline.log"
cd "$REPO"
mkdir -p artifacts/logs
[[ -n "${TMUX:-}" ]] || { echo 'Run inside tmux.' >&2; exit 2; }

# Share the original queue's lock without relaunching that completed queue.
exec 9>/tmp/phone69_460h_serial.lock
flock -n 9 || { echo 'Another phone69 training pipeline holds the lock.' >&2; exit 2; }
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$MASTER_LOG"; }
exp_dir=$("$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["paths"]["exp_dir"])' "$CONFIG")
mkdir -p "$exp_dir/logs"
if [[ -f "$exp_dir/TRAINING_COMPLETE" ]]; then
  log "already complete: $CONFIG"
  exit 0
fi
if [[ -f "$exp_dir/config.yaml" ]]; then
  cmp -s "$CONFIG" "$exp_dir/config.yaml" || { log 'ERROR saved config differs'; exit 2; }
else
  cp "$CONFIG" "$exp_dir/config.yaml"
  sha256sum "$CONFIG" > "$exp_dir/config.sha256"
fi
while true; do
  used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -n 1 | tr -d ' ')
  [[ "$used_mib" -lt 2000 ]] && break
  log "GPU in use (${used_mib} MiB); waiting 60 seconds"
  sleep 60
done

log "training $CONFIG"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  RESUME_LAST=1 "$PYTHON" -u scripts/server/train_phone69_460h_one.py "$CONFIG" \
  >> "$exp_dir/logs/train.log" 2>&1 &
train_pid=$!
(
  while kill -0 "$train_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader -i 0 >> "$exp_dir/logs/gpu_usage.csv"
    sleep 60
  done
) &
monitor_pid=$!
set +e
wait "$train_pid"
train_rc=$?
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
if [[ "$train_rc" -ne 0 ]]; then
  log "ERROR training failed (exit=$train_rc)"
  exit "$train_rc"
fi
touch "$exp_dir/TRAINING_COMPLETE"
log 'stride2 training completed'
