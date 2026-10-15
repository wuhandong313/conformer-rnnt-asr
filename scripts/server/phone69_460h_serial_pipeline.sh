#!/usr/bin/env bash
# Serial 460h-clean phone69 experiments. Run this script inside the approved tmux session.
set -Eeuo pipefail

REPO=/test21/conformer-rnnt-asr
PYTHON="${PYTHON:-/root/anaconda3/envs/whd/bin/python}"
SESSION_NAME=phone69_460h_serial
MASTER_LOG="$REPO/artifacts/logs/phone69_460h_serial_pipeline.log"
CONFIGS=(
  configs/server_phone69_460h_stride4_none_s42_e50.yaml
  configs/server_phone69_460h_stride1_cosd_learned_s42_e50.yaml
  configs/server_phone69_460h_stride4_cosd_learned_s42_e50.yaml
)

cd "$REPO"
mkdir -p "$REPO/artifacts/logs"

if [[ -z "${TMUX:-}" ]]; then
  echo "Run this pipeline inside tmux session: $SESSION_NAME" >&2
  exit 2
fi

exec 9>"/tmp/${SESSION_NAME}.lock"
if ! flock -n 9; then
  echo "Another $SESSION_NAME pipeline already holds the lock." >&2
  exit 2
fi

log() {
  echo "[$(date -u +%FT%TZ)] $*" | tee -a "$MASTER_LOG"
}

wait_for_gpu() {
  while true; do
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -n 1 | tr -d ' ')
    if [[ "$used_mib" -lt 2000 ]]; then
      return
    fi
    log "GPU in use (${used_mib} MiB); check again in 60 seconds"
    sleep 60
  done
}

log "serial pipeline start; order=${CONFIGS[*]}"

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    log "ERROR config missing: $config"
    exit 1
  fi

  exp_dir=$("$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["paths"]["exp_dir"])' "$config")
  mkdir -p "$exp_dir/logs"
  if [[ -f "$exp_dir/config.yaml" ]] && ! cmp -s "$config" "$exp_dir/config.yaml"; then
    log "ERROR config differs from the saved experiment config: $exp_dir/config.yaml"
    exit 1
  fi
  cp -f "$config" "$exp_dir/config.yaml"
  sha256sum "$config" > "$exp_dir/config.sha256"
  if [[ -f "$exp_dir/TRAINING_COMPLETE" ]]; then
    log "already complete; skipping $config"
    continue
  fi

  log "waiting for GPU before $config"
  wait_for_gpu

  log "training $config in $exp_dir"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    RESUME_LAST=1 "$PYTHON" -u scripts/server/train_phone69_460h_one.py "$config" \
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
    log "ERROR training failed (exit=$train_rc); later experiments will not start"
    exit "$train_rc"
  fi

  touch "$exp_dir/TRAINING_COMPLETE"
  log "completed $config"
done

log "all three training experiments completed"
