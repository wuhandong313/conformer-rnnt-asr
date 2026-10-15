#!/bin/bash
# chain watcher: bigdec session end -> verify pipeline done -> pull -> champion run
REPO=/test21/conformer-rnnt-asr
LOG=$REPO/artifacts/logs/pipeline_server_lstm_word_bigdec.log
CW=$REPO/artifacts/logs/chain_watcher.log
MARK=$REPO/CHAIN_ABORTED
cd "$REPO" || exit 1
mkdir -p "$REPO/artifacts/logs" "$REPO/artifacts/evaluations" "$REPO/artifacts/checkpoints" || exit 1
echo "watcher start $(date '+%m-%d %H:%M:%S')" >> "$CW"
while true; do
  if ! tmux has-session -t bigdec 2>/dev/null; then
    echo "bigdec session gone $(date '+%m-%d %H:%M:%S')" >> "$CW"
    sleep 120
    if grep -q "pipeline done" "$LOG" 2>/dev/null; then
      echo "pipeline done confirmed, pulling $(date '+%m-%d %H:%M:%S')" >> "$CW"
      git pull origin main >> "$CW" 2>&1
      rc=$?
      echo "git pull rc=$rc HEAD=$(git rev-parse --short HEAD)" >> "$CW"
      if [ $rc -ne 0 ]; then
        echo "CHAIN_ABORTED: git pull failed $(date '+%m-%d %H:%M:%S')" >> "$CW"
        touch "$MARK"; exit 1
      fi
      if [ ! -f configs/server_lstm_word_cosd_learned.yaml ]; then
        echo "CHAIN_ABORTED: champion config missing after pull $(date '+%m-%d %H:%M:%S')" >> "$CW"
        touch "$MARK"; exit 1
      fi
      echo "launching champion run $(date '+%m-%d %H:%M:%S')" >> "$CW"
      RESUME_LAST=0 bash scripts/word_960h_pipeline.sh configs/server_lstm_word_cosd_learned.yaml >> "$CW" 2>&1
      rc=$?
      echo "champion pipeline exited rc=$rc $(date '+%m-%d %H:%M:%S')" >> "$CW"
      exit $rc
    else
      echo "CHAIN_ABORTED: bigdec ended without 'pipeline done' $(date '+%m-%d %H:%M:%S')" >> "$CW"
      tail -5 "$LOG" >> "$CW" 2>/dev/null
      touch "$MARK"; exit 1
    fi
  fi
  sleep 300
done
