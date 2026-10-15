#!/bin/bash
# One-connection chain: soak group mode -> if pass, restart training on it.
set -u
cd /test21/conformer-rnnt-asr

echo "== pre-state =="
git log --oneline -1
pgrep -f "envs/whd/bin/python" | head -3 | xargs -r ps -o pid,etime,rss,cmd -p 2>/dev/null | head -5
ls -la /test21/conformer-rnnt-asr/exp_word_960h/checkpoints/ 2>/dev/null | tail -5

echo "== phase A: soak (DBG3_STEPS=500) =="
DBG3_STEPS=500 timeout 2400 /root/anaconda3/envs/whd/bin/python scripts/diagnostics/dbg3_group.py > artifacts/logs/dbg3_group.log 2>&1
rc=$?
echo "SOAK_RC=$rc"
grep -E "^\[phase0|^\[step|^\[group|^\[dbg3|Error|Traceback" artifacts/logs/dbg3_group.log | tail -25
if [ $rc -ne 0 ]; then
  echo "SOAK_FAILED — keeping current training untouched"
  exit 1
fi

echo "== phase B: restart training with group mode =="
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
echo "killing GPU pids: $PIDS"
[ -n "$PIDS" ] && kill $PIDS
tmux kill-session -t word960 2>/dev/null
sleep 15
LEFT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
if [ -n "$LEFT" ]; then echo "force kill: $LEFT"; kill -9 $LEFT; fi
sleep 10
echo "GPU now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

tmux new-session -d -s word960 'RESUME_LAST=1 bash scripts/word_960h_pipeline.sh'
sleep 90
echo "== pipeline log =="
tail -4 artifacts/logs/pipeline_word_960h.log
echo "== train log tail =="
tail -3 artifacts/logs/word_960h.train.log
tmux ls
echo "== DONE =="
