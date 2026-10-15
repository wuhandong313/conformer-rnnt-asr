#!/bin/bash
# probe: bigdec training status (tag: server_lstm_word_bigdec)
TAG=server_lstm_word_bigdec
REPO=/test21/conformer-rnnt-asr
EXP=/test21/conformer-rnnt-asr/exp_word_960h_cosd_learned
P=/root/anaconda3/envs/whd/bin/python
cd "$REPO"
mkdir -p "$REPO/artifacts/logs" "$REPO/artifacts/evaluations" "$REPO/artifacts/checkpoints"
echo "NOW: $(date '+%m-%d %H:%M') / server UTC $(date -u '+%m-%d %H:%M')"
echo "PROGRESS: $(tail -c 300 ${TAG}.train.log | tr '\r' '\n' | grep . | tail -1)"
echo "PIPELINE: $(tail -2 pipeline_${TAG}.log | tr '\n' ' | ')"
CK=$(ls -t $EXP/checkpoints/last*.ckpt 2>/dev/null | head -1)
[ -n "$CK" ] && echo "CKPTFILE: $(basename $CK) mtime: $(date -r $CK '+%m-%d %H:%M:%S')"
$P - <<PY 2>/dev/null
import torch
ck = torch.load("$CK", map_location="cpu", weights_only=False)
print("CKPT epoch=%s global_step=%s" % (ck.get("epoch"), ck.get("global_step")))
cb = ck.get("callbacks") or {}
for k, v in cb.items():
    if "EarlyStopping" in str(v.get("__class__", "")) or "early" in k.lower():
        st = v.get("state_dict", v)
        print("CB EarlyStopping:", {kk: st.get(kk) for kk in ("wait_count", "stopped_epoch", "best_score", "patience") if kk in st})
    if "ModelCheckpoint" in str(v.get("__class__", "")) or "model_checkpoint" in k.lower():
        st = v.get("state_dict", v)
        print("CB ModelCheckpoint:", {kk: st.get(kk) for kk in ("best_model_score", "best_model_path") if kk in st})
PY
$P - <<PY 2>/dev/null
import glob, csv
rows = []
for fp in sorted(glob.glob("$EXP/lightning_logs/version_*/metrics.csv")):
    try:
        with open(fp) as f:
            for r in csv.DictReader(f):
                v = r.get("Losses/val_loss_epoch")
                if v and r.get("step"):
                    rows.append((int(float(r["step"])), float(v)))
    except Exception:
        pass
rows = sorted(set(rows))[-5:]
if rows:
    print("VAL_LAST:", "  ".join(f"{s}:{v:.2f}" for s, v in rows))
PY
echo "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
echo "TMUX: $(tmux ls 2>/dev/null | grep -E 'bigdec|word960|chain' || echo none)"
# ── chain watcher / champion run status ──
echo "CHAIN: $(tmux has-session -t chain 2>/dev/null && echo alive || echo dead)"
[ -f CHAIN_ABORTED ] && echo "!!! CHAIN_ABORTED: $(head -1 CHAIN_ABORTED)"
[ -f artifacts/logs/chain_watcher.log ] && echo "WATCHER: $(tail -3 artifacts/logs/chain_watcher.log | tr '\n' '|')"
[ -f artifacts/logs/pipeline_server_lstm_word_cosd_learned.log ] && echo "CHAMPION: $(tail -3 artifacts/logs/pipeline_server_lstm_word_cosd_learned.log | tr '\n' '|')"
