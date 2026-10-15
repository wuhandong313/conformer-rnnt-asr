#!/bin/bash
# Word-BPE 960h 训练与 beam20 WER 测评管线
#   等 GPU 空闲 → 训练 → best val ckpt 评测 (test-clean + test-other)
# 用法: bash scripts/word_960h_pipeline.sh [config.yaml]
#   config 默认 configs/server_lstm_word.yaml; 所有日志/产物按 config 文件名 (TAG) 区分,
#   逐句 WER CSV 写入 eval_wer_per_sample_<TAG>.csv, 各实验互不覆盖
set -u
REPO=/test21/conformer-rnnt-asr-main
cd "$REPO"
P=/root/anaconda3/envs/whd/bin/python
CFG=${1:-configs/server_lstm_word.yaml}
TAG=$(basename "$CFG" .yaml)
export PIPELINE_CFG="$CFG" PIPELINE_TAG="$TAG"
LOG="$REPO/pipeline_${TAG}.log"

echo "[$(date -u +%FT%T)] pipeline start (repo: $(git rev-parse --short HEAD))" >> "$LOG"

# ── 1) 等 GPU 空闲 (显存占用 < 2GB) ─────────────────────────────────────
echo "[$(date -u +%FT%T)] waiting for GPU ..." >> "$LOG"
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -gt 2000 ]; do
  sleep 120
done
echo "[$(date -u +%FT%T)] GPU free, starting training" >> "$LOG"

# ── 2) 训练，记录 best val ckpt（失败自动从 last.ckpt 续训，最多 3 次尝试）─
rc=1
for attempt in 1 2 3; do
# external RESUME_LAST=1 wins (lets an operator restart into an existing run);
# otherwise retries resume, the first attempt starts fresh
RESUME_LAST=${RESUME_LAST:-$([ "$attempt" -gt 1 ] && echo 1 || echo 0)} \
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
"$P" - > "$REPO/${TAG}.train.log" 2>&1 <<'PYEOF'
import glob
import os

import torch, yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
import train as T

cfg = T.load_config(os.environ["PIPELINE_CFG"])
seed_everything(cfg.get("seed", 42), workers=True)
torch.set_float32_matmul_precision(cfg.get("trainer", {}).get("matmul_precision", "medium"))

model = LibriSpeechRNNTModule.from_config(cfg)
trainer = T.build_trainer(cfg)
ckpt_path = cfg.get("paths", {}).get("from_checkpoint")
if os.environ.get("RESUME_LAST") == "1":
    # save_last rotates to last-v1/-v2/... whenever last.ckpt is already taken,
    # and manual checkpoint surgery can refresh mtimes, so neither the literal
    # path nor mtime identifies the live state; global_step is the monotone one
    cands = glob.glob(os.path.join(cfg["paths"]["exp_dir"], "checkpoints", "last*.ckpt"))
    if cands:
        ckpt_path = max(
            cands, key=lambda p: torch.load(p, map_location="cpu", weights_only=False).get("global_step", -1)
        )
        print("RESUMING from", ckpt_path, flush=True)
trainer.fit(model, ckpt_path=ckpt_path)

val_cks = [c for c in trainer.callbacks
           if isinstance(c, ModelCheckpoint) and (c.monitor or "").endswith("val_loss")]
best = val_cks[0].best_model_path if val_cks else ""
print("BEST_CKPT:", best)
with open(f"best_ckpt_{os.environ['PIPELINE_TAG']}.txt", "w") as f:
    f.write(best)
PYEOF
  rc=$?
  echo "[$(date -u +%FT%T)] training attempt $attempt exit=$rc" >> "$LOG"
  [ $rc -eq 0 ] && break
done
if [ $rc -ne 0 ] || [ ! -s "$REPO/best_ckpt_${TAG}.txt" ]; then
  echo "[$(date -u +%FT%T)] TRAINING FAILED — see ${TAG}.train.log, abort" >> "$LOG"
  exit 1
fi
BEST=$(cat "$REPO/best_ckpt_${TAG}.txt")
echo "[$(date -u +%FT%T)] best ckpt: $BEST" >> "$LOG"

# ── 3) beam20 WER 评测（test = clean + other 5559 句）───────────────────
"$P" eval.py "$CFG" --checkpoint "$BEST" --metric wer \
  --out "$REPO/eval_wer_per_sample_${TAG}.csv" \
  > "$REPO/${TAG}.beam20.log" 2>&1
rc=$?
echo "[$(date -u +%FT%T)] beam20 eval exit=$rc" >> "$LOG"
grep -E "^WER:|^RTF:|Evaluation complete" "$REPO/${TAG}.beam20.log" | tail -4 >> "$LOG"
echo "[$(date -u +%FT%T)] pipeline done" >> "$LOG"
