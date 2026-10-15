#!/bin/bash
# LSTM 复现全自动管线（emformer_rnnt_copy stride4_comp 配方）
#   等 GPU 空闲 → 训练 50ep（naive filter + sum loss）→ best val ckpt 直接 beam20 评测
# 用法: tmux 里 bash scripts/lstm_repro_pipeline.sh（工作目录任意）
set -u
REPO=/test21/conformer-rnnt-asr
cd "$REPO"
mkdir -p "$REPO/artifacts/logs" "$REPO/artifacts/evaluations" "$REPO/artifacts/checkpoints"
P=/root/anaconda3/envs/whd/bin/python
LOG="$REPO/artifacts/logs/pipeline.log"

echo "[$(date -u +%FT%T)] pipeline start (repo: $(git rev-parse --short HEAD))" >> "$LOG"

# ── 1) 等 baseline beam20 eval 结束（GPU 释放）────────────────────────────
echo "[$(date -u +%FT%T)] waiting for GPU (baseline eval running) ..." >> "$LOG"
# 锚定真实 python 进程（tmux server 进程 argv 里也含 eval.py 字样，不能裸 pgrep）
while pgrep -f "^/root/anaconda3/envs/whd/bin/python eval.py configs/server_baseline_none.yaml" >/dev/null; do sleep 60; done
echo "[$(date -u +%FT%T)] GPU free, starting training" >> "$LOG"

# ── 2) 训练，记录 best val ckpt ──────────────────────────────────────────
"$P" - > "$REPO/artifacts/logs/lstm_naive.train.log" 2>&1 <<'PYEOF'
import torch, yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
import train as T

cfg = T.load_config("configs/server_lstm_naive.yaml")
seed_everything(cfg.get("seed", 42), workers=True)
torch.set_float32_matmul_precision(cfg.get("trainer", {}).get("matmul_precision", "medium"))

model = LibriSpeechRNNTModule.from_config(cfg)
trainer = T.build_trainer(cfg)
trainer.fit(model, ckpt_path=cfg.get("paths", {}).get("from_checkpoint"))

val_cks = [c for c in trainer.callbacks
           if isinstance(c, ModelCheckpoint) and (c.monitor or "").endswith("val_loss")]
best = val_cks[0].best_model_path if val_cks else ""
print("BEST_CKPT:", best)
with open("artifacts/checkpoints/best_ckpt.txt", "w") as f:
    f.write(best)
PYEOF
rc=$?
echo "[$(date -u +%FT%T)] training exit=$rc" >> "$LOG"
if [ $rc -ne 0 ] || [ ! -s "$REPO/artifacts/checkpoints/best_ckpt.txt" ]; then
  echo "[$(date -u +%FT%T)] TRAINING FAILED — see artifacts/logs/lstm_naive.train.log, abort" >> "$LOG"
  exit 1
fi
BEST=$(cat "$REPO/artifacts/checkpoints/best_ckpt.txt")
echo "[$(date -u +%FT%T)] best ckpt: $BEST" >> "$LOG"

# ── 3) beam20 评测（同一 config，保证架构一致）────────────────────────────
"$P" eval.py configs/server_lstm_naive.yaml --checkpoint "$BEST" \
  > "$REPO/artifacts/logs/lstm_naive.beam20.log" 2>&1
rc=$?
echo "[$(date -u +%FT%T)] beam20 eval exit=$rc" >> "$LOG"
grep -E "^PER:|^RTF:|Evaluation complete" "$REPO/artifacts/logs/lstm_naive.beam20.log" | tail -4 >> "$LOG"
echo "[$(date -u +%FT%T)] pipeline done" >> "$LOG"
