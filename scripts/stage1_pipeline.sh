#!/bin/bash
# Stage 1 pipeline — run inside tmux session 'pipeline'.
#
#   tmux new -s pipeline 'bash scripts/stage1_pipeline.sh'
#
# 1. waits for the baseline training tmux session (baseline-none) to end
# 2. validates the full train→ckpt→eval→CSV chain via the 2-epoch smoke run
# 3. A0 diagnostics on the none anchor (rate measurement + redundancy, CPU)
# 4. builds manifests/a4.tsv — oracle eval first (anchored), then A4 runs
# 5. launches the A4 queue (serial, ~60 GPU·h)
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p results manifests
LOG=results/stage1_pipeline.log
exec >> "$LOG" 2>&1
echo "=== stage1 pipeline start $(date) ==="

source /root/anaconda3/etc/profile.d/conda.sh
conda activate whd
export PYTHONPATH="$REPO"

ANCHOR_DIR=/test21/conformer-rnnt-asr/exp_baseline_none

# ── 1. wait for baseline training ─────────────────────────────────────
while tmux has-session -t baseline-none 2>/dev/null; do sleep 300; done
echo "baseline session ended $(date)"
sleep 60  # final ckpt flush

ANCHOR_CKPT=$(ls -t "$ANCHOR_DIR"/checkpoints/val-*.ckpt 2>/dev/null | head -1)
if [ -z "$ANCHOR_CKPT" ]; then
    echo "FATAL: no val-*.ckpt in $ANCHOR_DIR/checkpoints"
    exit 1
fi
echo "anchor ckpt: $ANCHOR_CKPT"

# ── 2. smoke chain validation (2-epoch, GPU now free) ─────────────────
printf 'smoke-detector\tconfigs/exp/smoke-detector.yaml\tgreedy\n' > manifests/smoke.tsv
if ! python scripts/queue_runner.py manifests/smoke.tsv; then
    echo "FATAL: smoke chain failed — A4 queue NOT launched"
    exit 1
fi
echo "smoke chain OK $(date)"

# ── 3. A0 diagnostics on the none anchor (CPU) ────────────────────────
python analysis/rate_distribution.py configs/server_baseline_none.yaml \
    --checkpoint "$ANCHOR_CKPT" --cpu --max-samples 300 \
    --out analysis/out/rate_distribution_none_anchor.csv \
    || echo "WARN: rate measurement failed (non-fatal)"
python analysis/layer_redundancy.py configs/server_baseline_none.yaml \
    --checkpoint "$ANCHOR_CKPT" --cpu --max-samples 200 \
    --out analysis/out/layer_redundancy_none.csv \
    || echo "WARN: layer redundancy failed (non-fatal)"
echo "A0 diagnostics done $(date)"

# ── 4. A4 manifest: oracle ceiling first, then trainable baselines ────
{
    printf 'oracle-dp\tconfigs/exp/oracle-dp.yaml\tgreedy\t%s\n' "$ANCHOR_CKPT"
    for f in configs/exp/a4-*.yaml; do
        n=$(basename "$f" .yaml)
        printf '%s\t%s\tgreedy\n' "$n" "$f"
    done
} > manifests/a4.tsv
echo "--- manifests/a4.tsv ---"
cat manifests/a4.tsv

# ── 5. A4 queue (serial) ───────────────────────────────────────────────
python scripts/queue_runner.py manifests/a4.tsv
echo "=== stage1 pipeline done $(date) ==="
