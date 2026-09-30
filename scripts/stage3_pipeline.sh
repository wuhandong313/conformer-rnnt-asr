#!/bin/bash
# Stage 3 — B-matrix queue (15 runs: 5 detectors × 3 mixers, keep_rate=0.27).
# Launch ONLY after the A4 Go/No-Go judgment (training-plan Stage 2):
#
#   tmux new -s queue-b 'bash scripts/stage3_pipeline.sh'
#
# Builds manifests/b.tsv from configs/exp/b-*.yaml (idempotent queue_runner:
# already-finished runs in results/results.csv are skipped, safe to relaunch).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p results manifests
exec >> results/stage3_pipeline.log 2>&1
echo "=== stage3 B-matrix start $(date) ==="

source /root/anaconda3/etc/profile.d/conda.sh
conda activate whd
export PYTHONPATH="$REPO"

{
    for f in configs/exp/b-*.yaml; do
        n=$(basename "$f" .yaml)
        printf '%s\t%s\tgreedy\n' "$n" "$f"
    done
} > manifests/b.tsv
echo "--- manifests/b.tsv ---"
cat manifests/b.tsv

python scripts/queue_runner.py manifests/b.tsv
echo "=== stage3 B-matrix done $(date) ==="
