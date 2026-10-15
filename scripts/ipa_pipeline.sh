#!/usr/bin/env bash
# Run from any directory. Review docs/ipa_vocabulary_review.md before training.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/anaconda3/envs/whd/bin/python}"
DATASET="${DATASET:-/test21/librispeech}"
CONFIG="${CONFIG:-configs/server_ipa.yaml}"
cd "$REPO"

OPENBLAS_NUM_THREADS=1 "$PYTHON" prepare_ipa.py "$DATASET" --source words --report exp_ipa_960h_none/ipa_audit.json
"$PYTHON" train.py "$CONFIG"
"$PYTHON" eval.py "$CONFIG" --metric per --out exp_ipa_960h_none/eval_per_sample.csv
