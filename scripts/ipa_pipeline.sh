#!/usr/bin/env bash
# Run from any directory. Review docs/ipa_vocabulary_review.md before training.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/anaconda3/envs/whd/bin/python}"
DATASET="${DATASET:-/test21/librispeech}"
LABELS_ROOT="${LABELS_ROOT:-/test21/librispeech}"
CONFIG="${CONFIG:-configs/server_ipa.yaml}"
cd "$REPO"

"$PYTHON" audit_ipa_data.py "$DATASET" --labels-root "$LABELS_ROOT" --report exp_ipa_960h_none/ipa_audit.json
"$PYTHON" train.py "$CONFIG"
"$PYTHON" eval.py "$CONFIG" --metric per --out exp_ipa_960h_none/eval_per_sample.csv
