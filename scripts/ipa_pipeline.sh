#!/usr/bin/env bash
# Run from any directory. Review docs/ipa_vocabulary_review.md before training.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/anaconda3/envs/whd/bin/python}"
if [[ -x "$REPO/.venv-ipa-prep/bin/python" ]]; then
  PREP_PYTHON="${PREP_PYTHON:-$REPO/.venv-ipa-prep/bin/python}"
else
  PREP_PYTHON="${PREP_PYTHON:-$PYTHON}"
fi
export NLTK_DATA="${NLTK_DATA:-$REPO/exp_ipa_960h_none/nltk_data}"
DATASET="${DATASET:-/test21/librispeech}"
CONFIG="${CONFIG:-configs/server_ipa.yaml}"
cd "$REPO"

OPENBLAS_NUM_THREADS=1 "$PREP_PYTHON" prepare_ipa.py "$DATASET" --source words --report exp_ipa_960h_none/ipa_audit.json
"$PYTHON" train.py "$CONFIG"
"$PYTHON" eval.py "$CONFIG" --metric per --out exp_ipa_960h_none/eval_per_sample.csv
