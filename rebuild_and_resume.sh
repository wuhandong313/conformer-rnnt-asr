#!/bin/bash
# Pod restart recovery: rebuild whd env -> verify -> auto-resume bigdec training
set -u
REPO=/test21/conformer-rnnt-asr-main
LOG="$REPO/env_rebuild.log"
exec >> "$LOG" 2>&1
echo "===== rebuild start $(date -u '+%F %T') UTC ====="

echo "[1/3] conda create whd (python 3.10)"
/root/anaconda3/bin/conda create -n whd python=3.10 -y
if [ ! -x /root/anaconda3/envs/whd/bin/python ]; then
  echo "FATAL: conda env create failed"; exit 1
fi

echo "[2/3] pip install (Tsinghua mirror, ~9GB nvidia wheels, 30-50 min)"
/root/anaconda3/envs/whd/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  torch==2.6.0 torchaudio==2.6.0 numpy scipy sentencepiece pyyaml soundfile pytorch-lightning

echo "[3/3] verify"
/root/anaconda3/envs/whd/bin/python - <<'PY'
import torch, torchaudio, soundfile, yaml, sentencepiece
import pytorch_lightning as pl
assert torch.cuda.is_available(), "CUDA not available"
print("ENV_OK", "torch", torch.__version__, "| pl", pl.__version__, "|", torch.cuda.get_device_name(0))
PY

if ! grep -q ENV_OK "$LOG"; then
  echo "FATAL: env verify failed - training NOT started"; exit 1
fi

echo "===== env ready $(date -u '+%F %T') UTC - launching training (RESUME_LAST=1) ====="
cd "$REPO"
RESUME_LAST=1 bash scripts/word_960h_pipeline.sh configs/server_lstm_word_bigdec.yaml
