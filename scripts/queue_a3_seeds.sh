#!/bin/bash
# Serial queue for A3 baseline seeds (50 epochs each): s1111 then s2222.
cd /test21/conformer-rnnt-asr
PY=/root/anaconda3/envs/whd/bin/python
echo "$(date -Is) queue start, launching s1111"
mkdir -p exp/a3-none-s1111
$PY train.py configs/exp/a3-none-s1111.yaml > exp/a3-none-s1111/train.log 2>&1
echo "$(date -Is) s1111 finished, launching s2222"
mkdir -p exp/a3-none-s2222
$PY train.py configs/exp/a3-none-s2222.yaml > exp/a3-none-s2222/train.log 2>&1
echo "$(date -Is) all A3 seed runs done"
