#!/bin/bash
# Beam20 evals for top B-matrix runs + A4 champion (auto-triggered after eval-none)
cd /test21/conformer-rnnt-asr-attn
P=/root/anaconda3/envs/whd/bin/python
CK_MAX=$(find /test21/conformer-rnnt-asr/exp_b /test21/conformer-rnnt-asr/exp -name 'val-attn-max_pool-hard-r0.27-epoch=30-step=19100.ckpt' 2>/dev/null | head -1)
echo "$(date -u '+%F %T') max-pool ckpt: $CK_MAX"
$P eval.py configs/exp/b-cosine_diff-learned.yaml --checkpoint /test21/conformer-rnnt-asr/exp_b/b-cosine_diff-learned/checkpoints/val-attn-cosine_diff-learned-r0.27-epoch=47-step=29900.ckpt --beam 20 --out results/logs/b-cosine_diff-learned.beam20.per_sample.csv > results/logs/b-cosine_diff-learned.beam20.log 2>&1
echo "$(date -u '+%F %T') cosine_diff-learned beam20 done"
$P eval.py configs/exp/b-fir-seg_avg.yaml --checkpoint /test21/conformer-rnnt-asr/exp_b/b-fir-seg_avg/checkpoints/val-attn-fir-seg_avg-r0.27-epoch=41-step=26100.ckpt --beam 20 --out results/logs/b-fir-seg_avg.beam20.per_sample.csv > results/logs/b-fir-seg_avg.beam20.log 2>&1
echo "$(date -u '+%F %T') fir-seg_avg beam20 done"
$P eval.py configs/exp/b-savgol-learned.yaml --checkpoint /test21/conformer-rnnt-asr/exp_b/b-savgol-learned/checkpoints/val-attn-savgol-learned-r0.27-epoch=42-step=26700.ckpt --beam 20 --out results/logs/b-savgol-learned.beam20.per_sample.csv > results/logs/b-savgol-learned.beam20.log 2>&1
echo "$(date -u '+%F %T') savgol-learned beam20 done"
if [ -n "$CK_MAX" ]; then
  $P eval.py configs/exp/a4-max-pool.yaml --checkpoint "$CK_MAX" --beam 20 --out results/logs/a4-max-pool.beam20.per_sample.csv > results/logs/a4-max-pool.beam20.log 2>&1
  echo "$(date -u '+%F %T') max-pool beam20 done"
fi
echo "$(date -u '+%F %T') ALL BEAM EVALS DONE"
