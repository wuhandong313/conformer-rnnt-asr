#!/bin/bash
cd /test21/conformer-rnnt-asr-main || exit 1
/root/anaconda3/envs/whd/bin/python -u eval.py /test21/conformer-rnnt-asr-main/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/config.yaml --checkpoint /test21/conformer-rnnt-asr/exp_word_960h_cosd_learned_b14000/checkpoints/val-none-epoch=11-step=227500.ckpt --metric wer --out /test21/conformer-rnnt-asr-main/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/per_sample_wer.csv > /test21/conformer-rnnt-asr-main/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/eval.log 2>&1 &
eval_pid=$!
printf "%s\n" "$eval_pid" > /test21/conformer-rnnt-asr-main/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/pid
wait "$eval_pid"
rc=$?
printf "%s\n" "$rc" > /test21/conformer-rnnt-asr-main/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/exit_code
exit "$rc"
