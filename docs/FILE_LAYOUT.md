# 分支文件布局

训练、评估、推理入口及共享运行模块保留在根目录，内部网络模块在 `conformer_rnnt/`。

| 目录 | 内容 |
| --- | --- |
| artifacts/logs/ | 小型运行、训练、评估日志 |
| artifacts/evaluations/ | 逐句指标、评估 CSV/JSON 及可复现实验快照 |
| artifacts/training/ | 损失曲线数据 |
| artifacts/figures/ | 图表 |
| artifacts/checkpoints/ | 最优 checkpoint 的路径记录，不含大模型 |
| assets/tokenizers/ | 小型词表模型 |
| assets/statistics/ | 特征统计文件 |
| scripts/diagnostics/ | 性能探测与调试工具 |
| scripts/server/ | 服务器辅助脚本 |
| papers/ | 参考论文 |
| docs/ | 研究计划与报告 |
| tests/ | 验证脚本与测试 |

仅创建本分支实际需要的目录。服务器统一使用 `/test21/conformer-rnnt-asr`，旧工作树目录仅作兼容链接。运行命令以仓库根目录为工作目录。超过 10 MiB 的文件留在服务器，凭据和运行缓存不入库。

## 本次文件迁移

| 原路径 | 新路径 |
| --- | --- |
| `ArtNet_camera_ready_20260611.pdf` | `papers/artnet/ArtNet_camera_ready_20260611.pdf` |
| `baseline_vs_bigdec_curves.png` | `artifacts/figures/baseline_vs_bigdec_curves.png` |
| `bench.log` | `artifacts/logs/bench.log` |
| `bench_rnnt.py` | `scripts/diagnostics/bench_rnnt.py` |
| `best_ckpt.txt` | `artifacts/checkpoints/best_ckpt.txt` |
| `best_ckpt_server_lstm_word_bigdec.txt` | `artifacts/checkpoints/best_ckpt_server_lstm_word_bigdec.txt` |
| `best_ckpt_server_lstm_word_cosd_learned_b14000.txt` | `artifacts/checkpoints/best_ckpt_server_lstm_word_cosd_learned_b14000.txt` |
| `best_ckpt_word_960h.txt` | `artifacts/checkpoints/best_ckpt_word_960h.txt` |
| `chain_watcher.log` | `artifacts/logs/chain_watcher.log` |
| `chain_watcher.sh` | `scripts/server/chain_watcher.sh` |
| `dbg2_chunk.log` | `artifacts/logs/dbg2_chunk.log` |
| `dbg2_chunk.py` | `scripts/diagnostics/dbg2_chunk.py` |
| `dbg3_group.log` | `artifacts/logs/dbg3_group.log` |
| `dbg3_group.py` | `scripts/diagnostics/dbg3_group.py` |
| `dbg_guard_crash.log` | `artifacts/logs/dbg_guard_crash.log` |
| `dbg_guard_crash.py` | `scripts/diagnostics/dbg_guard_crash.py` |
| `dbg_rnnt.log` | `artifacts/logs/dbg_rnnt.log` |
| `dbg_rnnt2.log` | `artifacts/logs/dbg_rnnt2.log` |
| `dbg_rnnt3.log` | `artifacts/logs/dbg_rnnt3.log` |
| `dbg_rnnt4.log` | `artifacts/logs/dbg_rnnt4.log` |
| `env_rebuild.log` | `artifacts/logs/env_rebuild.log` |
| `eval_per_sample_server_lstm_word_bigdec.csv` | `artifacts/evaluations/eval_per_sample_server_lstm_word_bigdec.csv` |
| `eval_per_sample_server_lstm_word_cosd_learned_b14000.csv` | `artifacts/evaluations/eval_per_sample_server_lstm_word_cosd_learned_b14000.csv` |
| `eval_per_sample_word960.csv` | `artifacts/evaluations/eval_per_sample_word960.csv` |
| `eval_per_sample_word960_r2.csv` | `artifacts/evaluations/eval_per_sample_word960_r2.csv` |
| `eval_probe_bigdec.csv` | `artifacts/evaluations/eval_probe_bigdec.csv` |
| `eval_probe_bigdec_ep4.csv` | `artifacts/evaluations/eval_probe_bigdec_ep4.csv` |
| `eval_probe_now.log` | `artifacts/logs/eval_probe_now.log` |
| `global_stats.json` | `assets/statistics/global_stats.json` |
| `lstm_naive.beam20.log` | `artifacts/logs/lstm_naive.beam20.log` |
| `lstm_naive.train.log` | `artifacts/logs/lstm_naive.train.log` |
| `pipeline.log` | `artifacts/logs/pipeline.log` |
| `pipeline_server_lstm_word_bigdec.log` | `artifacts/logs/pipeline_server_lstm_word_bigdec.log` |
| `pipeline_server_lstm_word_cosd_learned.log` | `artifacts/logs/pipeline_server_lstm_word_cosd_learned.log` |
| `pipeline_server_lstm_word_cosd_learned_b10000.log` | `artifacts/logs/pipeline_server_lstm_word_cosd_learned_b10000.log` |
| `pipeline_server_lstm_word_cosd_learned_b14000.log` | `artifacts/logs/pipeline_server_lstm_word_cosd_learned_b14000.log` |
| `pipeline_word_960h.log` | `artifacts/logs/pipeline_word_960h.log` |
| `probe_word960.sh` | `scripts/server/probe_word960.sh` |
| `rebuild_and_resume.sh` | `scripts/server/rebuild_and_resume.sh` |
| `remote_soak_restart.sh` | `scripts/server/remote_soak_restart.sh` |
| `server_lstm_word_bigdec.beam20.log` | `artifacts/logs/server_lstm_word_bigdec.beam20.log` |
| `server_lstm_word_cosd_learned.train.log` | `artifacts/logs/server_lstm_word_cosd_learned.train.log` |
| `server_lstm_word_cosd_learned_b10000.train.log` | `artifacts/logs/server_lstm_word_cosd_learned_b10000.train.log` |
| `server_lstm_word_cosd_learned_b14000.beam20.log` | `artifacts/logs/server_lstm_word_cosd_learned_b14000.beam20.log` |
| `spm_bpe_4000_word.model` | `assets/tokenizers/spm_bpe_4000_word.model` |
| `spm_bpe_4000_word_960h.model` | `assets/tokenizers/spm_bpe_4000_word_960h.model` |
| `tb_read.py` | `scripts/diagnostics/tb_read.py` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code/conformer_rnnt/metrics.py` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code/conformer_rnnt/metrics.py` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code/eval.py` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code/eval.py` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code/scripts/lstm_repro_pipeline.sh` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code/scripts/lstm_repro_pipeline.sh` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code/scripts/word_960h_pipeline.sh` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code/scripts/word_960h_pipeline.sh` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code_before/eval.py` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code_before/eval.py` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code_before/scripts/lstm_repro_pipeline.sh` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code_before/scripts/lstm_repro_pipeline.sh` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code_before/scripts/word_960h_pipeline.sh` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/code_before/scripts/word_960h_pipeline.sh` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/config.yaml` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/config.yaml` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/eval.log` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/eval.log` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/exit_code` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/exit_code` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/manifest.json` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/manifest.json` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/per_sample_wer.csv` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/per_sample_wer.csv` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/pid` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/pid` |
| `wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/run.sh` | `artifacts/evaluations/wer_eval_runs/best_ep11_step227500_beam20_5a0ff5e8/run.sh` |
| `word960_loss.csv` | `artifacts/training/word960_loss.csv` |
| `word960_loss.png` | `artifacts/figures/word960_loss.png` |
| `word_960h.beam20.log` | `artifacts/logs/word_960h.beam20.log` |
| `word_bigdec.evalprobe.log` | `artifacts/logs/word_bigdec.evalprobe.log` |
| `word_bigdec.evalprobe_ep4.log` | `artifacts/logs/word_bigdec.evalprobe_ep4.log` |
