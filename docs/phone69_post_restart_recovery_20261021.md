# Phone69 post-restart runtime recovery — 2026-10-21

The user explicitly reported restarting the server and authorized reinstalling the runtime after ten minutes, then testing the four completed models. The delayed recovery was started after 2026-10-21T15:03:33Z local UTC. The hourly phone69 monitor remains active.

## Scope and protected artifacts

No training is launched by this recovery. The preserved stride4_none baseline and the three five_frames experiments remain at epoch 50. Baseline/config/checkpoint/log/run.json files and the old evaluation directory are read only. The first recovery precheck verified all 63 baseline files by SHA-256, size and nanosecond mtime, plus the four best checkpoint and configuration hashes from the original evaluation plan. All checks passed.

The old evaluation attempt at `artifacts/evaluations/phone69_460h_five_frames_20261020` has no final per-sample CSV files and no surviving evaluation process after the restart. Its logs and launch_state are preserved. The new attempt uses `artifacts/evaluations/phone69_460h_five_frames_20261021_post_restart`, with an exclusive launch_state guard and at most four evaluation processes.

## Runtime restoration

SSH and the persistent `/test21` NFS mount recovered. The four trained model directories, best and epoch-boundary checkpoints, and completion markers remain present. The `whd` Python executable was missing. GPU inspection found an idle NVIDIA A100-SXM4-80GB with driver 535.104.12.

The runtime is restored from the prior successful `artifacts/logs/env_rebuild.log`: Python 3.10.18, torch/torchaudio 2.6.0 (prior CUDA 12.4 builds), pytorch-lightning 2.6.6 and the exact recorded dependency versions in `requirements-restored.txt`. Conda environment creation completed; dependency installation was running when this launch record was written. This record does not assert final runtime or evaluation success.

The recovery pipeline validates dependency consistency, CUDA RNN-T forward/backward, audio decoding and 2620 test-clean utterances, then checks all model/optimizer/scheduler states and offline_native_definition values. Only after those checks does it launch the four best-checkpoint evaluations with the existing monotonic-clock evaluator, PER, beam width 20 and the original configs. Completed results require 2620 rows per model, 189023 reference phones, matching reference order, finite nonnegative PER/RTF and unchanged baseline fingerprints.

## Operation and records

Recovery directory: `artifacts/recoveries/phone69_post_restart_20261021`.
Authoritative live state: `recovery_state.json`. Live logs: `pipeline.log`, `conda_create.log`, `pip_install.log`, and subsequent verification logs. These changing files are not included in this static launch snapshot. `recovery_launch_snapshot.json` records the state at publication.

The installation pipeline runs detached on the server and automatically continues to validation and evaluation. A failed stage records its error and preserves the files and any already-started evaluation processes. It does not restart failed full evaluations. The user approval covers runtime restoration and post-restart testing; other pending A01–A04/A06–A14 training changes remain unapproved.

The server wall clock was approximately 48 minutes behind the local clock. Server file ages must use server UTC, while evaluation RTF uses the existing monotonic-clock evaluator. Runtime download caches are stored on the server and are not committed. Credentials, data, checkpoints and files larger than 10 MiB are excluded from this record.
