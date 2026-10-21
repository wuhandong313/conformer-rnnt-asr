# Phone69 460h five_frames: completed test-clean evaluation

All four best-checkpoint evaluations completed and passed validation after the server restart and runtime restoration. The training queue is complete; no retraining was performed.

## Results

Test-clean; 2620 utterances and 189023 reference phones per model; phone69 corpus PER; beam width 20. RTF is the arithmetic mean of per-utterance monotonic-clock RTF, measured while up to four jobs shared one A100. Concurrency fell as jobs completed, so these are not isolated-GPU speed benchmarks.

| Model | Training epoch / step | Best checkpoint epoch / step | Phone errors | PER | Mean utterance RTF |
| --- | --- | --- | ---: | ---: | ---: |
| stride4_none | 50 / 56900 | 41 / 46858 | 11584 | 6.1284% | 0.444677 |
| stride1_cosd_learned | 50 / 126550 | 27 / 69137 | 10433 | 5.5194% | 0.497685 |
| stride2_cosd_learned | 50 / 59850 | 20 / 24540 | 12692 | 6.7145% | 0.395501 |
| stride4_cosd_learned | 50 / 48550 | 44 / 43524 | 14379 | 7.6070% | 0.278114 |

Stride1 five_frames has the lowest PER: 5.5194%, a 0.6089 percentage-point reduction (9.94% relative) from the preserved baseline. This is the result of one seed and this test set.

## Validation and protection

- Every experiment has TRAINING_COMPLETE, status trained and epoch 50. Best and epoch-boundary checkpoints include full model, optimizer, scheduler and offline_native_definition; the three dynamic models use five_frames consistently.
- Every CSV has exactly 2620 rows in index order, 189023 reference phones, matching reference order across models, and finite nonnegative PER/RTF.
- The final baseline audit rehashed all 63 original files (2762421657 bytes), comparing SHA-256, size and nanosecond modification time. All unchanged.
- No training source/configuration change was approved or made during recovery. A01–A04/A06–A14 remain pending.
- All evaluation jobs exited; GPU snapshot: NVIDIA A100-SXM4-80GB, 4, 81920, 0.

## Records

Server result directory: `/test21/conformer-rnnt-asr/artifacts/evaluations/phone69_460h_five_frames_20261021_post_restart`. Each model has `<name>.log` and `<name>_per_sample.csv`; `summary.json`, `validation_report.json`, `run_plan.json`, `launch_state.json` and `EVALUATION_COMPLETE` are preserved.

Local result directory: `/Users/wuhandong/Desktop/Projects/conformer-rnnt-asr/artifacts/evaluations/phone69_460h_five_frames_20261021_post_restart`. Runtime and training integrity reports: `/Users/wuhandong/Desktop/Projects/conformer-rnnt-asr/artifacts/recoveries/phone69_post_restart_20261021`.

Latest observation: server UTC 2026-10-21T17:21:47.671974+00:00; local UTC 2026-10-21T18:09:39.821461+00:00; server minus local -2872.149 seconds. File ages use the server clock. Latest model log UTC 2026-10-21T17:19:13.781961+00:00 (age about 154 seconds at observation); latest completed training log UTC 2026-10-20T08:06:03.773590+00:00 and epoch-boundary checkpoint UTC 2026-10-20T08:05:58.832520+00:00 belong to completed training.

The hourly phone69 automation is to be paused after this completion report. Credentials, datasets, checkpoints and files over 10 MiB are not part of the records commit. GitHub follows the existing Gitee mirror process.
