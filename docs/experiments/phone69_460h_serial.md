# Phone69 460h clean serial experiments

> 历史实验，已于 2026-10-18 被四组自然峰谷非流式重训取代；旧 checkpoint 已删除。请以 [新计算逻辑报告](../phone69_460h_native_offline_computation_20261018.md) 和 offline_native 配置为准。

Approved on 2026-10-15. The queue runs training and training-time dev-clean
validation only. No standalone test-clean evaluation is launched.

All experiments use fixed phone69 labels, train-clean-100 + train-clean-360,
seed 42, 50 total epochs, no early stopping, 8 attention heads, FFN width 512
and output/predictor interface 256. Input projection is 32 for stride4 and
128 for stride1. Time reduction concatenates frames, making internal Conformer
width 128 in all experiments. test-clean remains configured for future use.

| Order | Configuration | Stride | Internal width | Dynamic compression | Phone budget |
|---|---|---:|---:|---|---:|
| 1 | server_phone69_460h_stride4_none_s42_e50.yaml | 4 | 128 | none | 24500 |
| 2 | server_phone69_460h_stride1_cosd_learned_s42_e50.yaml | 1 | 128 | cosine_diff / learned, keep_rate=0.27, diff5 | 16500 |
| 3 | server_phone69_460h_stride4_cosd_learned_s42_e50.yaml | 4 | 128 | cosine_diff / learned, keep_rate=0.27, diff5 | 68000 |

## Batch memory checks and resume

At the user's request, the initial 8000 / 2000 / 8000 phone budgets were
increased to target approximately 90% of the A100 80 GiB device. These are
phone-token budgets, not fixed utterance counts. The RNN-T joint-element
limit remains 1,900,000,000 and can reduce a batch below its phone budget.

A one-time diagnostic uses actual FLAC durations and target lengths across
135256 train/dev utterances. It selects the largest attention, joint-tensor,
utterance-count and duration cases, and runs a disposable model through
forward, backward and Adam updates. Allocator cache is retained across cases.
These checks do not evaluate recognition quality or modify training weights.

| Experiment | Phone budget | Peak PyTorch reserved memory |
|---|---:|---:|
| stride4 / none | 24500 | 71.11 GiB |
| stride1 / CosDiff learned | 16500 | 69.77 GiB |
| stride4 / CosDiff learned | 68000 | 70.30 GiB |

CUDA context adds device memory beyond PyTorch's reserved-memory measurement.
The target is approximately 72 GiB total device use; actual use varies with
batch shape. This is a measured peak target, not a constant allocation.

Before changing batch composition, the active first experiment resumed its
last complete checkpoint at step 7362 using the original 8000-phone recipe
and finished epoch 4. A full boundary checkpoint preserves model, optimizer,
scheduler and loop state at step 8216, with four completed epochs:

    /test21/conformer-rnnt-asr/exp_phone69_460h_stride4_none_s42_e50/checkpoints/epoch_boundary_before_batch_resize_20261015T143222Z.ckpt

The larger-batch run resumes at epoch 5 and still ends at epoch 50. The other
two experiments start fresh in order. The resume selector rejects a changed
budget at a mid-epoch checkpoint; after resuming, newer checkpoints with the
new budget can be used normally. Old snapshots and logs are retained.

Feature rate is 100 frames/s (16 kHz audio, hop 160), or 25 frames/s after
stride4. CosDiff compression follows the first 2 Conformer layers. Its
right-context setting uses the post-stride frame unit: 4 for stride1 and 1
for stride4.

## Execution and monitoring

scripts/server/phone69_460h_serial_pipeline.sh runs in tmux session
`phone69_460h_serial`, serializes the three experiments, records GPU usage
every 60 seconds, resumes full checkpoints, saves the best validation
checkpoint path, and stops the queue on failure. Each experiment receives a
TRAINING_COMPLETE marker only after all 50 epochs. No eval.py command is run.
run.json records each process, config hash, epoch, global step and resume point.
Logs append across restarts.

The heartbeat checks every 30 minutes and stays quiet during normal progress.
It may repair recoverable errors and resume the queue. Any OOM-related budget
change requires a full checkpoint at a completed epoch boundary; core model,
data, learning-rate, seed and epoch settings remain fixed. Small logs and
records are synchronized; checkpoints and runtime duration caches stay on the
server.

## Launch record

Started at 2026-10-15T13:43:35.573274+00:00 on code commit
`b5d7e290e0b10602d873c2c723934ab125c2516e` with the initial budgets.
Runtime snapshots and memory-check records are tracked under
artifacts/runs/phone69_460h_serial/.
