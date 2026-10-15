# Phone69 460h clean serial experiments

Status: approved on 2026-10-15; server startup includes three GPU memory probes
before the full serial training queue.

All experiments use fixed phone69 labels, train-clean-100 + train-clean-360,
dev-clean validation, test-clean evaluation, seed 42, 50 epochs, no early
stopping, 8 attention heads, FFN width 512 and output/predictor interface 256.
Input projection is 32 for stride4 and 128 for stride1. Time reduction
concatenates frames, making internal Conformer width 128 in all experiments.

| Order | Configuration | Stride | Internal width | Dynamic compression | Phone budget |
|---|---|---:|---:|---|---:|
| 1 | server_phone69_460h_stride4_none_s42_e50.yaml | 4 | 128 | none | 8000 |
| 2 | server_phone69_460h_stride1_cosd_learned_s42_e50.yaml | 1 | 128 | cosine_diff / learned, keep_rate=0.27, diff5 | 2000 |
| 3 | server_phone69_460h_stride4_cosd_learned_s42_e50.yaml | 4 | 128 | cosine_diff / learned, keep_rate=0.27, diff5 | 8000 |

## Batch estimate

The historical clean-100 stride4 baseline had internal width 512, FFN 1024
and 8 heads. Its 8000-phone probe peaked at 49.73 GiB reserved. The full
10000-phone training run later sampled 77.88 GiB GPU usage. These measurements
are in artifacts/runs/phone69_100h_none_s42_e40/.

The earlier 750-phone stride1 cap was conservative. For similar utterance
lengths and approximately proportional batch size, a 2000-phone stride1 run
has the following total attention-matrix size relative to the old 8000-phone
stride4 baseline:

    (2000 / 8000) * (2 * 4^2 + 8 * (4 * 0.27)^2) / 10 = 1.03328

The first 2 layers see four times as many frames. The following 8 layers see
at most approximately 1.08 times as many frames after CosDiff compression.
Using internal width 128 reduces channel-dependent activations and parameters;
attention matrices still depend on 8 heads and sequence length squared.
FFN and RNN-T memory must be considered separately, so this ratio is not a
prediction of total GPU memory. It excludes padding/context rounding and
assumes similar batch composition. Clean-360 may have different worst cases.

The initial limits are estimates. probe_phone69_config.py scans real audio
headers using a shared local cache, selects batches with the largest attention,
joint tensor, utterance count and duration across both training subsets and
dev-clean, and runs disposable-model forward/backward/optimizer steps.
Reports are saved as batch_probe.json in each experiment directory. OOM,
non-finite values or peak reserved memory above 85% stop the queue before full
training. The actual training runs use fresh models and the approved configs.

Feature rate is 100 frames/s (16 kHz audio, hop 160), or 25 frames/s after
stride4. CosDiff compression follows the first 2 Conformer layers. Its
right-context setting uses the post-stride frame unit: 4 for stride1 and 1
for stride4.

## Execution

scripts/server/phone69_460h_serial_pipeline.sh runs inside tmux, serializes the
three experiments, records GPU usage every 60 seconds, resumes last full
checkpoints, evaluates the best validation checkpoint on test-clean PER, and
stops the queue on failure. run.json records each training process, config hash,
epoch and global step. Logs append across restarts.
