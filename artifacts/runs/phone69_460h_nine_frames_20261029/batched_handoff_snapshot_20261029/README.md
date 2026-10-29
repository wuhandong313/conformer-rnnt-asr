# Nine-frame batched-backend handoff, 2026-10-29

The running experiment switched at the complete epoch-10 boundary, step
25310. The checkpoint retained model, optimizer and scheduler state. The
original log and checkpoints remained on the server. The tracked recipe adds
only `execution_backend: batched`; the runtime recipe additionally specifies
the boundary checkpoint. The seed and batch budget are unchanged, but restarting
the data loader can change subsequent shuffle and augmentation random streams.

The GPU memory probe completed five real batch shapes with a disposable model.
Its maximum PyTorch reserved memory was 68.584 GiB and maximum device usage
69.101 GiB. The preflight ran one optimizer step per tested shape. Separately,
the two 4-utterance real-sample comparisons used the trained checkpoint and
passed selected-index, output, input-gradient and mixer-gradient comparisons.

The `speed_monotonic.json` observation measures steps 25500 through 25700,
including one scheduled dev validation, and excludes startup and preflight.
It took 133.53 seconds (1.498 steps/s). The server's wall clock advanced only
121.19 seconds over the same interval, so historic wall-clock timestamps and
progress-bar rates cannot support a reliable end-to-end speedup comparison.
Small-module comparisons from the separate GPU test are not epoch speedups.

Files named `*_snapshot` were copied while training continued and are not final
run records. The live state and full logs remain in the server experiment and
`batched_handoff_20261029` directory.
