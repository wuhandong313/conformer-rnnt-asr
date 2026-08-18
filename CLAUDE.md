# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Conformer-RNN-T (transducer) ASR system trained on LibriSpeech **phoneme** transcriptions. Two research features distinguish it from a vanilla RNN-T:

1. A **dynamic inter-layer compression** module ([conformer_rnnt/compression.py](conformer_rnnt/compression.py)) between two Conformer blocks that drops frames at acoustic change points.
2. **Medusa-style multi-head future-label prediction** ([conformer_rnnt/medusa.py](conformer_rnnt/medusa.py)) plus a **lossless self-speculative greedy decoder** ([conformer_rnnt/spec_decode.py](conformer_rnnt/spec_decode.py)) for faster inference.

Final metric is **PER** (Phoneme Error Rate), not WER. Active work lives on the `multihead` branch; `master`/`main` are the pre-Medusa base.

## Commands

Train (config is the only argument — all params live in YAML):
```bash
python train.py configs/default.yaml      # local
python train.py configs/server.yaml       # GPU server
```

Evaluate (auto-loads latest `val-*.ckpt` if `--checkpoint` omitted):
```bash
python eval.py configs/server.yaml
python eval.py configs/server.yaml --checkpoint exp/checkpoints/val-epoch=02-step=4000.ckpt
python eval.py configs/server.yaml --cpu   # force CPU
```

There is **no requirements file, no test suite, no build step**. Dependencies are implicit: `torch`, `torchaudio`, `pytorch-lightning`, `sentencepiece`, `numpy`, `scipy`, `pyyaml`.

## Architecture

### Data flow
`audio (.flac) → MelSpectrogram → piecewise_linear_log → GlobalStatsNormalization → (SpecAugment train-only, right-context pad) → ConformerEncoder → Predictor + Joiner → RNNT loss`

### The four RNN-T components ([conformer_rnnt/model.py](conformer_rnnt/model.py))
- **transcriber** = `ConformerEncoder`: `input_linear → TimeReduction (stride 4) → ConformerBlock1 → DynamicCompression → ConformerBlock2 → output_linear + LayerNorm`. The two-stage split exists *so compression can sit between stages*: Block1 captures local patterns, compression selects key frames, Block2 models longer-range context on the reduced sequence.
- **predictor** = `Predictor`: embedding + stacked layer-norm `CustomLSTM`s (target/language side).
- **joiner** = `Joiner`: additive fusion `activation(src + tgt)` → linear → logits, shape `(B,T,U,V)`.
- **medusa_heads** = `MedusaHeads` (optional; `None` when `medusa.num_heads: 0`): K residual heads predicting future labels for speculative decoding.

For beam-search inference, `ConformerRNNT` exposes `@torch.jit.export` methods (`transcribe`, `predict`, `join`) consumed by `torchaudio.models.RNNTBeamSearch`. `forward_train_aux()` is the single training forward — it fuses the joiner once (`joiner.fuse`) and returns both the RNN-T logits and the Medusa head logits; `predict_heads()` applies the heads to the same fused features (used by speculative decoding).

### Medusa heads & speculative decoding
**Medusa heads** ([medusa.py](conformer_rnnt/medusa.py)) graft a K-head future-label predictor onto the joiner's fused features `joint = activation(src_enc + tgt_enc)` (the fusion itself lives in `Joiner.fuse`, shared with the joiner — computed once per training step):
- **head 0** (`w_vocab(joint)`) = direct projection, used only as a KL anchor (its logits are the joiner's own logits — same projection, same joint).
- **heads 1..K** = `w_vocab(ResBlock_k(joint))`, each predicting `y_{u+k}`. Each `ResBlock` is **zero-initialized**, so initial drafts equal head-0 (stable training start — the core Medusa trick).
- **`w_vocab` is the SAME module as `joiner.linear`** (shared, not a copy — see `conformer_rnnt/model.py` factory): head 0 ≡ joiner exactly, and drafts are scored by the same projection that the verifier uses, so acceptance is not capped by two divergent argmaxes. With `freeze_base: true` the freeze loop unfreezes only `medusa_heads.res_blocks` — never `w_vocab` (unfreezing it would unfreeze the frozen `joiner.linear`).
- Aux losses: **shifted CE** (`shifted_ce_loss`, weights `0.8^k`) plus **KL-to-base** (`kl_to_base_loss`, `kl_weight`) to stop drafts drifting from the base distribution. Both losses **mask padding** (`t < src_lengths[b]` and valid label positions only), so short samples' padded frames/tokens don't corrupt head training. `medusa.freeze_base: true` freezes the base model and trains only heads.

**Self-speculative decoding** ([spec_decode.py](conformer_rnnt/spec_decode.py)) is **lossless**: the heads draft K future labels at the predictor's current position, and head-0's argmax re-runs the RNN-T greedy verifier (== the width-1 oracle) to accept the matching prefix. Since verifier == oracle argmax, output is token-for-token identical to greedy — no PER sacrifice, just fewer predictor round-trips. `greedy_decode` is the oracle baseline; `assert_lossless` checks spec == oracle on any model. (Chained single-path argmax only — tree/top-k candidates are not implemented.)

Decoding is selected by config `decoding.mode`:
- `"beam"` — standard `RNNTBeamSearch` at `beam_width` (top-1 hypothesis).
- `"spec_greedy"` — runs the real `speculative_greedy_decode` (greedy-oracle-equivalent); draft acceptance is logged from the decoder's stats. Falls back to beam on exception. Requires Medusa heads.
- `beam_width: 0` → width-1 greedy (the lossless oracle baseline). `forward()` auto-falls back to beam if no Medusa heads are present.

### Non-obvious design decisions
- **Blank is auto-added**: `num_symbols` in config excludes blank; it's instantiated as `num_symbols + 1` and `blank_idx = sp_model.get_piece_size()` ([lightning_module.py](conformer_rnnt/lightning_module.py)). Don't add blank manually in config.
- **MelSpectrogram is NOT a registered submodule**: lazily built on CPU via `_get_mel_transform()`, kept out of `state_dict()`, so checkpoints contain only model weights.
- **Right-context padding happens in the data pipeline**, not the encoder — the feature tensor is padded by `right_context_length` frames before entering the encoder, then separated inside `DynamicCompression` (right-context frames pass through uncompressed).
  - **Known issue (unfixed)**: `DynamicCompression` runs *after* stride-4 TimeReduction but subtracts `right_context_length` (=4) from the *reduced* length, so the "right context" it preserves is 4 reduced frames (1 real rc frame + the last 3 content frames), i.e. 3 content frames per sample bypass compression. Behavior is consistent between train and eval (no skew), but compression ratios are inflated vs. design. Fixing requires `right_context_length // stride` — deferred until the next from-scratch training round because old checkpoints depend on the current frame alignment.
- **Batching is by token count, not fixed size**: `CustomDataset` pre-sorts samples by target length (descending) and `batch_by_token_count` ([common.py](common.py)) groups until `max_token_limit` is hit. DataLoader uses `batch_size=None` with a custom collate.
- **Phoneme transcriptions**: `LIBRISPEECH` ([lightning_module.py](conformer_rnnt/lightning_module.py)) reads `{speaker}-{chapter}-ph.trans.txt` files, not word transcriptions. Parsing is centralized in `_read_transcript` (cached per chapter).
- **`load_state_dict` is overridden** ([lightning_module.py](conformer_rnnt/lightning_module.py)) to (a) auto-fix the `model.` prefix mismatch, (b) **drop `medusa_heads.w_vocab.*` alias keys** so an old pre-sharing Medusa checkpoint can't clobber `joiner.linear` with stale draft weights, and (c) tolerate missing keys only under the `medusa_heads.*` prefix (old base checkpoints) — anything else missing/unexpected raises instead of silently corrupting.
- **Checkpoint naming & snapshots**: filenames embed the true training-mode tag from the config — `{filter}-h{heads}-{frz|free}` (e.g. `val-naive-h1-free-epoch=04-step=800.ckpt`; `naive-base` when no Medusa) via `model_ckpt_tag` in [common.py](common.py). Each saved checkpoint also embeds a `_config_snapshot` dict (`CheckpointSnapshot`, [train.py](train.py)) recording the training config; [eval.py](eval.py) reads it on auto-load and warns (`SNAPSHOT MISMATCH`) if `compression.filter` / `medusa.num_heads` / `medusa.freeze_base` / `decoding.mode` differ from the eval config, and **auto-loads by the config's tag** (`val-{tag}-*.ckpt`) so it won't pick up a different experiment's checkpoint. This is the fix for the old trap: `val-none-*.ckpt` was actually trained with `compression.filter: naive` — evaluating it with `none` gives PER 0.377 vs 0.077 with `naive`, and the mismatched frame alignment causes decoding hallucinations. Old checkpoints without a snapshot warn and can't be verified. Always match `compression.filter` to the checkpoint's true training mode (the snapshot makes this checkable, not guessable).

### Dynamic compression ([conformer_rnnt/compression.py](conformer_rnnt/compression.py))
Per-sample, computes inter-frame abs-diff, detects **peaks+valleys** (`scipy.signal.find_peaks`), and keeps only those frames. Filter modes (`compression.filter`): `none` (bypass), `naive`, `hanning` (learnable Hanning-weighted blending), `fir`, `savgol`, `savgol_hanning`. The `hanning`/`savgol_hanning` modes add a **learnable weight-predictor MLP** (`weight_predictor_mode`: `diff5` | `single_frame` | `five_frames`). Note: the batch loop in `forward()` is per-sample Python, not vectorized — right-context frames are split off before processing. (This numpy dependency is also why spec_decode is eager, not JIT.)

### Config-driven everything
`LibriSpeechRNNTModule.from_config(config)` is the canonical factory ([lightning_module.py](conformer_rnnt/lightning_module.py)); the bare-model entry point is `conformer_rnnt_model` ([model.py](conformer_rnnt/model.py)). `load_config` / `setup_logging` / `model_ckpt_tag` / `checkpoint_dir` / `strip_special_tokens` live in [common.py](common.py), shared by train.py and eval.py. To change behavior, edit the YAML — do not add CLI flags to `train.py` (it deliberately takes only the config path).

### Configs
- [configs/default.yaml](configs/default.yaml) — local; `hop_length: 80` (200fps), `compression.filter: hanning`, `medusa.num_heads: 0`, `decoding.mode: beam`.
- [configs/server.yaml](configs/server.yaml) — GPU server (`/test21/medusa`); `hop_length: 160` (100fps), `compression.filter: naive`, `medusa.num_heads: 2` + `freeze_base: true`, `decoding.mode: spec_greedy`, `matmul_precision: high`, A100-tuned `num_workers`/`max_token_limit`, absolute paths.
- [configs/server_scratch.yaml](configs/server_scratch.yaml) — Medusa head=1, unfrozen, from-scratch ablation (same as server.yaml minus `from_checkpoint`, isolated `exp_dir`, `beam_width: 0`).
- [configs/eval_none.yaml](configs/eval_none.yaml) / [configs/eval_naive.yaml](configs/eval_naive.yaml) / [configs/eval_specgreedy.yaml](configs/eval_specgreedy.yaml) — eval-only configs differing only in `compression.filter` / `medusa.num_heads` / `decoding.*`, to disambiguate what the `val-none` checkpoint was actually trained with.
- `decoding.spec` supports only `max_draft` and `per_token_blank` (the `frame_window` / `tree` keys were removed — tree/top-k draft candidates are not implemented).

## Development environment

Training runs on a remote GPU server (K8s Docker, no `/dev/net/tun`, so ZeroTier can't run there). Development happens on this Mac over a ZeroTier→jumphost SSH tunnel — see [remote-access.md](remote-access.md) for the full topology, SSH config, credentials, and file-transfer aliases (`sshin`, `scpin`, `webin`, `duckin`). If server connection info changes, that file lists every place to update in `~/.ssh/config`, `~/.zshrc`, and itself.

Checkpoints, logs, and datasets are gitignored (`exp/`, `lightning_logs/`, `*.ckpt`, `librispeech/`) — they live on the server, not in the repo.
