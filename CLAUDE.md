# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Conformer-RNN-T (transducer) ASR system trained on LibriSpeech **phoneme** transcriptions. The distinguishing feature is a **dynamic inter-layer compression** module ([conformer_rnnt/compression.py](conformer_rnnt/compression.py)) inserted between two Conformer blocks that drops frames at acoustic change points. Final metric is **PER** (Phoneme Error Rate), not WER.

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

### The three RNN-T components ([conformer_rnnt/model.py](conformer_rnnt/model.py))
- **transcriber** = `ConformerEncoder`: `input_linear → TimeReduction (stride 4) → ConformerBlock1 → DynamicCompression → ConformerBlock2 → output_linear + LayerNorm`. The two-stage split exists *so compression can sit between stages*: Block1 captures local patterns, compression selects key frames, Block2 models longer-range context on the reduced sequence.
- **predictor** = `Predictor`: embedding + stacked layer-norm `CustomLSTM`s (target/language side).
- **joiner** = `Joiner`: additive fusion `activation(src + tgt)` → linear → logits, shape `(B,T,U,V)`.

For beam-search inference, `ConformerRNNT` exposes `@torch.jit.export` methods (`transcribe`, `predict`, `join`) consumed by `torchaudio.models.RNNTBeamSearch`. `forward()` is training-only.

### Non-obvious design decisions
- **Blank is auto-added**: `num_symbols` in config excludes blank; everywhere it's instantiated as `num_symbols + 1` (see [lightning_module.py:363](conformer_rnnt/lightning_module.py#L363), [model.py:168](conformer_rnnt/model.py#L168)). Don't add blank manually in config.
- **MelSpectrogram is NOT a registered submodule**: it's lazily built on CPU via `_get_mel_transform()` and kept out of `state_dict()`, so checkpoints contain only the Conformer/Predictor/Joiner weights.
- **Right-context padding happens in the data pipeline**, not the encoder — the feature tensor is padded by `right_context_length` frames before entering the encoder, then separated inside `DynamicCompression` (right-context frames pass through uncompressed).
- **Batching is by token count, not fixed size**: `CustomDataset` pre-sorts samples by target length (descending) and `batch_by_token_count` ([common.py](common.py)) groups until `max_token_limit` is hit. DataLoader uses `batch_size=None` with a custom collate.
- ** Phoneme transcriptions**: the `LIBRISPEECH` class ([lightning_module.py](conformer_rnnt/lightning_module.py)) reads `{speaker}-{chapter}-ph.trans.txt` files, not standard word transcriptions.

### Dynamic compression ([conformer_rnnt/compression.py](conformer_rnnt/compression.py))
Per-sample, computes inter-frame abs-diff, detects **peaks+valleys** (`scipy.signal.find_peaks`), and keeps only those frames. Filter modes (`compression.filter`): `none` (bypass), `naive`, `hanning` (learnable Hanning-weighted blending), `fir`, `savgol`, `savgol_hanning`. The `hanning`/`savgol_hanning` modes add a **learnable weight-predictor MLP** (`weight_predictor_mode`: `diff5` | `single_frame` | `five_frames`). Note: the batch loop in `forward()` is per-sample Python, not vectorized — right-context frames are split off before processing.

### Config-driven everything
`LibriSpeechRNNTModule.from_config(config)` is the canonical factory ([lightning_module.py:448](conformer_rnnt/lightning_module.py#L448)); `conformer_rnnt_model_from_config` mirrors it for the bare model. To change behavior, edit the YAML — do not add CLI flags to `train.py` (it deliberately takes only the config path).

### Two configs, one diff that matters
[configs/default.yaml](configs/default.yaml) (local) vs [configs/server.yaml](configs/server.yaml) (GPU server at `/test21/medusa`): server uses `hop_length: 160` (100fps vs 200fps), `compression.filter: savgol_hanning`, `matmul_precision: high`, tuned `num_workers`/`max_token_limit` for A100, and absolute dataset paths.

## Development environment

Training runs on a remote GPU server (K8s Docker, no `/dev/net/tun`, so ZeroTier can't run there). Development happens on this Mac over a ZeroTier→jumphost SSH tunnel — see [remote-access.md](remote-access.md) for the full topology, SSH config, credentials, and file-transfer aliases (`sshin`, `scpin`, `webin`, `duckin`). If server connection info changes, that file lists every place to update in `~/.ssh/config`, `~/.zshrc`, and itself.

Checkpoints, logs, and datasets are gitignored (`exp/`, `lightning_logs/`, `*.ckpt`, `librispeech/`) — they live on the server, not in the repo.
