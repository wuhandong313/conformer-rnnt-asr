# Repository Guidelines

## Project Structure & Module Organization

This repository implements Conformer-RNN-T speech recognition on LibriSpeech, with phoneme and word-BPE configurations.

- `conformer_rnnt/` contains the encoder, predictor, joiner, model factories, and Lightning training/data module.
- `conformer_rnnt/compression/` separates frame detectors, mixers, and compression orchestration.
- `train.py` and `eval.py` are entry points; `common.py` provides feature normalization, batching, and decoding utilities.
- `configs/` holds YAML experiment recipes; `scripts/` contains GPU-server pipelines with machine-specific paths.
- `papers/` stores research references; `artifacts/` stores logs, metrics and figures, and `assets/` holds small tokenizer models and statistics. Focused regression tests live in `tests/`.

## Build, Test, and Development Commands

Run commands from the repository root. There is no build step or dependency manifest. The runtime uses `torch`, `torchaudio`, `pytorch-lightning`, `numpy`, `scipy`, and `pyyaml`; word-BPE additionally requires `sentencepiece`.

- `python train.py configs/default.yaml` starts training after configuring dataset, tokenizer, statistics, and accelerator settings.
- `python eval.py configs/default.yaml --checkpoint /path/to/model.ckpt --cpu --max-utts 10 --out /tmp/eval-smoke.csv` evaluates a small CPU subset using a matching configuration.
- `python -m compileall -q conformer_rnnt common.py train.py eval.py` checks Python syntax without training.

For a short training check, set `trainer.fast_dev_run: true` in a temporary YAML configuration. `eval.py` selects corpus PER for `tokenizer.type: ipa` or `phone69`; legacy configs select WER when `data.strip_spaces: false` and CER otherwise. Adapt server pipeline paths before running them.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_CASE` constants. Follow existing type hints and docstrings; use two-space YAML indentation. No formatter or linter is configured.

Keep training options in YAML: `train.py` accepts one configuration path. Keep model factories consistent when adding parameters. `model.num_symbols` excludes the automatically added blank symbol.

## Testing Guidelines

No coverage threshold is configured. Add focused regression tests as `tests/test_<feature>.py` and run them with `python -m unittest discover -s tests`. For compression changes, verify tensor shapes, sequence lengths, preserved right-context frames, and legacy behavior. Run a small evaluation when assets are available; fixed phone69/IPA configs report corpus PER and word-BPE configs report corpus WER, along with RTF.

## Commit & Pull Request Guidelines

History commonly uses short, imperative messages with prefixes such as `feat(eval):`, `fix(pipeline):`, and `artifacts:`. Follow that pattern and keep commits focused.

PRs should explain the behavior change, affected configuration, validation commands/results, and related issues when applicable. Include before/after metrics or plots for model changes, identifying the dataset, checkpoint, and decoding settings.

## Configuration & Artifacts

Keep credentials in ignored `remote-access.md`. Keep datasets and files larger than 10 MiB untracked. Small logs, scripts, experiment records and word tokenizer models are synchronized when explicitly requested; runtime caches and environments stay local. Match tokenizer vocabulary and preprocessing settings to the checkpoint, and use distinct experiment/output paths. See `docs/repository_sync.md` for consolidated branches and server paths.

`configs/ipa_vocab.json` is the tracked fixed IPA inventory. IPA labels use whitespace-delimited complete phones in external Epitran files, read from external `<subset_with_underscores>/ipa` files by `ipa_labels.py`; do not split Unicode characters or mix legacy phoneme conventions. See `docs/ipa_vocabulary_review.md` for the mapping and data audit.

`configs/phone69_vocab.json` fixes exactly 69 stressed ARPABET phones (blank ID 69); source noise label `spn` is explicitly filtered during encoding. Phone69 reads external per-chapter `*-ph.trans.txt`; preserve case, stress and spaces. Phone69 SentencePiece configs/checkpoints are no longer supported. Server `train-other-500` uses IPA and cannot join phone69 training. See `docs/phone69_tokenizer.md`.
