# Repository Guidelines

## Project Structure & Module Organization

This repository implements Conformer-RNN-T speech recognition on LibriSpeech, with phoneme and word-BPE configurations.

- `conformer_rnnt/` contains the encoder, predictor, joiner, model factories, and Lightning training/data module.
- `conformer_rnnt/compression/` separates frame detectors, mixers, and compression orchestration.
- `train.py` and `eval.py` are entry points; `common.py` provides feature normalization, batching, and decoding utilities.
- `configs/` holds YAML experiment recipes; `scripts/` contains GPU-server pipelines with machine-specific paths.
- `papers/` stores research references; root CSV/PNG files record selected experiment results. No test sources are currently tracked.

## Build, Test, and Development Commands

Run commands from the repository root. There is no build step or dependency manifest. The runtime uses `torch`, `torchaudio`, `pytorch-lightning`, `sentencepiece`, `numpy`, `scipy`, and `pyyaml`.

- `python train.py configs/default.yaml` starts training after configuring dataset, tokenizer, statistics, and accelerator settings.
- `python eval.py configs/default.yaml --checkpoint /path/to/model.ckpt --cpu --max-utts 10 --out /tmp/eval-smoke.csv` evaluates a small CPU subset using a matching configuration.
- `python -m compileall -q conformer_rnnt common.py train.py eval.py` checks Python syntax without training.

For a short training check, set `trainer.fast_dev_run: true` in a temporary YAML configuration. `eval.py` selects corpus PER for `tokenizer.type: ipa`; legacy configs select WER when `data.strip_spaces: false` and CER otherwise. Adapt server pipeline paths before running them.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_CASE` constants. Follow existing type hints and docstrings; use two-space YAML indentation. No formatter or linter is configured.

Keep training options in YAML: `train.py` accepts one configuration path. Keep model factories consistent when adding parameters. `model.num_symbols` excludes the automatically added blank symbol.

## Testing Guidelines

No coverage threshold is configured. Add focused regression tests as `tests/test_<feature>.py` and run them with `python -m unittest discover -s tests`. For compression changes, verify tensor shapes, sequence lengths, preserved right-context frames, and legacy behavior. Run a small evaluation when assets are available; fixed IPA configs report corpus PER, word-BPE configs report corpus WER, and legacy phoneme-SPM configs report CER, along with RTF.

## Commit & Pull Request Guidelines

History commonly uses short, imperative messages with prefixes such as `feat(eval):`, `fix(pipeline):`, and `artifacts:`. Follow that pattern and keep commits focused.

PRs should explain the behavior change, affected configuration, validation commands/results, and related issues when applicable. Include before/after metrics or plots for model changes, identifying the dataset, checkpoint, and decoding settings.

## Configuration & Artifacts

Keep credentials in ignored `remote-access.md`. Keep datasets, tokenizer/model binaries, checkpoints, and large experiment outputs untracked. Match tokenizer vocabulary and preprocessing settings to the checkpoint, and use distinct experiment/output paths.

`configs/ipa_vocab.json` is the tracked fixed IPA inventory. IPA labels use whitespace-delimited complete phones in `*-ipa.trans.txt`, generated offline by `prepare_ipa.py`; do not split Unicode characters or mix legacy phoneme conventions. See `docs/ipa_vocabulary_review.md` for the mapping and data audit.
