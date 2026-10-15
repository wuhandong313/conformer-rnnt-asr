# Repository Guidelines

## Project Structure

- `conformer_rnnt/` contains the model components: Conformer encoder, predictor, joiner, dynamic compression, Medusa heads, and speculative decoding.
- `train.py` and `eval.py` are the training and evaluation entry points.
- `configs/` contains YAML configuration for local training, server experiments, ablations, and evaluation modes. Change behavior through YAML rather than adding CLI flags.
- `common.py` provides shared batching, feature normalization, configuration, and checkpoint helpers.
- There is currently no test suite or build system. Datasets, checkpoints, logs, and experiment outputs are external or gitignored.

## Development Commands

```bash
python train.py configs/default.yaml
python eval.py configs/server.yaml --cpu
python -m compileall -q train.py eval.py common.py conformer_rnnt
```

The first command trains from a YAML configuration. The second evaluates a checkpoint using the configured data and decoding mode; use `--checkpoint` to select one explicitly. The compile command performs a quick syntax check. Training normally runs on the configured GPU server.

## Experiment Workflow (GPU server)

Experiments run on the internal server at `/test21/conformer-rnnt-asr` (access via `ssh internal`, see `remote-access.md`):

- **Environment**: always use `conda env whd` (`/root/anaconda3/envs/whd/bin/python`, python 3.10 + torch 2.6 cu124). `soundfile` is required for FLAC loading; the bare `python` on the server is Python 2 — never use it.
- **Launch training** in a screen session named `iter2-train` (the monitor checks this name; the server has `screen`, not tmux):
  ```bash
  screen -dmS iter2-train bash -c 'cd /test21/conformer-rnnt-asr && /root/anaconda3/envs/whd/bin/python train.py configs/server_chain.yaml 2>&1 | tee artifacts/logs/exp_iterN_train.log'
  ```
- **Periodic monitoring** (server-side watcher; evaluates newest val ckpt every 15 min, appends one status line):
  ```bash
  screen -dmS iter8-watch bash -c 'for i in $(seq 1 48); do echo "=== $(date +%H:%M) ===" >> /test21/conformer-rnnt-asr/artifacts/logs/iter8_watch.log; cd /test21/conformer-rnnt-asr && timeout 1900 /root/anaconda3/envs/whd/bin/python scripts/monitor_iter2.py >> /test21/conformer-rnnt-asr/artifacts/logs/iter8_watch.log 2>&1; sleep 900; done'
  ```
  `monitor_iter2.py` skips ckpts younger than 180 s (Lightning may still be writing them) and caches one JSON per ckpt — purge `exp_iter2/monitor_eval/*` when a new run reuses the same tag/step names, or stale results will be reported.
- **Warm-start convention**: copy the chosen best ckpt to `BEST_iterN.ckpt`, point `paths.from_checkpoint` at it, and bump the monitor's `TRAIN_LOG`. Warm-starts are weight-only (fresh optimizer/LR warmup); a brief PER transient right after restart is expected and self-heals.

## Experiment Log

Every iteration's motivation, changes, and measured results **must** be recorded in `docs/experiment-log.md` (update the overview table and append a section) before committing the iteration's results. Include the metric conventions (100-sample vs full test-clean) and the checkpoint identifier (epoch/step) with every number.

## Coding Style

Use four-space indentation, type hints for public functions, and clear docstrings for non-obvious tensor shapes or alignment rules. Use `snake_case` for functions and variables, `PascalCase` for classes, and descriptive YAML keys consistent with existing configuration files. Preserve the current PyTorch module structure and keep model behavior configuration-driven. No formatter or linter is currently configured.

## Testing Guidelines

Because no automated test suite exists, run the compile check and `scripts/smoke_chain.py` (CPU, no dataset needed — covers zero-init warm-start equivalence, sim-scan dual-path consistency, loss backprop, and speculative-decode losslessness) for any changed model path. For decoder changes, verify speculative output against greedy output. If tests are added, place them under `tests/` with names such as `test_medusa.py` and keep them deterministic where possible.

## Commits and Pull Requests

Use concise conventional prefixes matching the history, such as `feat:`, `fix:`, `refactor:`, `chore:`, or scoped forms like `feat(eval):`. Pull requests should explain the change, identify affected YAML configurations and checkpoint compatibility, and report relevant PER, RTF, or Medusa acceptance-rate results for model or decoder changes.

## Security and Configuration

Do not commit credentials, private server addresses, local dataset paths, or checkpoint contents. Keep machine-specific settings in local YAML or SSH configuration, and review `remote-access.md` carefully before sharing changes.
