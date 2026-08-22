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

## Coding Style

Use four-space indentation, type hints for public functions, and clear docstrings for non-obvious tensor shapes or alignment rules. Use `snake_case` for functions and variables, `PascalCase` for classes, and descriptive YAML keys consistent with existing configuration files. Preserve the current PyTorch module structure and keep model behavior configuration-driven. No formatter or linter is currently configured.

## Testing Guidelines

Because no automated test suite exists, run the compile check and focused smoke tests for changed model paths. For decoder changes, verify speculative output against greedy output. If tests are added, place them under `tests/` with names such as `test_medusa.py` and keep them deterministic where possible.

## Commits and Pull Requests

Use concise conventional prefixes matching the history, such as `feat:`, `fix:`, `refactor:`, `chore:`, or scoped forms like `feat(eval):`. Pull requests should explain the change, identify affected YAML configurations and checkpoint compatibility, and report relevant PER, RTF, or Medusa acceptance-rate results for model or decoder changes.

## Security and Configuration

Do not commit credentials, private server addresses, local dataset paths, or checkpoint contents. Keep machine-specific settings in local YAML or SSH configuration, and review `remote-access.md` carefully before sharing changes.
