# Preparing LibriSpeech and fixed Phone69 labels

Keep the repository and dataset separate. The existing server recipes use these paths:

```text
/test21/conformer-rnnt-asr/                 Git checkout
/test21/librispeech/LibriSpeech/<subset>/   Downloaded audio and chapter labels
/test21/librispeech/global_stats.json      Feature normalization statistics
```

Synchronize source code with Git before running these commands. Download audio on the target machine from [OpenSLR SLR12](https://www.openslr.org/12/); do not copy another machine's audio tree. The data preparation scripts use Python's standard library and do not require the training environment.

## Download audio

From the repository root, prepare the four subsets used by the current 460-hour experiments:

```bash
python3 scripts/data/download_librispeech.py \
  --dataset-root /test21/librispeech \
  --subsets train-clean-100 train-clean-360 dev-clean test-clean
```

For the full original LibriSpeech audio collection, also include `train-other-500`, `dev-other`, and `test-other`. OpenSLR lists `https://openslr.magicdatatech.com/resources/12/` as a Chinese mirror; it can be selected with `--base-url`. The downloader verifies the original SHA-256 checksums before publishing an extracted subset and supports resuming a partial download.

An HTTP response containing a campus login page is not a successful dataset download. Complete the target machine's network authentication before starting a download job.

## Prepare chapter labels

The official audio archives contain English text, not the Phone69 references used in these experiments. The original `extract_ph.py` in the Emformer project distributed existing flat MFA-derived labels into chapter files; it did not generate new phonemes. Use the original labels, rather than substituting a new G2P or forced-alignment run.

Keep the flat source files outside Git, for example:

```text
/home/handongwu/datasets/phone69_sources/
  train_clean_100/phone69.trans.txt
  train_clean_360/phone69.trans.txt
  dev_clean/phone69.trans.txt
  test_clean/phone69.trans.txt
  dev_other/phone69
  test_other/phone69
```

The audited source hashes and counts are recorded in [phone69_sources.json](../scripts/data/phone69_sources.json). Explicit permission to transfer the original label files does not permit copying the audio corpus or putting labels in Git.

After downloading audio and verifying the source label hashes, run:

```bash
python3 scripts/data/extract_phone69.py \
  --labels-root /home/handongwu/datasets/phone69_sources \
  --dataset-root /test21/librispeech \
  --source-manifest scripts/data/phone69_sources.json \
  --subsets train-clean-100 train-clean-360 dev-clean test-clean
```

The script validates all selected input labels and audio IDs before writing chapter files. Repeated runs do not append duplicate lines. It preserves case, stress, and the original `spn`; the training tokenizer filters `spn` during encoding. `dev-other` and `test-other` can be prepared separately from their original Phone69 files. There is no original Phone69 file for `train-other-500`; never include that subset in Phone69 training.

The expected counts for the current experiment are 28,539 train-clean-100, 104,014 train-clean-360, 2,703 dev-clean, and 2,620 test-clean utterances. See [phone69_tokenizer.md](phone69_tokenizer.md) for the label definition.

## Match feature statistics

The original Emformer statistics and the existing server's archived statistics have the same canonical JSON SHA-256, `97837ec2328a5ffc379c3f4d7b6038a637c6cd6519f7eb1d536f1f646c4f5725`. This is a hash of UTF-8 JSON serialized with sorted keys and compact separators, not of the file's whitespace.

The matching file is already tracked at `artifacts/runs/phone69_460h_native_offline_20261018/normalization_stats_server_20261018.json`. The separate `assets/statistics/global_stats.json` contains different values, so do not substitute it when matching the original server.

For a new dataset root without an existing statistics file, link the tracked reference to the configured path:

```bash
ln -s /test21/conformer-rnnt-asr/artifacts/runs/phone69_460h_native_offline_20261018/normalization_stats_server_20261018.json \
  /test21/librispeech/global_stats.json
```

Do not overwrite an existing statistics file without checking its values. Data preparation does not start training; GPU batch budgets must be chosen for the target device when training is explicitly requested.
