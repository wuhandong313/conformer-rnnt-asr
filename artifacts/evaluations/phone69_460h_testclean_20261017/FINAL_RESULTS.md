# Phone69 460h: final test-clean evaluation

All three models use their recorded best checkpoint, fixed phone69, beam width 20, and one A100 80 GiB GPU. Each evaluation contains 2,620 utterances and 189,023 reference phones.

| Model | Corpus PER | Mean utterance RTF | Substitutions S | Deletions D | Insertions I |
| --- | ---: | ---: | ---: | ---: | ---: |
| stride4_none | 5.5173% | 0.1358 | 7,210 (3.8144%) | 2,473 (1.3083%) | 746 (0.3947%) |
| stride1_cosd_learned | 5.7559% | 0.1615 | 7,593 (4.0170%) | 2,024 (1.0708%) | 1,263 (0.6682%) |
| stride4_cosd_learned | 9.0328% | 0.0625 | 10,116 (5.3517%) | 5,920 (3.1319%) | 1,038 (0.5491%) |

S/D/I percentages use the number of reference phones as their denominator. PER = (S + D + I) / reference phones.

RTF is the arithmetic mean of per-utterance inference-time/audio-duration ratios, measured with time.perf_counter(). It is not a duration-weighted corpus RTF. Feature extraction/data loading are outside the existing evaluator timing scope.

The original wall-clock RTF values were invalid because they included negative elapsed times. Original CSVs and logs are preserved in this directory for audit. The independent monotonic remeasurement used the same inputs, checkpoints and decoding configuration; all 7,860 hypotheses and phone-error counts match the original run exactly.

Validation checks ordered unique sample IDs, identical reference sets, preserved raw CSV columns, independently recomputed phone edit distances, every S+D+I sum, corpus aggregates, finite nonnegative RTFs, configuration hashes and checkpoint identity.

The final validator originally had a circular state prerequisite. validate_results_recheck.py removes only that prerequisite; the failed log, original script and repair note are retained. All result checks passed without additional model runs.

- Final summary: [rtf_monotonic/summary.json](rtf_monotonic/summary.json)
- Validation: [rtf_monotonic/validation_report.json](rtf_monotonic/validation_report.json)
- Prediction comparison: [rtf_monotonic/prediction_comparison.json](rtf_monotonic/prediction_comparison.json)
- Per-utterance CSVs, logs, matching config snapshots and measurement scripts are in rtf_monotonic/.
