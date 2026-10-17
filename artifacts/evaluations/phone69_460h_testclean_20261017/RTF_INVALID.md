# Original RTF values are invalid

The original eval.py used time.time() and recorded negative elapsed durations during wall-clock rollback. Do not use the original RTF means. PER and S/D/I have been independently validated. RTF is being remeasured in rtf_monotonic/ with time.perf_counter(), the same checkpoints, configurations, test-clean set and beam width. Every prediction is compared against the original run. Original CSV and logs are preserved.
