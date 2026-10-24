"""Case-sensitive dictionary-token edits and shared speech-rate deciles; no RTF."""
from __future__ import annotations

import math
import re


def base_phone(phone):
    return re.sub(r"[1-5]$", "", phone)


def alignment(reference, hypothesis):
    ref, hyp = list(reference), list(hypothesis)
    dp = [list(range(len(hyp) + 1))]
    for i, a in enumerate(ref, 1):
        row = [i]
        for j, b in enumerate(hyp, 1):
            row.append(min(dp[-1][j] + 1, row[-1] + 1, dp[-1][j - 1] + (a != b)))
        dp.append(row)
    i, j, trace = len(ref), len(hyp), []
    while i or j:
        if i and j and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            trace.append(("=", i - 1, j - 1)); i -= 1; j -= 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            trace.append(("S", i - 1, j - 1)); i -= 1; j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            trace.append(("D", i - 1, None)); i -= 1
        else:
            trace.append(("I", None, j - 1)); j -= 1
    trace.reverse()
    counts = {op: sum(t[0] == op for t in trace) for op in ("S", "D", "I")}
    if sum(counts.values()) != dp[-1][-1]:
        raise ValueError("Edit traceback disagrees with minimum distance")
    return counts, trace


def sample_metrics(reference, hypothesis):
    reference, hypothesis = list(reference), list(hypothesis)
    if not reference:
        raise ValueError("Empty reference cannot define PER")
    counts, _ = alignment(reference, hypothesis)
    stripped, trace = alignment([base_phone(p) for p in reference], [base_phone(p) for p in hypothesis])
    tone_pairs = [(reference[i], hypothesis[j]) for op, i, j in trace
                  if op == "=" and re.search(r"[1-5]$", reference[i]) and re.search(r"[1-5]$", hypothesis[j])]
    errors = sum(counts.values())
    return {**counts, "phone_errors": errors, "reference_phones": len(reference), "PER": errors / len(reference),
            "toneless_errors": sum(stripped.values()), "toneless_reference_phones": len(reference),
            "tone_errors": sum(a[-1] != b[-1] for a, b in tone_pairs), "tone_comparable_pairs": len(tone_pairs)}


def decile_assignment(rows):
    ordered = sorted(rows, key=lambda r: (len(r["phones"]) * 16000 / r["num_samples"], r["utterance_id"]))
    if len({r["utterance_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated reference ID")
    return {r["utterance_id"]: min(9, index * 10 // len(ordered)) + 1 for index, r in enumerate(ordered)}


def summarize(records):
    if not records:
        raise ValueError("No evaluation samples")
    totals = {k: sum(r[k] for r in records) for k in
              ("S", "D", "I", "phone_errors", "reference_phones", "toneless_errors", "tone_errors", "tone_comparable_pairs")}
    if totals["S"] + totals["D"] + totals["I"] != totals["phone_errors"]:
        raise ValueError("S/D/I totals disagree")
    n = totals["reference_phones"]
    ratios = {op + "_per_reference_phone": totals[op] / n for op in ("S", "D", "I")}
    return {"num_samples": len(records), **totals, "PER": totals["phone_errors"] / n, **ratios,
            "toneless_PER": totals["toneless_errors"] / n,
            "conditional_tone_error_rate": (totals["tone_errors"] / totals["tone_comparable_pairs"]
                                            if totals["tone_comparable_pairs"] else None)}


def validate_predictions(rows, records, *, decile_rows=None):
    if [r["utterance_id"] for r in rows] != [r["utterance_id"] for r in records]:
        raise ValueError("Predictions must cover the exact frozen test sequence once")
    deciles = decile_assignment(rows if decile_rows is None else decile_rows)
    for source, result in zip(rows, records):
        if result["ref"].split() != source["phones"]:
            raise ValueError("Evaluation reference differs from the frozen label")
        expected = sample_metrics(source["phones"], result["hyp"].split())
        if any(result[k] != value for k, value in expected.items()):
            raise ValueError("Stored sample metrics disagree with independent re-alignment")
        if result["decile"] != deciles[source["utterance_id"]]:
            raise ValueError("Speech-rate bin assignment changed")
        if any(not isinstance(result[k], int) or result[k] <= 0
               for k in ("mel_frames", "pre_compression_frames", "post_compression_frames")):
            raise ValueError("Invalid observed compression length")
        if result["post_compression_frames"] > result["pre_compression_frames"]:
            raise ValueError("Compression output exceeds its input length")
        if (result["mel_frames"] != source["num_frames"]
                or result["pre_compression_frames"] > source["num_frames"]
                or result["duration_seconds"] != source["num_samples"] / 16000
                or result["reference_symbols_per_second"] != len(source["phones"]) * 16000 / source["num_samples"]):
            raise ValueError("Observed audio/frame/rate metadata differs from frozen manifest")
    summary = summarize(records)
    summary["deciles"] = {str(d): summarize([r for r in records if r["decile"] == d])
                          for d in range(1, 11) if any(r["decile"] == d for r in records)}
    inputs = sum(r["pre_compression_frames"] for r in records)
    outputs = sum(r["post_compression_frames"] for r in records)
    summary["observed_compression"] = {"input_frames": inputs, "output_frames": outputs,
                                         "retained_fraction": outputs / inputs,
                                         "meaning": "observed lengths, never a target keep rate"}
    from .preparation import distribution
    summary["observed_compression"].update(
        input_distribution=distribution([r["pre_compression_frames"] for r in records]),
        output_distribution=distribution([r["post_compression_frames"] for r in records]),
        fraction_distribution=distribution([r["post_compression_frames"] / r["pre_compression_frames"] for r in records]))
    return summary
