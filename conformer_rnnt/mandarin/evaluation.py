"""Corpus and per-dataset tone-marked syllable edits, shared rate deciles."""
from collections import defaultdict

from conformer_rnnt.aishell.evaluation import sample_metrics as phone_metrics
from conformer_rnnt.aishell.preparation import distribution


def sample_metrics(reference, hypothesis):
    original = phone_metrics(reference, hypothesis)
    names = {"phone_errors": "syllable_errors", "reference_phones": "reference_syllables",
             "PER": "syllable_error_rate", "toneless_reference_phones": "toneless_reference_syllables"}
    return {names.get(key, key): value for key, value in original.items()}


def decile_assignment(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["corpus"]].append(row)
    result = {}
    for corpus, items in groups.items():
        ordered = sorted(items, key=lambda r: (len(r["syllables"]) / r["duration_seconds"], r["utterance_id"]))
        for i, row in enumerate(ordered):
            if row["utterance_id"] in result:
                raise ValueError("Duplicate reference ID")
            result[row["utterance_id"]] = i * 10 // len(ordered) + 1
    return result


def summarize(records):
    if not records:
        raise ValueError("No syllable evaluation records")
    totals = {key: sum(r[key] for r in records) for key in
              ("S", "D", "I", "syllable_errors", "reference_syllables", "toneless_errors", "tone_errors", "tone_comparable_pairs")}
    if totals["S"] + totals["D"] + totals["I"] != totals["syllable_errors"]:
        raise ValueError("Syllable edit totals disagree")
    denominator = totals["reference_syllables"]
    ratios = {op + "_per_reference_syllable": totals[op] / denominator for op in ("S", "D", "I")}
    pre, post = sum(r["pre_compression_frames"] for r in records), sum(r["post_compression_frames"] for r in records)
    return {"num_samples": len(records), **totals, **ratios,
            "syllable_error_rate": totals["syllable_errors"] / denominator,
            "toneless_syllable_error_rate": totals["toneless_errors"] / denominator,
            "conditional_tone_error_rate": totals["tone_errors"] / totals["tone_comparable_pairs"] if totals["tone_comparable_pairs"] else None,
            "observed_compression": {"input_frames": pre, "output_frames": post, "retained_fraction": post / pre,
                                     "input_distribution": distribution([r["pre_compression_frames"] for r in records]),
                                     "output_distribution": distribution([r["post_compression_frames"] for r in records])}}


def validate_predictions(rows, records, *, decile_rows=None):
    if [r["utterance_id"] for r in rows] != [r["utterance_id"] for r in records]:
        raise ValueError("Predictions must cover the exact frozen combined test sequence")
    bins = decile_assignment(rows if decile_rows is None else decile_rows)
    for source, result in zip(rows, records):
        if result["ref"].split() != source["syllables"] or result["corpus"] != source["corpus"]:
            raise ValueError("Reference syllables/corpus changed")
        expected = sample_metrics(source["syllables"], result["hyp"].split())
        if any(result[key] != value for key, value in expected.items()):
            raise ValueError("Stored syllable metrics disagree with independent alignment")
        if result["decile"] != bins[source["utterance_id"]]:
            raise ValueError("Per-corpus speech-rate bin changed")
        if (result["duration_seconds"] != source["duration_seconds"] or result["mel_frames"] != source["num_frames"]
                or result["reference_symbols_per_second"] != len(source["syllables"]) / source["duration_seconds"]
                or not 0 < result["post_compression_frames"] <= result["pre_compression_frames"] <= result["mel_frames"]):
            raise ValueError("Invalid measured duration/frame metadata")
    summary = summarize(records)
    summary.update(target_type="tone_marked_syllables", metric="corpus_syllable_error_rate", rtf_reported=False,
                   rate_deciles="separate_per_corpus_from_frozen_reference_syllables_and_original_duration")
    summary["by_corpus"] = {}
    for corpus in sorted({r["corpus"] for r in records}):
        selected = [r for r in records if r["corpus"] == corpus]
        item = summarize(selected)
        item["deciles"] = {str(d): summarize([r for r in selected if r["decile"] == d])
                           for d in range(1, 11) if any(r["decile"] == d for r in selected)}
        summary["by_corpus"][corpus] = item
    return summary
