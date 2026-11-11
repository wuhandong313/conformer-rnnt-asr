"""Phone-level metrics used by ASR evaluation."""

from __future__ import annotations






def phone_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Phone edit count and reference count, preserving case and stress."""
    ref = reference.split()
    hyp = hypothesis.split()
    previous = list(range(len(hyp) + 1))
    for i, phone in enumerate(ref, 1):
        row = [i]
        for j, predicted in enumerate(hyp, 1):
            row.append(min(row[-1] + 1, previous[j] + 1,
                           previous[j - 1] + (phone != predicted)))
        previous = row
    return previous[-1], len(ref)
