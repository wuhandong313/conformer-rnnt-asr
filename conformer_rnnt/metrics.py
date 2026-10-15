"""Text-level metrics used by ASR evaluation."""

from __future__ import annotations


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return word edit count and reference word count.

    Text is lowercased and tokenized on whitespace. Punctuation remains part
    of a token, matching the transcript text supplied to the evaluator.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Levenshtein distance with O(len(hyp_words)) memory.
    previous = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp_words, start=1):
            substitution = previous[j - 1] + (ref_word != hyp_word)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current

    return previous[-1], len(ref_words)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate using case-insensitive whitespace tokens."""
    errors, reference_words = word_error_counts(reference, hypothesis)
    if reference_words == 0:
        if errors == 0:
            return 0.0
        raise ValueError("WER is undefined when the reference has no words")
    return errors / reference_words


def phone_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Phone edit count and reference phone count, preserving multi-codepoint IPA."""
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


def format_wer_trace(reference: str, hypothesis: str) -> str:
    """Show tokenization, DP rows, and one minimum-cost word alignment."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    dp = [list(range(len(hyp_words) + 1))]
    for i, ref_word in enumerate(ref_words, start=1):
        row = [i]
        for j, hyp_word in enumerate(hyp_words, start=1):
            row.append(min(
                dp[i - 1][j - 1] + (ref_word != hyp_word),
                row[j - 1] + 1,
                dp[i - 1][j] + 1,
            ))
        dp.append(row)

    alignment = []
    i, j = len(ref_words), len(hyp_words)
    while i or j:
        if i and j and ref_words[i - 1] == hyp_words[j - 1]:
            alignment.append(("=", ref_words[i - 1], hyp_words[j - 1]))
            i, j = i - 1, j - 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append(("S", ref_words[i - 1], hyp_words[j - 1]))
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append(("D", ref_words[i - 1], "<empty>"))
            i -= 1
        else:
            alignment.append(("I", "<empty>", hyp_words[j - 1]))
            j -= 1
    alignment.reverse()

    counts = {op: sum(a[0] == op for a in alignment) for op in ("S", "D", "I")}
    errors, reference_words = word_error_counts(reference, hypothesis)
    assert dp[-1][-1] == errors == sum(counts.values())
    rate = errors / reference_words if reference_words else (0.0 if errors == 0 else float("inf"))
    lines = [
        f"reference_raw = {reference!r}",
        f"hypothesis_raw = {hypothesis!r}",
        f"reference.lower().split() = {ref_words!r}",
        f"hypothesis.lower().split() = {hyp_words!r}",
        "DP rows: columns are hypothesis prefixes; rows are reference prefixes",
        f"DP[0] (<empty>) = {dp[0]}",
    ]
    lines.extend(f"DP[{k}] ({ref_words[k - 1]!r}) = {dp[k]}" for k in range(1, len(dp)))
    lines.append("Minimum-cost alignment (= match, S substitute, D delete, I insert):")
    lines.extend(f"  {op}  {ref!r} -> {hyp!r}" for op, ref, hyp in alignment)
    lines.append(
        f"S={counts['S']}, D={counts['D']}, I={counts['I']}, "
        f"errors={errors}, reference_words={reference_words}"
    )
    lines.append(f"sample WER = {errors}/{reference_words} = {rate:.8f} ({rate:.4%})")
    return "\n".join(lines)
