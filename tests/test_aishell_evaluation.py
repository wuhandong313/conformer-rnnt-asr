import copy
import unittest

from conformer_rnnt.aishell.evaluation import (alignment, decile_assignment, sample_metrics,
                                             summarize, validate_predictions)


class AISHELLEvaluationTests(unittest.TestCase):
    def test_dictionary_symbols_are_atomic_and_tone_pairs_conditioned_on_base_match(self):
        result = sample_metrics("zh ong1 g uo2".split(), "zh ong2 g uo2 sil".split())
        self.assertEqual((result["S"], result["D"], result["I"]), (1, 0, 1))
        self.assertEqual((result["toneless_errors"], result["tone_errors"], result["tone_comparable_pairs"]), (1, 1, 2))
        other = sample_metrics(["ong1"], ["uo2"])
        self.assertEqual(other["tone_comparable_pairs"], 0)

    def test_empty_hypothesis_insertions_and_deterministic_tie(self):
        self.assertEqual(sample_metrics(["zh", "ong1"], [])["D"], 2)
        counts, trace = alignment(["a1", "b"], ["b", "a1"])
        self.assertEqual(counts, {"S": 2, "D": 0, "I": 0})
        self.assertEqual([t[0] for t in trace], ["S", "S"])

    def test_corpus_per_is_not_average_sentence_per(self):
        records = [sample_metrics(["a1"], []), sample_metrics(["b"] * 9, ["b"] * 9)]
        self.assertEqual(summarize(records)["PER"], .1)

    def test_stable_ten_deciles_complete_coverage_and_stored_metrics_revalidated(self):
        rows = [{"utterance_id": f"u{i:02}", "phones": ["zh", "ong1"], "num_samples": 16000,
                 "num_frames": 101} for i in range(20)]
        bins = decile_assignment(list(reversed(rows)))
        self.assertEqual([bins[r["utterance_id"]] for r in rows], [i // 2 + 1 for i in range(20)])
        records = [{"utterance_id": r["utterance_id"], "ref": "zh ong1", "hyp": "zh ong1",
                    "decile": bins[r["utterance_id"]], "mel_frames": 101,
                    "duration_seconds": 1., "reference_symbols_per_second": 2.,
                    "pre_compression_frames": 101, "post_compression_frames": 40,
                    **sample_metrics(r["phones"], r["phones"])} for r in rows]
        result = validate_predictions(rows, records)
        self.assertEqual(result["num_samples"], 20)
        self.assertEqual(len(result["deciles"]), 10)
        self.assertNotIn("RTF", result)
        for key, bad in (("S", 1), ("decile", 10), ("ref", "changed"), ("post_compression_frames", 102)):
            altered = copy.deepcopy(records); altered[0][key] = bad
            with self.assertRaises(ValueError):
                validate_predictions(rows, altered)
        with self.assertRaisesRegex(ValueError, "exact frozen"):
            validate_predictions(rows, records[:-1])


if __name__ == "__main__":
    unittest.main()
