import importlib.util
from pathlib import Path
import unittest

METRICS_PATH = Path(__file__).resolve().parents[1] / "conformer_rnnt" / "metrics.py"
SPEC = importlib.util.spec_from_file_location("asr_metrics_under_test", METRICS_PATH)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)
compute_wer = METRICS.compute_wer
word_error_counts = METRICS.word_error_counts
format_wer_trace = METRICS.format_wer_trace
phone_error_counts = METRICS.phone_error_counts


class PhoneErrorRateTests(unittest.TestCase):
    def test_multicharacter_phones_count_as_one(self):
        self.assertEqual(phone_error_counts("tʃ aɪ l d", "dʒ aʊ l"), (3, 4))
        self.assertEqual(phone_error_counts("ə ɚ", "ʌ ɝ"), (2, 2))

    def test_insertions_and_empty_references(self):
        self.assertEqual(phone_error_counts("aɪ", "aɪ tʃ"), (1, 1))
        self.assertEqual(phone_error_counts("", "tʃ aɪ"), (2, 0))
        self.assertEqual(phone_error_counts("", ""), (0, 0))


class WordErrorRateTests(unittest.TestCase):
    def test_case_and_whitespace_are_normalized(self):
        self.assertEqual(compute_wer("  THE   cat\nsat ", "the cat sat"), 0.0)

    def test_substitution_and_insertion_are_counted(self):
        errors, reference_words = word_error_counts(
            "the cat sat", "the dog sat down"
        )
        self.assertEqual((errors, reference_words), (2, 3))
        self.assertAlmostEqual(compute_wer("the cat sat", "the dog sat down"), 2 / 3)

    def test_deletion_is_counted(self):
        self.assertEqual(compute_wer("the cat sat", "the sat"), 1 / 3)

    def test_trace_alignment_agrees_with_word_edit_count(self):
        trace = format_wer_trace("the cat sat", "the dog sat down")
        self.assertIn("S=1, D=0, I=1, errors=2, reference_words=3", trace)
        self.assertIn("  S  'cat' -> 'dog'", trace)
        self.assertIn("  I  '<empty>' -> 'down'", trace)
        self.assertIn("S=0, D=1, I=0", format_wer_trace("the cat sat", "the sat"))


if __name__ == "__main__":
    unittest.main()
