import importlib.util
from pathlib import Path
import unittest

METRICS_PATH = Path(__file__).resolve().parents[1] / "conformer_rnnt" / "metrics.py"
SPEC = importlib.util.spec_from_file_location("asr_metrics_under_test", METRICS_PATH)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)
phone_error_counts = METRICS.phone_error_counts


class PhoneErrorRateTests(unittest.TestCase):
    def test_multicharacter_phones_count_as_one(self):
        self.assertEqual(phone_error_counts("CH AY1 L D", "JH AW1 L"), (3, 4))
        self.assertEqual(phone_error_counts("AH0 ER0", "AH1 ER1"), (2, 2))

    def test_insertions_and_empty_references(self):
        self.assertEqual(phone_error_counts("AY1", "AY1 CH"), (1, 1))
        self.assertEqual(phone_error_counts("", "CH AY1"), (2, 0))
        self.assertEqual(phone_error_counts("", ""), (0, 0))




if __name__ == "__main__":
    unittest.main()
