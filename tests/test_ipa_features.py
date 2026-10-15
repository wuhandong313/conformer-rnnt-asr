import json
import tempfile
import unittest
from pathlib import Path

from ipa_features import DEFAULT_FEATURES, IPAFeatureTable
from ipa_tokenizer import FixedIPATokenizer


class IPAFeatureTests(unittest.TestCase):
    def test_all_phones_have_24_features_and_blank_is_masked(self):
        tokenizer = FixedIPATokenizer()
        table = IPAFeatureTable(tokenizer)
        vectors, mask = table.lookup(range(39))
        self.assertEqual(len(vectors), 39)
        self.assertTrue(all(len(row) == 24 for row in vectors))
        self.assertEqual(mask, [True] * 38 + [False])
        self.assertEqual(vectors[-1], [0] * 24)

    def test_syllabic_r_retains_its_physical_distinction(self):
        tokenizer = FixedIPATokenizer()
        table = IPAFeatureTable(tokenizer)
        vectors, _ = table.lookup(tokenizer.encode("ɹ ɹ̩"))
        changed = [name for name, left, right in zip(table.names, *vectors) if left != right]
        self.assertEqual(changed, ["syl"])
        self.assertEqual([row[table.names.index("syl")] for row in vectors], [-1, 1])

    def test_table_fingerprint_and_row_order_are_checked(self):
        spec = json.loads(DEFAULT_FEATURES.read_text())
        spec["tokens"][0], spec["tokens"][1] = spec["tokens"][1], spec["tokens"][0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "token IDs"):
                IPAFeatureTable(path=path)


if __name__ == "__main__":
    unittest.main()
