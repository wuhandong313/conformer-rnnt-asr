import unittest

from ipa_phonemizer import WordPhonemizer
from ipa_tokenizer import FixedIPATokenizer


class FakeG2P:
    cmu = {"child": [["CH", "AY1", "L", "D"]],
           "a": [["AH0"], ["EY1"]], "her": [["HH", "ER0"]]}

    def __init__(self):
        self.calls = []

    def __call__(self, word):
        self.calls.append(word)
        return ["Z", "IY1", " ", "."]


class PhonemizerTests(unittest.TestCase):
    def test_dictionary_first_pronunciation_and_cached_oov(self):
        backend = FakeG2P()
        phonemizer = WordPhonemizer(FixedIPATokenizer(), backend)
        self.assertEqual(phonemizer.arpabet("A CHILD"), "AH0 CH AY1 L D")
        self.assertEqual(phonemizer.arpabet("ZEE ZEE"), "Z IY1 Z IY1")
        self.assertEqual(backend.calls, ["zee"])
        self.assertEqual(phonemizer.audit()["oov_words"], ["zee"])

    def test_unknown_backend_phone_is_not_silently_removed(self):
        backend = FakeG2P()
        backend.cmu = {"bad": [["SPN"]]}
        phonemizer = WordPhonemizer(FixedIPATokenizer(), backend)
        with self.assertRaisesRegex(ValueError, "SPN"):
            phonemizer.arpabet("bad")


if __name__ == "__main__":
    unittest.main()
