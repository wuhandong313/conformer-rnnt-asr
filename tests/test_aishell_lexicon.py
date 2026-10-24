import json
from pathlib import Path
import tempfile
import unittest

from aishell_lexicon_tokenizer import AISHELLLexiconTokenizer
from conformer_rnnt.aishell.lexicon import Lexicon, matches_pinyin


class AISHELLLexiconTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "lexicon.txt"
        self.path.write_text("中国 zh ong1 g uo2\n银行 ii in2 h ang2\n行 h ang2\n行 x ing2\n行 x ing2\n"
                             "而 ee er2\n而 er2\n啊 aa a1\nSIL sil\n", encoding="utf-8")
        self.lex = Lexicon(self.path)

    def select(self, words, prediction, **kwargs):
        return self.lex.select_sentence("sample", words, prediction, split="train", audio_path="wav/train/S1/a.wav", **kwargs)

    def test_original_symbols_and_source_lines_are_preserved(self):
        result = self.select(["银行"], ["yin2", "hang2"])
        self.assertEqual(result["phones"], ["ii", "in2", "h", "ang2"])
        self.assertEqual(result["label_provenance"][0]["source_lines"], [2])

    def test_context_selects_second_candidate_not_first(self):
        result = self.select(["行"], ["xing2"])
        self.assertEqual(result["phones"], ["x", "ing2"])
        self.assertEqual(result["label_provenance"][0]["source_lines"], [4, 5])

    def test_conflict_missing_word_and_prediction_alignment_are_audited(self):
        for words, pred, kind in [(["行"], ["hang4"], "prediction_dictionary_conflict"),
                                  (["缺词"], ["que1", "ci2"], "missing_word"),
                                  (["中国"], ["zhong1"], "prediction_alignment")]:
            result = self.select(words, pred)
            self.assertIsNone(result["phones"])
            self.assertEqual(result["issues"][0]["kind"], kind)
            self.assertFalse(result["issues"][0]["resolved"])

    def test_equivalent_raw_candidates_are_not_arbitrarily_collapsed(self):
        result = self.select(["而"], ["er2"])
        self.assertIsNone(result["phones"])
        issue = result["issues"][0]
        self.assertEqual(issue["kind"], "ambiguous_equivalent_candidates")
        decision = {"issue_id": issue["issue_id"], "candidate_phones": ["er2"], "reason": "verified original candidate", "version": "review1"}
        chosen = self.select(["而"], ["er2"], resolutions={issue["issue_id"]: decision})
        self.assertEqual(chosen["phones"], ["er2"])
        decision["candidate_phones"] = ["NEW"]
        with self.assertRaisesRegex(ValueError, "existing raw"):
            self.select(["而"], ["er2"], resolutions={issue["issue_id"]: decision})

    def test_variable_length_syllables_and_no_fuzzy_matching(self):
        self.assertTrue(matches_pinyin("s iy4 b ai3 er4 sh ix2 s an1".split(), "si4 bai3 er4 shi2 san1".split()))
        self.assertTrue(matches_pinyin("vv ve4 uu ueng1 n v3".split(), "yue4 weng1 nv3".split()))
        self.assertFalse(matches_pinyin(["h", "ang2"], ["xing2"]))
        self.assertFalse(matches_pinyin(["h", "ang2"], ["hang"]))

    def test_issue_identity_includes_utterance_context(self):
        a = self.select(["行"], ["hang4"])["issues"][0]["issue_id"]
        b = self.select(["中国", "行"], ["zhong1", "guo2", "hang4"])["issues"][0]["issue_id"]
        self.assertNotEqual(a, b)

    def test_special_silence_is_explicit_and_not_automatically_inserted(self):
        self.assertEqual(self.select(["SIL"], ["S", "I", "L"])["phones"], ["sil"])
        self.assertNotIn("sil", self.select(["中国"], ["zhong1", "guo2"])["phones"])

    def test_fixed_tokenizer_keeps_tones_and_sil_but_removes_blank(self):
        tokenizer = AISHELLLexiconTokenizer()
        self.assertEqual((tokenizer.get_piece_size(), tokenizer.blank_id), (217, 217))
        raw = "aa a5 ii in2 h ang2 sil"
        ids = tokenizer.encode(raw)
        self.assertEqual(tokenizer.decode([217, *ids, 217]), raw)
        for invalid in ["zhong1", "AH0", "ang", "<blank>"]:
            with self.assertRaises(ValueError):
                tokenizer.encode(invalid)
        self.assertNotEqual(tokenizer.encode("ang2"), tokenizer.encode("ang4"))


if __name__ == "__main__":
    unittest.main()
