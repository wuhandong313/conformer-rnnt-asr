"""Exercise phone69 training, batching, decoding, checkpoint and PER together."""

import builtins
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from common import Batch, post_process_hypos
    from conformer_rnnt.lightning_module import CustomDataset, LIBRISPEECH, LibriSpeechRNNTModule
    from eval import run_eval
    from phone69_tokenizer import FixedPhone69Tokenizer
    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    RUNTIME_AVAILABLE = False


@unittest.skipUnless(RUNTIME_AVAILABLE, "Requires the training runtime")
class Phone69IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stats = self.root / "stats.json"
        stats.write_text(json.dumps({"mean": [0.0] * 8, "invstddev": [1.0] * 8}))
        self.kwargs = dict(
            librispeech_path=str(self.root), global_stats_path=str(stats),
            tokenizer_type="phone69", strip_spaces=False, num_symbols=69,
            input_dim=8, n_mels=8, encoding_dim=8, time_reduction_input_dim=8,
            num_heads=2, ffn_dim=16, num_layers_block1=1, num_layers_block2=1,
            depthwise_conv_kernel_size=3, compression_filter="none", dropout=0.0,
            symbol_embedding_dim=8, num_lstm_layers=1, lstm_hidden_dim=8,
            lstm_dropout=0.0, num_workers=0, beam_width=0,
        )

    def test_training_and_decoding_without_spm(self):
        original_import = builtins.__import__

        def reject_spm(name, *args, **kwargs):
            if name == "sentencepiece" or name.startswith("sentencepiece."):
                raise AssertionError("phone69 must not import SentencePiece")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_spm):
            model = LibriSpeechRNNTModule(**self.kwargs)
            targets, lengths = model._extract_labels([
                (None, 16000, "CH AE1 P T ER0"), (None, 16000, "AA0 spn"),
            ])
            self.assertEqual(lengths.tolist(), [5, 1])
            self.assertEqual(targets[1].tolist(), [0, 69, 69, 69, 69])
            batch = Batch(torch.randn(2, 36, 8), torch.tensor([32, 24], dtype=torch.int32), targets, lengths)
            with patch.object(model, "log"):
                loss = model._step(batch, 0, "train")
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.parameters()))
            decoded = post_process_hypos([([69, 0, 1, 69], None, None, 0.0)], model.tokenizer)[0]
            self.assertEqual((decoded[0], decoded[2]), ("AA0 AA1", [0, 1]))

    def test_checkpoint_and_config_reject_old_spm(self):
        model = LibriSpeechRNNTModule(**self.kwargs)
        ckpt = {}
        model.on_save_checkpoint(ckpt)
        model.on_load_checkpoint(ckpt)
        for other in ({}, {"ipa_tokenizer": ckpt["phone69_tokenizer"]},
                      {"phone69_tokenizer": {"fingerprint": "wrong IDs"}}):
            with self.assertRaisesRegex(ValueError, "phone69 vocabulary"):
                model.on_load_checkpoint(other)
        with self.assertRaisesRegex(ValueError, "strip_spaces"):
            LibriSpeechRNNTModule(**{**self.kwargs, "strip_spaces": True})
        with self.assertRaisesRegex(ValueError, "fixed tokenizer"):
            LibriSpeechRNNTModule(**{**self.kwargs, "tokenizer_type": "sentencepiece", "sp_model_path": "old.model"})
        with self.assertRaisesRegex(ValueError, "does not match"):
            LibriSpeechRNNTModule(**{**self.kwargs, "num_symbols": 74})

    def test_phone_batch_budget_and_wrong_label_convention(self):
        chapter = self.root / "LibriSpeech/train-clean-100/1/2"
        chapter.mkdir(parents=True)
        source = chapter / "1-2-ph.trans.txt"
        source.write_text("1-2-0000 CH AE1 P T ER0\n1-2-0001 AH0 spn\n")
        for uid in ("1-2-0000", "1-2-0001"):
            (chapter / f"{uid}.flac").touch()
        base = LIBRISPEECH(self.root)
        dataset = CustomDataset(base, 5, sp_model=FixedPhone69Tokenizer(), strip_spaces=False, rnnt_vocab=70)
        self.assertEqual(dataset.batches, [[0], [1]])
        self.assertEqual(base.get_metadata(0)[2], "CH AE1 P T ER0")
        source.write_text("1-2-0000 t͡ʃ æ p\n1-2-0001 ə\n")
        with self.assertRaisesRegex(ValueError, "Unknown phone69"):
            CustomDataset(base, 5, sp_model=FixedPhone69Tokenizer(), strip_spaces=False)

    def test_per_counts_stress_substitution_as_one_phone(self):
        class FakeModel(torch.nn.Module):
            tokenizer_type = "phone69"
            tokenizer = FixedPhone69Tokenizer()
            strip_spaces = False
            _mel_kwargs = {"hop_length": 160, "sample_rate": 16000}

            def test_dataloader(self):
                return [(Batch(None, torch.tensor([100]), None, None), ["CH AH0 spn"])]

            def forward(self, batch):
                return "CH AH1"

        result = run_eval(FakeModel(), torch.device("cpu"), str(self.root / "per.csv"), metric="per")
        self.assertEqual((result["phone_errors"], result["reference_phones"]), (1, 2))
        self.assertAlmostEqual(result["per"], 1 / 2)

    def test_word_bpe_still_loads_its_own_spm(self):
        class WordTokenizer:
            def __init__(self, model_file):
                self.model_file = model_file

            def get_piece_size(self):
                return 4000

        with patch.dict("sys.modules", {"sentencepiece": types.SimpleNamespace(SentencePieceProcessor=WordTokenizer)}):
            model = LibriSpeechRNNTModule(**{
                **self.kwargs, "tokenizer_type": "sentencepiece", "transcript_suffix": "",
                "sp_model_path": "word.model", "num_symbols": 4000,
            })
        self.assertEqual(model.tokenizer.model_file, "word.model")
        self.assertEqual((model.blank_idx, model.padding_idx), (4000, 1))


if __name__ == "__main__":
    unittest.main()
