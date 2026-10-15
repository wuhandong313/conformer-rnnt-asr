"""Runtime checks; require the same Torch/torchaudio/Lightning stack as training."""

import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from common import Batch, post_process_hypos
    from conformer_rnnt.lightning_module import CustomDataset, LIBRISPEECH, LibriSpeechRNNTModule
    from eval import run_eval
    from ipa_tokenizer import FixedIPATokenizer
    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    RUNTIME_AVAILABLE = False


@unittest.skipUnless(RUNTIME_AVAILABLE, "Requires the training runtime")
class IPAIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stats = self.root / "stats.json"
        stats.write_text(json.dumps({"mean": [0.0] * 8, "invstddev": [1.0] * 8}))
        self.kwargs = dict(
            librispeech_path=str(self.root), global_stats_path=str(stats),
            tokenizer_type="ipa", strip_spaces=False, transcript_suffix="-ipa",
            num_symbols=41, input_dim=8, n_mels=8, encoding_dim=8,
            time_reduction_input_dim=8, num_heads=2, ffn_dim=16,
            num_layers_block1=1, num_layers_block2=1,
            depthwise_conv_kernel_size=3, compression_filter="none",
            dropout=0.0, symbol_embedding_dim=8, num_lstm_layers=1,
            lstm_hidden_dim=8, lstm_dropout=0.0, num_workers=0,
            beam_width=0,
        )

    def test_ipa_model_never_imports_sentencepiece_and_loss_has_gradients(self):
        original_import = builtins.__import__

        def reject_spm(name, *args, **kwargs):
            if name == "sentencepiece" or name.startswith("sentencepiece."):
                raise AssertionError("IPA must not import SentencePiece")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_spm):
            model = LibriSpeechRNNTModule(**self.kwargs)
            targets, lengths = model._extract_labels([
                (None, 16000, "tʃ aɪ l d"), (None, 16000, "ɑ ə"),
            ])
            self.assertEqual(lengths.tolist(), [4, 2])
            self.assertEqual(targets[1].tolist(), [0, 39, 41, 41])
            batch = Batch(torch.randn(2, 36, 8), torch.tensor([32, 24], dtype=torch.int32),
                          targets, lengths)
            with patch.object(model, "log"):
                loss = model._step(batch, 0, "train")
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.model.parameters()))

    def test_checkpoint_rejects_word_or_reordered_ipa_inventory(self):
        model = LibriSpeechRNNTModule(**self.kwargs)
        checkpoint = {}
        model.on_save_checkpoint(checkpoint)
        model.on_load_checkpoint(checkpoint)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            model.on_load_checkpoint({})
        checkpoint["ipa_tokenizer"]["fingerprint"] = "wrong inventory"
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            model.on_load_checkpoint(checkpoint)

    def test_decoding_keeps_phone_zero_and_removes_only_blank(self):
        tokenizer = FixedIPATokenizer()
        hypothesis = ([41, 0, 7, 5, 41], None, None, 0.0)
        text, _, ids = post_process_hypos([hypothesis], tokenizer)[0]
        self.assertEqual(text, "ɑ tʃ aɪ")
        self.assertEqual(ids, [0, 7, 5])

    def test_batch_budget_uses_phone_counts(self):
        chapter = self.root / "LibriSpeech/train-clean-100/1/2"
        chapter.mkdir(parents=True)
        (chapter / "1-2-ipa.trans.txt").write_text("1-2-0000 tʃ aɪ l d\n1-2-0001 ə ɚ\n")
        for uid in ("1-2-0000", "1-2-0001"):
            (chapter / f"{uid}.flac").touch()
        dataset = CustomDataset(LIBRISPEECH(self.root, transcript_suffix="-ipa"),
                                4, sp_model=FixedIPATokenizer(), strip_spaces=False,
                                rnnt_vocab=42)
        self.assertEqual(dataset.batches, [[0], [1]])

    def test_per_is_corpus_weighted_in_eval(self):
        class FakeModel(torch.nn.Module):
            tokenizer_type = "ipa"
            tokenizer = FixedIPATokenizer()
            strip_spaces = False
            _mel_kwargs = {"hop_length": 160, "sample_rate": 16000}

            def __init__(self):
                super().__init__()
                self.predictions = iter(["", "aɪ l d"])

            def test_dataloader(self):
                batch = Batch(None, torch.tensor([100]), None, None)
                return [(batch, ["tʃ"]), (batch, ["aɪ l d"])]

            def forward(self, batch):
                return next(self.predictions)

        result = run_eval(FakeModel(), torch.device("cpu"),
                          str(self.root / "per.csv"), metric="per")
        self.assertEqual(result["phone_errors"], 1)
        self.assertEqual(result["reference_phones"], 4)
        self.assertEqual(result["per"], 0.25)
        self.assertIn("reference_phones", (self.root / "per.csv").read_text())
        with self.assertRaisesRegex(ValueError, "PER"):
            run_eval(FakeModel(), torch.device("cpu"), metric="wer")


if __name__ == "__main__":
    unittest.main()
