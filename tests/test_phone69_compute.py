import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import nn

from conformer_rnnt.compression.natural import NaturalPeakCompression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/evaluation"))
from measure_phone69_460h_compute import COMPONENTS, observe, validate_result
import run_phone69_460h_cpu_batch as timing


class ComputeCounterTest(unittest.TestCase):
    def test_snapshot_and_live_timing_runner_are_fingerprinted_separately(self):
        key = str(Path(timing.__file__).resolve().relative_to(timing.ROOT))
        with TemporaryDirectory() as tmp:
            source = Path(tmp)
            for name in ("common.py", "phone69_tokenizer.py", key):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Historical snapshot\n")
            archived = timing.code_fingerprints(source)
            live = timing.execution_fingerprints(source)
            self.assertEqual(archived[key], timing.sha(source / key))
            self.assertEqual(live[key], timing.sha(Path(timing.__file__)))
            self.assertNotEqual(archived[key], live[key])

    def test_fusion_counts_valid_frames_and_the_single_frame_bypass(self):
        encoder = nn.Module()
        encoder.input_linear = nn.Linear(4, 4)
        encoder.output_linear = nn.Linear(4, 4)
        encoder.compression = NaturalPeakCompression(
            detector={"type": "cosine_diff"}, mixer={"type": "learned"},
            feature_dim=4, window_size=5, weight_predictor_mode="five_frames",
        )
        network = SimpleNamespace(transcriber=encoder, predictor=nn.Linear(4, 4),
                                  joiner=SimpleNamespace(linear=nn.Linear(4, 4)))
        counts = Counter()
        observe(SimpleNamespace(model=network), counts, [])
        frames = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4)
        encoder.compression(frames, torch.tensor([1]))
        self.assertEqual(sum(counts.values()), 0)
        _, lengths = encoder.compression(frames, torch.tensor([4]))
        outputs = lengths.item()
        self.assertEqual(counts["detector_dots"], 3 * 4)
        self.assertEqual(counts["fusion_weighted_sum"], outputs * 5 * 4)
        self.assertEqual(counts["fusion_mlp"], outputs * (5 * 4 * 32 + 32 * 5))

    def test_cached_result_requires_complete_traces_and_consistent_sums(self):
        entry = dict(id="E09", config_sha256="cfg", checkpoint_sha256="weights")
        audio = [dict(uid="a", samples=16000, sample_rate=16000)]
        counts = dict(encoder_weights=3, predictor=2, joiner=1)
        row = dict(idx=0, uid="a", layer_frames=[1] * 10, output_steps=1, predictor_tokens=1,
                   joiner_rows=1, joiner_calls=1, **counts)
        result = dict(**entry, rows=[row], utterances=1, predictions_checked=1, audio_seconds=1,
                      encoder_output_steps=1, predictor_tokens=1, joiner_rows=1, joiner_calls=1,
                      components_macs={k: counts.get(k, 0) for k in COMPONENTS}, total_major_macs=6,
                      gmac_per_audio_second=6e-9, provenance={})
        validate_result(result, entry, audio, [0], {})
        for change in (dict(rows=[]), dict(rows=[dict(row, uid="b")]), dict(total_major_macs=5)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_result(dict(result, **change), entry, audio, [0], {})


if __name__ == "__main__":
    unittest.main()
