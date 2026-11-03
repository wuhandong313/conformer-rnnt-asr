"""A/B the smoothing alpha on one checkpoint, everything else fixed.

Runs the same encoder twice per utterance: alpha=0 (plain cosine diff) and
alpha=0.25. Reports aggregate retention plus the per-utterance frame-count
delta distribution, so a null result cannot come from aggregation.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CODE = Path('/Users/wuhandong/Desktop/Projects/conformer-rnnt-asr/output/local_cpu_20261103/code/feature_smooth')
os.chdir(CODE)
sys.path.insert(0, str(CODE))
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')

import torch

from conformer_rnnt.lightning_module import LibriSpeechRNNTModule

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

config_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
reference_csv = Path(sys.argv[3])
alphas = [0.0, 0.25]

cfg = yaml.safe_load(config_path.read_text())
model = LibriSpeechRNNTModule.from_config(cfg)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.on_load_checkpoint(checkpoint)
model.load_state_dict(checkpoint['state_dict'], strict=True)
del checkpoint
model.eval()

encoder = model.model.transcriber
detector = encoder.compression.detector
assert detector.__class__.__name__ == 'CosineDiffPeaksDetector'
with reference_csv.open(newline='') as stream:
    references = [(int(row['idx']), row['ref']) for row in csv.DictReader(stream)]

records = []
start = time.monotonic()
with torch.no_grad():
    for idx, (batch, transcripts) in enumerate(model.test_dataloader()):
        actual = ' '.join(model.tokenizer.tokenize(transcripts[0]))
        assert (idx, actual) == references[idx], idx
        raw = int(batch.feature_lengths[0])
        counts = {}
        for alpha in alphas:
            detector.feature_smoothing_alpha = alpha
            _, output_lengths = encoder(batch.features, batch.feature_lengths)
            counts[alpha] = int(output_lengths[0])
        records.append({'idx': idx, 'raw': raw, 'f0': counts[0.0], 'f25': counts[0.25]})
        if (idx + 1) % 400 == 0:
            print(idx + 1, 'samples', round(time.monotonic() - start, 1), 's', flush=True)

def aggregate(key):
    out = sum(r[key] for r in records)
    raw = sum(r['raw'] for r in records)
    return out, raw, out / raw

f0, raw_total, r0 = aggregate('f0')
f25, _, r25 = aggregate('f25')
deltas = [r['f25'] - r['f0'] for r in records]
changed = [d for d in deltas if d != 0]

result = {
    'name': 'alpha_ab',
    'num_samples': len(records),
    'checkpoint': str(checkpoint_path),
    'alphas': alphas,
    'threads': 4,
    'frames': {'alpha_0': f0, 'alpha_025': f25, 'input_mel': raw_total},
    'overall_retention': {'alpha_0': r0, 'alpha_025': r25},
    'delta_pp': 100 * (r25 - r0),
    'delta_frames': f25 - f0,
    'utterances_with_changed_count': len(changed),
    'utterances_unchanged': len(records) - len(changed),
    'delta_histogram': {str(d): deltas.count(d) for d in sorted(set(deltas))},
    'per_utterance': records,
    'elapsed_seconds': round(time.monotonic() - start, 1),
}
(ROOT / 'alpha_ab.result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('frames', 'overall_retention', 'delta_pp', 'delta_frames',
                                         'utterances_with_changed_count', 'delta_histogram')}, indent=2))
