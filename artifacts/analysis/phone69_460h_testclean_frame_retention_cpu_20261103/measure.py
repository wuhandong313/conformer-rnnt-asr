"""Measure frame retention of one checkpoint on test-clean.

Mirrors artifacts/analysis/phone69_460h_testclean_frame_retention_cpu_20261027/worker.py:
overall_retention = sum(final encoder output frames) / sum(input Mel frames),
stage retention uses the same observed per-utterance counters.
"""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = Path('/Users/wuhandong/Desktop/Projects/conformer-rnnt-asr/output/local_cpu_20261103/code/feature_smooth')
os.chdir(CODE)
sys.path.insert(0, str(CODE))
os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')

import torch
import yaml

from conformer_rnnt.lightning_module import LibriSpeechRNNTModule

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

name = sys.argv[1]
config_path = Path(sys.argv[2])
checkpoint_path = Path(sys.argv[3])
reference_csv = Path(sys.argv[4])

cfg = yaml.safe_load(config_path.read_text())
model = LibriSpeechRNNTModule.from_config(cfg)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.on_load_checkpoint(checkpoint)
assert checkpoint['offline_native_definition'] == model.offline_native_definition
model.load_state_dict(checkpoint['state_dict'], strict=True)
del checkpoint
model.eval()

encoder = model.model.transcriber
stride = encoder.time_reduction_stride
with reference_csv.open(newline='') as stream:
    references = [(int(row['idx']), row['ref']) for row in csv.DictReader(stream)]
assert len(references) == 2620

totals = {'input_mel_frames': 0, 'encoder_input_frames': 0, 'encoder_output_frames': 0}
stages = {}
utterance_keeps = []
length_hash = hashlib.sha256()
start = time.monotonic()

with torch.no_grad():
    for idx, (batch, transcripts) in enumerate(model.test_dataloader()):
        actual = ' '.join(model.tokenizer.tokenize(transcripts[0]))
        assert (idx, actual) == references[idx], idx
        raw = int(batch.feature_lengths[0])
        assert raw > 0
        _, output_lengths = encoder(batch.features, batch.feature_lengths)
        final = int(output_lengths[0])
        assert 0 < final <= raw
        trace = encoder.compression.last_lengths
        if isinstance(trace, dict) and 'stages' in trace:
            encoder_in = int(trace['encoder_input'][0])
            assert int(trace['input'][0]) == raw and int(trace['output'][0]) == final
            for row in trace['stages']:
                key = 'after_layer_' + str(row['after_layer'])
                value = stages.setdefault(key, {'input_frames': 0, 'output_frames': 0, 'candidate_frames': 0})
                value['input_frames'] += int(row['input'][0])
                value['output_frames'] += int(row['output'][0])
                value['candidate_frames'] += int(row['candidates'][0])
        else:
            encoder_in = (raw + stride - 1) // stride
            assert int(trace['input'][0]) == encoder_in and int(trace['output'][0]) == final
            key = 'after_layer_' + str(cfg['model']['num_layers_block1'])
            value = stages.setdefault(key, {'input_frames': 0, 'output_frames': 0, 'candidate_frames': 0})
            value['input_frames'] += int(trace['input'][0])
            value['output_frames'] += int(trace['output'][0])
            value['candidate_frames'] += int(encoder.compression.last_candidate_lengths[0])
        totals['input_mel_frames'] += raw
        totals['encoder_input_frames'] += encoder_in
        totals['encoder_output_frames'] += final
        utterance_keeps.append(final / raw)
        length_hash.update(f'{idx}:{raw}\n'.encode())
        if (idx + 1) % 200 == 0:
            print(idx + 1, 'samples', round(time.monotonic() - start, 1), 's', flush=True)

assert len(utterance_keeps) == 2620
for value in stages.values():
    value['retention'] = value['output_frames'] / value['input_frames']
    value['reduction'] = 1 - value['retention']

result = {
    'name': name,
    'status': 'complete',
    'num_samples': len(utterance_keeps),
    'device': 'cpu',
    'threads': 4,
    'checkpoint': str(checkpoint_path),
    'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    'config': str(config_path),
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'definition_matches_checkpoint': True,
    'input_sequence_sha256': length_hash.hexdigest(),
    'stride': stride,
    'frame_totals': totals,
    'overall_retention': totals['encoder_output_frames'] / totals['input_mel_frames'],
    'overall_reduction': 1 - totals['encoder_output_frames'] / totals['input_mel_frames'],
    'mean_utterance_retention': sum(utterance_keeps) / len(utterance_keeps),
    'stages': stages,
    'elapsed_seconds': round(time.monotonic() - start, 1),
}
(ROOT / (name + '.result.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('frame_totals', 'overall_retention', 'mean_utterance_retention', 'stages', 'input_sequence_sha256')}, indent=2))
