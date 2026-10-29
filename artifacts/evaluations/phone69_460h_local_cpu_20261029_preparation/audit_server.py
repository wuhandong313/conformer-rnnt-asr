"""Read-only preflight; execute on the server through SSH stdin, without eval."""

import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path('/test21/conformer-rnnt-asr')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import soundfile
import torch
import torchaudio
import yaml
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
from phone69_tokenizer import FixedPhone69Tokenizer

torch.set_num_threads(1)


def file_record(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest(),
                mtime_ns=after.st_mtime_ns)


models = []
for experiment in sorted(ROOT.glob('exp_phone69_460h_offline_native_*')):
    run = json.loads((experiment / 'run.json').read_text())
    if run.get('status') != 'trained' or run.get('epoch') != 50:
        continue
    config_path = Path(run['config'])
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_record = file_record(config_path)
    assert config_record['sha256'] == run['config_sha256'], experiment
    config = yaml.safe_load(config_path.read_text())
    assert config['tokenizer']['type'] == 'phone69'
    assert config['model']['num_symbols'] == 69
    assert config['model']['sequence_mode'] == 'offline'
    assert config['data']['test_subset'] == 'test-clean'
    assert config['data']['transcript_suffix'] == '-ph'
    assert config['data']['strip_spaces'] is False
    assert config['decoding']['beam_width'] == 20
    assert config['model']['compression'].get('execution_backend', 'reference') == 'reference'
    best = Path((experiment / 'best_checkpoint.txt').read_text().strip())
    if not best.is_absolute():
        best = ROOT / best
    assert best == Path(run['best_checkpoint']), experiment
    best_record = file_record(best)
    checkpoint = torch.load(best, map_location='cpu', weights_only=False)
    model = LibriSpeechRNNTModule.from_config(config)
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    assert all(torch.isfinite(t).all().item() for t in checkpoint['state_dict'].values())
    layout = [(name, list(tensor.shape), str(tensor.dtype))
              for name, tensor in sorted(checkpoint['state_dict'].items())]
    layout_sha256 = hashlib.sha256(json.dumps(layout).encode()).hexdigest()
    definition = checkpoint['offline_native_definition']
    tokenizer_fingerprint = model.tokenizer.fingerprint
    epoch, step = checkpoint['epoch'], checkpoint['global_step']
    del model, checkpoint
    gc.collect()
    boundary = Path(run['epoch_boundary_checkpoint'])
    boundary_record = file_record(boundary)
    checkpoint = torch.load(boundary, map_location='cpu', weights_only=False)
    progress = checkpoint['loops']['fit_loop']['epoch_progress']['current']
    assert progress['ready'] == progress['completed'] == 50
    assert checkpoint['offline_native_definition'] == definition
    assert checkpoint['phone69_tokenizer']['fingerprint'] == tokenizer_fingerprint
    models.append(dict(
        experiment=experiment.name, run=file_record(experiment / 'run.json'),
        status=run['status'], completed_epochs=50,
        training_complete_marker=(experiment / 'TRAINING_COMPLETE').exists(),
        config=config_record, best=best_record, boundary=boundary_record,
        best_epoch_zero_based=epoch, best_step=step,
        offline_native_definition=definition, tokenizer_fingerprint=tokenizer_fingerprint,
        strict_state_dict=True, finite_parameters=True, checkpoint_hook=True,
        state_dict_layout_sha256=layout_sha256,
        full_epoch_boundary=progress, stats=file_record(config['paths']['global_stats_path']),
        vocab=file_record(config['tokenizer']['vocab_path']),
    ))
    del checkpoint
    gc.collect()

assert len(models) == 8, 'Completed model set changed; review scope'
dataset_root = Path('/test21/librispeech')
test_root = dataset_root / 'LibriSpeech/test-clean'
audio = sorted(test_root.glob('*/*/*.flac'))
labels = sorted(test_root.glob('*/*/*-ph.trans.txt'))
tokenizer = FixedPhone69Tokenizer()
transcripts = {}
for label_file in labels:
    for line in label_file.read_text().splitlines():
        uid, transcript = line.split(' ', 1)
        assert uid not in transcripts
        transcripts[uid] = tokenizer.tokenize(transcript)
assert len(audio) == 2620
assert {path.stem for path in audio} == set(transcripts)
reference_phones = sum(len(transcripts[path.stem]) for path in audio)
assert reference_phones == 189023
duration = 0.0
files = []
for path in audio + labels:
    record = file_record(path)
    record['relative_path'] = str(path.relative_to(dataset_root))
    if path.suffix == '.flac':
        info = soundfile.info(str(path))
        assert info.samplerate == 16000 and info.channels == 1 and info.frames > 0
        duration += info.frames / info.samplerate
    files.append(record)

source_paths = sorted((ROOT / 'conformer_rnnt').rglob('*.py'))
source_paths += [ROOT / p for p in ['common.py', 'eval.py', 'phone69_tokenizer.py',
                                  'ipa_tokenizer.py', 'ipa_features.py', 'ipa_labels.py']]
sources = {str(path.relative_to(ROOT)): file_record(path)['sha256'] for path in source_paths}
print(json.dumps(dict(
    server_utc=datetime.now(timezone.utc).isoformat(), scope='read_only_no_inference',
    torch=torch.__version__, torchaudio=torchaudio.__version__, models=models,
    dataset=dict(root=str(dataset_root), subset='test-clean', utterances=len(audio),
                 reference_phones=reference_phones, duration_seconds=duration,
                 audio_files=len(audio), label_files=len(labels),
                 bytes=sum(record['bytes'] for record in files), files=files),
    current_server_source_sha256=sources,
), ensure_ascii=False))
