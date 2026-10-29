from datetime import datetime, timezone
from pathlib import Path
import csv, hashlib, json, os, sys, time

try:
    os.nice(10)
except OSError:
    pass
repo = Path('/test21/conformer-rnnt-asr')
os.chdir(repo)
sys.path.insert(0, str(repo))
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
import torch
import yaml
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule

torch.set_num_threads(2)
out = Path(__file__).resolve().parent
plan = json.loads((out / 'run_plan.json').read_text())
name = sys.argv[1]
item = next(x for x in plan['models'] if x['name'] == name)
now = lambda: datetime.now(timezone.utc).isoformat()

def save(filename, obj):
    path = out / filename
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)

progress = {'name': name, 'status': 'running', 'pid': os.getpid(), 'samples': 0, 'started_at_utc': now()}
save(name + '.state.json', progress)
try:
    cfg = yaml.safe_load(Path(item['config']).read_text())
    signature = {k: cfg['data'].get(k) for k in ('sample_rate', 'n_fft', 'n_mels', 'hop_length', 'transcript_suffix', 'test_subset')}
    signature.update(dataset_path=cfg['paths']['dataset_path'], global_stats_path=cfg['paths']['global_stats_path'], tokenizer=cfg.get('tokenizer'))
    assert signature == plan['feature_signature']
    model = LibriSpeechRNNTModule.from_config(cfg)
    checkpoint = torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
    if cfg.get('model', {}).get('compression', {}).get('stages') is not None:
        model.on_load_checkpoint(checkpoint)
    else:
        model.validate_tokenizer_checkpoint(checkpoint)
    assert checkpoint.get('offline_native_definition') == model.offline_native_definition
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
    del checkpoint
    model.eval()
    enc = model.model.transcriber
    stride = enc.time_reduction_stride
    refs = []
    for source in plan['reference_csvs']:
        with Path(source).open(newline='') as f:
            refs.append([(int(row['idx']), row['ref']) for row in csv.DictReader(f)])
    assert len(refs) == 2 and len(refs[0]) == plan['expected_samples'] and refs[0] == refs[1]
    length_hash = hashlib.sha256()
    totals = {'input_mel_frames': 0, 'encoder_input_frames': 0, 'encoder_output_frames': 0}
    stages = {}
    utterance_keeps = []
    start = time.monotonic()
    with torch.no_grad():
        for idx, (batch, transcripts) in enumerate(model.test_dataloader()):
            assert idx < len(refs[0])
            actual = ' '.join(model.tokenizer.tokenize(transcripts[0]))
            assert (idx, actual) == refs[0][idx]
            raw = int(batch.feature_lengths[0])
            assert raw > 0
            _, output_lengths = enc(batch.features, batch.feature_lengths)
            final = int(output_lengths[0])
            assert 0 < final <= raw
            trace = enc.compression.last_lengths
            if isinstance(trace, dict) and 'stages' in trace:
                assert trace['input'] == [raw] and trace['output'] == [final]
                encoder_in = int(trace['encoder_input'][0])
                rows = trace['stages']
                if rows and rows[0]['after_layer'] == 0:
                    assert int(rows[0]['input'][0]) == raw
                    assert encoder_in == (int(rows[0]['output'][0]) + stride - 1) // stride
                else:
                    assert encoder_in == (raw + stride - 1) // stride
                for row in rows:
                    key = 'after_layer_' + str(row['after_layer'])
                    v = stages.setdefault(key, {'input_frames': 0, 'output_frames': 0, 'candidate_frames': 0})
                    v['input_frames'] += int(row['input'][0])
                    v['output_frames'] += int(row['output'][0])
                    v['candidate_frames'] += int(row['candidates'][0])
            else:
                encoder_in = (raw + stride - 1) // stride
                if trace is None:
                    assert final == encoder_in
                else:
                    assert int(trace['input'][0]) == encoder_in and int(trace['output'][0]) == final
                    key = 'after_layer_' + str(cfg['model']['num_layers_block1'])
                    v = stages.setdefault(key, {'input_frames': 0, 'output_frames': 0, 'candidate_frames': 0})
                    v['input_frames'] += int(trace['input'][0])
                    v['output_frames'] += int(trace['output'][0])
                    v['candidate_frames'] += int(enc.compression.last_candidate_lengths[0])
            totals['input_mel_frames'] += raw
            totals['encoder_input_frames'] += encoder_in
            totals['encoder_output_frames'] += final
            utterance_keeps.append(final / raw)
            length_hash.update(f'{idx}:{raw}\n'.encode())
            if (idx + 1) % 100 == 0:
                progress.update(samples=idx+1, updated_at_utc=now(), elapsed_seconds=round(time.monotonic()-start, 1))
                save(name + '.state.json', progress)
    assert len(utterance_keeps) == plan['expected_samples']
    for value in stages.values():
        value['retention'] = value['output_frames'] / value['input_frames']
        value['reduction'] = 1 - value['retention']
    result = {
        'name': name, 'status': 'complete', 'num_samples': len(utterance_keeps),
        'device': 'cpu', 'threads': 2, 'checkpoint': item['checkpoint'], 'checkpoint_sha256': item['checkpoint_sha256'],
        'config': item['config'], 'config_sha256': item['config_sha256'], 'definition_matches_checkpoint': True,
        'feature_signature': signature, 'reference_order_matches_previous_eval': True,
        'input_sequence_sha256': length_hash.hexdigest(), 'stride': stride,
        'frame_totals': totals, 'overall_retention': totals['encoder_output_frames'] / totals['input_mel_frames'],
        'overall_reduction': 1 - totals['encoder_output_frames'] / totals['input_mel_frames'],
        'mean_utterance_retention': sum(utterance_keeps) / len(utterance_keeps),
        'stages': stages, 'completed_at_utc': now(), 'elapsed_seconds': round(time.monotonic()-start, 1),
    }
    save(name + '.result.json', result)
    progress.update(status='complete', samples=len(utterance_keeps), completed_at_utc=now())
    save(name + '.state.json', progress)
except BaseException as exc:
    progress.update(status='failed', error=repr(exc), failed_at_utc=now())
    save(name + '.state.json', progress)
    raise
