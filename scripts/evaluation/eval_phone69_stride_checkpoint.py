#!/usr/bin/env python3
"""Evaluate one frozen fixed-stride checkpoint: corpus PER and major MACs.

Uses the established local CPU/FP32/batch-one/beam-20 protocol. Counts the
encoder and the actual predictor/joiner search path in the same decoding pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import platform
import sys

COMPONENTS = ('encoder_weights', 'attention_products', 'projections', 'fusion_mlp',
              'detector_dots', 'fusion_weighted_sum', 'predictor', 'joiner')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
N_UTTS, N_PHONES = 2620, 189023


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified(path, digest):
    if sha(path) != digest:
        raise ValueError('SHA-256 mismatch: ' + str(path))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def fixed_encoder_macs(frames, stride):
    width = 128 * stride
    steps = (frames + stride - 1) // stride
    return (80 * 128 * frames + 10 * (steps * (4 * width * 512 + 7 * width**2 + 31 * width)
                                    + 2 * width * steps**2) + 256 * width * steps)


def summarize(rows, expected_uids, expected_phones=N_PHONES):
    if [row['uid'] for row in rows] != expected_uids or len(set(expected_uids)) != len(expected_uids):
        raise ValueError('Evaluation coverage/order differs from frozen corpus')
    phones = sum(int(row['reference_phones']) for row in rows)
    if phones != expected_phones:
        raise ValueError('Reference phone coverage is incomplete')
    errors = sum(int(row['phone_errors']) for row in rows)
    duration = math.fsum(float(row['audio_seconds']) for row in rows)
    counts = {key: sum(int(row[key]) for row in rows) for key in COMPONENTS}
    if duration <= 0 or any(value < 0 for value in counts.values()) or errors < 0:
        raise ValueError('Invalid corpus totals')
    encoder = sum(counts[k] for k in COMPONENTS if k not in ('predictor', 'joiner'))
    return dict(utterances=len(rows), reference_phones=phones, phone_errors=errors,
                corpus_PER=errors / phones, audio_seconds=duration, components_macs=counts,
                total_major_macs=sum(counts.values()),
                gmac_per_audio_second=sum(counts.values()) / duration / 1e9,
                encoder_compression_gmac_per_audio_second=encoder / duration / 1e9,
                predictor_gmac_per_audio_second=counts['predictor'] / duration / 1e9,
                joiner_gmac_per_audio_second=counts['joiner'] / duration / 1e9)


def run(manifest_path, out):
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['model']
    stride = entry['stride']
    if stride not in (1, 2, 8) or manifest['protocol'] != dict(
            device='cpu', dtype='float32', batch=1, beam=20, threads=4, interop_threads=1,
            data_workers=0, openmp_max_active_levels=1, warmup_utterances=10, subset='test-clean'):
        raise ValueError('Evaluation differs from the fixed CPU protocol')
    if any(os.environ.get(key) != '4' for key in THREAD_ENV) or os.environ.get('OMP_MAX_ACTIVE_LEVELS') != '1':
        raise ValueError('Set CPU thread limits before Python starts')
    if (out / 'utterances.csv').exists() or (out / 'result.json').exists():
        raise FileExistsError('Preserve the previous evaluation attempt')
    source = Path(manifest['source']).resolve()
    for relative, digest in manifest['code_sha256'].items():
        file = (source / relative).resolve()
        if not file.is_relative_to(source):
            raise ValueError('Source path escapes its frozen directory')
        verified(file, digest)
    verified(__file__, manifest['evaluator_sha256'])
    for key in ('checkpoint', 'config', 'training_config'):
        verified(entry[key], entry[key + '_sha256'])
    verified(manifest['reference_csv'], manifest['reference_csv_sha256'])
    verified(manifest['runtime']['openmp']['path'], manifest['runtime']['openmp']['sha256'])
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(source / 'scripts/evaluation'))
    import soundfile
    import torch
    import torchaudio
    import yaml
    from threadpoolctl import threadpool_info
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from conformer_rnnt.metrics import phone_error_counts
    from measure_phone69_460h_compute import observe
    if not Path(sys.modules[LibriSpeechRNNTModule.__module__].__file__).resolve().is_relative_to(source):
        raise ValueError('Model was not imported from the frozen training source')
    runtime = manifest['runtime']
    if (platform.python_version(), torch.__version__, torchaudio.__version__) != (
            runtime['python_version'], runtime['torch'], runtime['torchaudio']):
        raise ValueError('Use the established local Python/PyTorch runtime')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = yaml.safe_load(Path(entry['config']).read_text())
    training = yaml.safe_load(Path(entry['training_config']).read_text())
    if cfg['model'] != training['model'] or cfg['model']['time_reduction_stride'] != stride:
        raise ValueError('Evaluation model differs from the training recipe')
    verified(cfg['paths']['global_stats_path'], manifest['global_stats_sha256'])
    verified(cfg['tokenizer']['vocab_path'], manifest['vocab_sha256'])
    audio = manifest['audio']
    subset = Path(cfg['paths']['dataset_path']) / 'LibriSpeech/test-clean'
    files = sorted(subset.glob('*/*/*.flac'), key=lambda file: file.stem)
    if len(audio) != N_UTTS or [p.stem for p in files] != [item['uid'] for item in audio]:
        raise ValueError('Test-clean inventory differs from the frozen corpus')
    for path, item in zip(files, audio, strict=True):
        verified(path, item['sha256'])
        header = soundfile.info(path)
        if (header.frames, header.samplerate) != (item['samples'], item['sample_rate']):
            raise ValueError('Audio header differs from the frozen corpus')
    for name, digest in manifest['chapter_labels'].items():
        speaker, chapter, _ = name.split('-', 2)
        verified(subset / speaker / chapter / name, digest)
    with Path(manifest['reference_csv']).open(newline='') as stream:
        references = list(csv.DictReader(stream))
    if [row['uid'] for row in references] != [item['uid'] for item in audio]:
        raise ValueError('Reference order differs from the frozen corpus')
    model = LibriSpeechRNNTModule.from_config(cfg)
    checkpoint = torch.load(entry['checkpoint'], map_location='cpu', weights_only=False)
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    checkpoint_epoch, checkpoint_step = checkpoint.get('epoch'), checkpoint.get('global_step')
    del checkpoint
    model.cpu().eval()
    if sum(p.numel() for p in model.parameters()) != entry['expected_parameters']:
        raise ValueError('Parameter count does not match the trained stride model')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Expected FP32 parameters on CPU')
    if model.beam_width != 20 or model.tokenizer_type != 'phone69':
        raise ValueError('Unexpected decoder or tokenizer')
    pools = threadpool_info()
    omp = {p['filepath'] for p in pools if p['internal_api'] == 'openmp'}
    if omp != {runtime['openmp']['path']} or any(p['num_threads'] != 4 for p in pools):
        raise ValueError('CPU runtime/thread budget differs: ' + repr(pools))
    loader = model.test_dataloader()
    if len(loader) != N_UTTS or loader.batch_size != 1 or loader.num_workers != 0:
        raise ValueError('Expected batch-one full test-clean loader')
    if list(loader.dataset.datasets[0]._walker) != [item['uid'] for item in audio]:
        raise ValueError('Dataset order differs from the frozen corpus')
    write_json(out / 'progress.json', dict(status='warming_up', completed=0, total=N_UTTS, updated=now()))
    counts, lengths = Counter(), []
    rows = []
    fields = ['idx', 'uid', 'audio_seconds', 'phone_errors', 'reference_phones', 'PER', 'ref', 'hyp',
              'layer_frames', *COMPONENTS, 'predictor_tokens', 'joiner_rows', 'joiner_calls']
    with torch.no_grad():
        for batch, _ in islice(loader, 10):
            model(batch)
        observe(model, counts, lengths)
        with (out / 'utterances.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, (batch, transcripts) in enumerate(loader):
                item = audio[index]
                reference = ' '.join(model.tokenizer.tokenize(transcripts[0]))
                if reference != references[index]['ref']:
                    raise ValueError('Reference phones changed at utterance %d' % index)
                counts.clear()
                lengths.clear()
                hypothesis = model(batch)
                errors, phones = phone_error_counts(reference, hypothesis)
                if phones != int(references[index]['reference_phones']) or len(lengths) != 10:
                    raise ValueError('Incomplete reference or encoder trace')
                encoder = sum(counts[k] for k in COMPONENTS if k not in ('predictor', 'joiner'))
                if encoder != fixed_encoder_macs(int(batch.feature_lengths[0]), stride):
                    raise ValueError('Encoder execution trace differs from the independently audited formula')
                if min(counts[k] for k in ('predictor', 'joiner', 'predictor_tokens', 'joiner_rows')) <= 0:
                    raise ValueError('Decoder execution trace is incomplete')
                row = dict(idx=index, uid=item['uid'], audio_seconds=item['samples']/item['sample_rate'],
                           phone_errors=errors, reference_phones=phones, PER=errors/phones,
                           ref=reference, hyp=hypothesis, layer_frames=json.dumps(lengths),
                           **{key: counts[key] for key in fields if key in (*COMPONENTS, 'predictor_tokens', 'joiner_rows', 'joiner_calls')})
                writer.writerow(row)
                rows.append(row)
                if (index + 1) % 100 == 0 or index + 1 == N_UTTS:
                    stream.flush()
                    write_json(out / 'progress.json', dict(status='running', completed=index+1,
                                                          total=N_UTTS, updated=now()))
                    print(entry['name'], index+1, '/', N_UTTS, flush=True)
    totals = summarize(rows, [item['uid'] for item in audio])
    with (out / 'utterances.csv').open(newline='') as stream:
        if summarize(list(csv.DictReader(stream)), [item['uid'] for item in audio]) != totals:
            raise ValueError('Persisted utterance records disagree with the corpus totals')
    result = dict(model=entry, **totals, protocol=manifest['protocol'], runtime=runtime,
                  checkpoint_epoch=checkpoint_epoch, checkpoint_global_step=checkpoint_step,
                  manifest_sha256=sha(manifest_path), csv_sha256=sha(out/'utterances.csv'),
                  counted_scope='linear, convolution, attention products, compression, and actual beam-search predictor/joiner matrices',
                  excluded_scope='feature extraction, normalization, softmax, activations, comparisons/sorting, memory access',
                  completed_at_utc=now(), verified_full_corpus=True)
    write_json(out / 'result.json', result)
    write_json(out / 'progress.json', dict(status='complete', completed=N_UTTS, total=N_UTTS, updated=now()))
    print(json.dumps(totals), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.manifest.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
