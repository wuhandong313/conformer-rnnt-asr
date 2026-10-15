#!/usr/bin/env python3
"""Probe fresh-model training memory on worst real phone69 batches.

Audio duration comes from cached FLAC headers across both training subsets and
validation. Probe optimizer steps use a disposable model, never a checkpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_samples(item):
    key, audio_path = item
    with audio_path.open('rb') as stream:
        header = stream.read(26)
    if header[:4] != b'fLaC' or header[4] & 127 != 0:
        raise ValueError('Unsupported FLAC header: {}'.format(audio_path))
    return key, int.from_bytes(header[18:26], 'big') & ((1 << 36) - 1)


def audio_sample_cache(datasets, path):
    cache = json.loads(path.read_text()) if path.exists() else {}
    missing = []
    for dataset in datasets:
        base = dataset.base_dataset
        for fileid in base._walker:
            key = base._url + '/' + fileid
            if key not in cache:
                speaker, chapter, _ = fileid.split('-')
                missing.append((key, Path(base._path) / speaker / chapter / (fileid + '.flac')))
    print('FLAC metadata cache: {} missing headers'.format(len(missing)), flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for index, (key, samples) in enumerate(pool.map(read_samples, missing), 1):
            cache[key] = samples
            if index % 10000 == 0:
                print('FLAC headers: {}/{}'.format(index, len(missing)), flush=True)
            if index % 20000 == 0:
                save_json(path, cache)
    if missing:
        save_json(path, cache)
    return cache


def select_batches(datasets, samples, model, cfg):
    m, d = cfg['model'], cfg['data']
    stride, rc = m['time_reduction_stride'], m['right_context_length']
    comp = m['compression']
    dynamic = 'detector' in comp
    records = []
    for dataset_index, dataset in enumerate(datasets):
        base = dataset.base_dataset
        transcripts = {}
        phones = [len(model.tokenizer.encode(dataset._transcript(fid, transcripts))) for fid in base._walker]
        frames = [samples[base._url + '/' + fid] // d['hop_length'] + 1 for fid in base._walker]
        for batch_index, batch in enumerate(dataset.batches):
            b = len(batch)
            raw_t = max(frames[i] for i in batch)
            t1 = (raw_t + rc) // stride
            t2 = max(1, round((t1 - rc // stride) * comp['keep_rate'])) + rc // stride if dynamic else t1
            u = max(phones[i] for i in batch)
            records.append({
                'dataset_index': dataset_index, 'subset': base._url, 'batch_index': batch_index,
                'utterances': b, 'max_feature_frames': raw_t, 'max_target_phones': u,
                'attention_elements': b * m['num_heads'] * (
                    m['num_layers_block1'] * t1**2 + m['num_layers_block2'] * t2**2),
                'joint_elements': b * t2 * (u + 1) * (m['num_symbols'] + 1),
            })
    selected = {}
    for key in ('attention_elements', 'joint_elements', 'utterances', 'max_feature_frames'):
        record = max(records, key=lambda row: row[key])
        selected[(record['dataset_index'], record['batch_index'])] = record
    for dataset_index in range(len(datasets)):
        record = max((row for row in records if row['dataset_index'] == dataset_index), key=lambda row: row['attention_elements'])
        selected[(dataset_index, record['batch_index'])] = record
    return list(selected.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-reserved-fraction', type=float, default=0.85)
    parser.add_argument('--budget', type=int, help='Probe an override without editing the recipe')
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--guard-ratio', type=float, help='Override only the RNNT frame estimate')
    parser.add_argument('--keep-cache', action='store_true', help='Keep allocator cache between different batch shapes')
    args = parser.parse_args()
    os.chdir(REPO)

    import torch
    import yaml
    from pytorch_lightning import seed_everything
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule

    torch.set_num_threads(8)
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg['data']['num_workers'] = 0
    if args.budget is not None:
        cfg['data']['max_token_limit'] = args.budget
    if args.guard_ratio is not None:
        cfg['data']['rnnt_frames_per_char'] = args.guard_ratio
    seed_everything(cfg['seed'], workers=True)
    torch.set_float32_matmul_precision(cfg['trainer']['matmul_precision'])
    if cfg['tokenizer']['type'] != 'phone69':
        raise ValueError('This probe requires fixed phone69')
    output = Path(args.out)
    report = {
        'status': 'preparing', 'config': args.config,
        'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'pid': os.getpid(), 'budget': cfg['data']['max_token_limit'],
        'rnnt_frames_per_char': cfg['data']['rnnt_frames_per_char'], 'results': [],
        'keep_allocator_cache': args.keep_cache,
    }
    save_json(output, report)
    model = LibriSpeechRNNTModule.from_config(cfg)
    datasets = list(model.train_dataloader().dataset.datasets) + list(model.val_dataloader().dataset.datasets)
    samples = audio_sample_cache(datasets, REPO / 'output/phone69_460h/audio_samples_cache.json')
    if args.metadata_only:
        report.update(status='metadata_cached', cached_utterances=len(samples))
        save_json(output, report)
        print(json.dumps(report), flush=True)
        return
    selected = select_batches(datasets, samples, model, cfg)
    worst_joint = max(selected, key=lambda row: row['joint_elements'])
    if worst_joint['joint_elements'] >= cfg['data']['rnnt_numel_limit']:
        report.update(status='unsafe_joint', selected_batches=selected,
                      error='Actual audio-duration joint upper bound exceeds RNNT element limit')
        save_json(output, report)
        raise SystemExit(2)
    report.update(status='running', selected_batches=selected,
                  batches_per_subset={ds.base_dataset._url: len(ds) for ds in datasets},
                  parameters=sum(p.numel() for p in model.parameters()))
    save_json(output, report)
    print(json.dumps({'selected_batches': selected}), flush=True)
    model.cuda().train()
    model.log = lambda *a, **kw: None
    total_memory = torch.cuda.get_device_properties(0).total_memory
    report['gpu_total_GiB'] = total_memory / 1024**3
    try:
        for record in selected:
            model.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if not args.keep_cache:
                torch.cuda.empty_cache()
            dataset = datasets[record['dataset_index']]
            batch = model._train_collate_fn(dataset[record['batch_index']])
            batch = type(batch)(*(value.cuda() for value in batch))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            loss = model._step(batch, record['batch_index'], 'train')
            if not torch.isfinite(loss):
                raise ValueError('Non-finite probe loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip_val'])
            if not torch.isfinite(norm):
                raise ValueError('Non-finite probe gradient')
            model.optimizer.step()
            torch.cuda.synchronize()
            result = dict(record, phones=int(batch.target_lengths.sum()),
                          peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3,
                          peak_reserved_GiB=torch.cuda.max_memory_reserved() / 1024**3,
                          seconds=time.monotonic() - started, loss=float(loss.detach()))
            report['results'].append(result)
            print(json.dumps(result), flush=True)
            save_json(output, report)
            if torch.cuda.max_memory_reserved() > total_memory * args.max_reserved_fraction:
                raise RuntimeError('Probe exceeded reserved-memory safety margin')
            del loss, batch
        report['status'] = 'passed'
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='oom', error=str(error))
    except Exception as error:
        report.update(status='failed', error=repr(error))
    finally:
        save_json(output, report)
    print(json.dumps({'status': report['status'], 'report': str(output)}), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
