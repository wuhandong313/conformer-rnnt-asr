#!/usr/bin/env python3
"""Aggregate existing full Phone69 predictions/counts on frozen rate groups."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ('encoder_weights', 'attention_products', 'projections', 'fusion_mlp',
              'detector_dots', 'fusion_weighted_sum', 'predictor', 'joiner')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def aggregate(predictions, counts, uids):
    selected = [predictions[uid] for uid in uids]
    phones = sum(int(row['reference_phones']) for row in selected)
    errors = sum(int(row['phone_errors']) for row in selected)
    seconds = math.fsum(float(row['audio_seconds']) for row in selected)
    macs = {key: sum(int(counts[uid].get(key, 0)) for uid in uids) for key in COMPONENTS}
    scale = seconds * 1e9
    return dict(utterances=len(uids), speakers=len({uid.split('-')[0] for uid in uids}),
                reference_phones=phones, phone_errors=errors, audio_seconds=seconds,
                corpus_PER=errors / phones, components_macs=macs,
                total_major_macs=sum(macs.values()),
                total_gmac_per_audio_second=sum(macs.values()) / scale,
                encoder_compression_gmac_per_audio_second=sum(macs[k] for k in COMPONENTS[:-2]) / scale,
                predictor_gmac_per_audio_second=macs['predictor'] / scale,
                joiner_gmac_per_audio_second=macs['joiner'] / scale)


def run(manifest_path, out):
    manifest = json.loads(manifest_path.read_text())
    groups, bins, hashes = {}, [], {}
    for index in range(1, 11):
        path = ROOT / manifest['decile_directory'] / ('decile_%02d.txt' % index)
        uids = [line.split()[0] for line in path.read_text().splitlines() if line.strip()]
        if len(uids) != 262 or len(set(uids)) != 262:
            raise ValueError('The frozen decile must contain 262 distinct utterances')
        groups['decile_%02d' % index] = uids
        bins.append(uids)
        hashes[str(path.relative_to(ROOT))] = sha(path)
    all_uids = sum(bins, [])
    if len(set(all_uids)) != 2620:
        raise ValueError('Frozen deciles do not partition the 2,620-utterance corpus')
    groups.update(all=all_uids, slow=sum(bins[:3], []), middle=sum(bins[3:7], []),
                  fast=sum(bins[7:], []))
    models, reference = {}, None
    for entry in manifest['models']:
        path = ROOT / entry['predictions_csv']
        rows = read_csv(path)
        predictions = {row['uid']: row for row in rows}
        if len(rows) != len(predictions) or set(predictions) != set(all_uids):
            raise ValueError('Expected complete, unique predictions for every frozen utterance')
        this_reference = {uid: (row['ref'], int(row['reference_phones']), float(row['audio_seconds']))
                          for uid, row in predictions.items()}
        if reference is None:
            reference = this_reference
        if this_reference != reference:
            raise ValueError('Reference phones or audio duration differ across systems')
        hashes[str(path.relative_to(ROOT))] = sha(path)
        result_path = ROOT / entry['result_json']
        source_result = json.loads(result_path.read_text())
        if (source_result['csv_sha256'] != sha(path) or source_result['utterances'] != 2620
                or source_result['reference_phones'] != 189023):
            raise ValueError('Prediction CSV is not the recorded full-corpus result')
        hashes[str(result_path.relative_to(ROOT))] = sha(result_path)
        if entry.get('counts_json'):
            count_path = ROOT / entry['counts_json']
            payload = json.loads(count_path.read_text())
            count_rows = payload['rows']
            if payload['utterances'] != 2620 or payload['predictions_checked'] != 2620:
                raise ValueError('Pilot counts cannot enter the full-corpus rate analysis')
            if (payload['checkpoint_sha256'] != source_result['checkpoint_sha256']
                    or payload['config_sha256'] != source_result['config_sha256']):
                raise ValueError('MAC count and predictions belong to different model configurations')
            hashes[str(count_path.relative_to(ROOT))] = sha(count_path)
        else:
            payload, count_rows = None, rows
        counts = {row['uid']: row for row in count_rows}
        if len(count_rows) != len(counts) or set(counts) != set(all_uids):
            raise ValueError('Full major-MAC records are missing or duplicated')
        values = {name: aggregate(predictions, counts, uids) for name, uids in groups.items()}
        if values['all']['reference_phones'] != 189023 or not math.isclose(
                values['all']['audio_seconds'], 19452.480625, abs_tol=1e-9):
            raise ValueError('Full corpus denominators changed')
        if payload and values['all']['total_major_macs'] != payload['total_major_macs']:
            raise ValueError('Per-utterance MAC totals differ from the completed count record')
        if (values['all']['phone_errors'] != source_result['phone_errors'] or not math.isclose(
                values['all']['corpus_PER'], source_result['corpus_PER'], rel_tol=1e-12)):
            raise ValueError('Rate groups do not reproduce the recorded full-corpus PER')
        if not payload and values['all']['total_major_macs'] != source_result['total_major_macs']:
            raise ValueError('Rate groups do not reproduce the direct full-corpus MAC count')
        models[entry['id']] = dict(name=entry['name'], archive_id=entry.get('archive_id'),
                                   predictions_csv=entry['predictions_csv'],
                                   counts_json=entry.get('counts_json'), groups=values)
    baseline = models[manifest['baseline_id']]['groups']
    for model in models.values():
        for name, values in model['groups'].items():
            values['delta_PER_vs_baseline_pp'] = 100 * (values['corpus_PER'] - baseline[name]['corpus_PER'])
            values['major_MAC_ratio_vs_baseline'] = values['total_major_macs'] / baseline[name]['total_major_macs']
        model['fast_minus_slow_delta_PER_pp'] = (
            model['groups']['fast']['delta_PER_vs_baseline_pp'] -
            model['groups']['slow']['delta_PER_vs_baseline_pp'])
    out.mkdir(parents=True, exist_ok=True)
    result = dict(schema=1, subset='test-clean', analysis='existing predictions and completed full MAC counts only',
                  exploratory=True, baseline_id=manifest['baseline_id'],
                  grouping='Frozen original PPM deciles; slow=1..3, middle=4..7, fast=8..10; no regrouping by errors',
                  rate_definition='Original space-separated reference phones per full audio minute, including pauses',
                  uncertainty='Point estimates only; paired speaker bootstrap and held-out confirmation remain pending for the added comparisons',
                  protocol=manifest['protocol'], manifest_sha256=sha(manifest_path), source_sha256=hashes,
                  aggregation_script_sha256=sha(__file__),
                  pending_models=manifest.get('pending_models', []), models=models)
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    fields = ['model', 'group', 'utterances', 'speakers', 'reference_phones', 'phone_errors', 'audio_seconds',
              'corpus_PER', 'total_gmac_per_audio_second', 'encoder_compression_gmac_per_audio_second',
              'predictor_gmac_per_audio_second', 'joiner_gmac_per_audio_second', 'delta_PER_vs_baseline_pp']
    with (out / 'groups.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, data in models.items():
            for group in ['all', 'slow', 'middle', 'fast'] + ['decile_%02d' % i for i in range(1, 11)]:
                writer.writerow(dict(model=model, group=group,
                                     **{key: data['groups'][group][key] for key in fields[2:]}))
    print(json.dumps(dict(models=len(models), utterances_per_model=2620,
                          inference_run=False, result=str(out / 'results.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest.resolve(), args.out.resolve())
