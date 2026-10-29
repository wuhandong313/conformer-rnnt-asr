"""Prepare pinned local configurations and code; no network or inference."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORK = REPO / 'output/local_cpu_20261029'
PROVENANCE = REPO / 'artifacts/runs/phone69_460h_provenance_20261029'


def sha(data):
    return hashlib.sha256(data).hexdigest()


audit = json.loads((HERE / 'server_preflight.json').read_text())
provenance = json.loads((PROVENANCE / 'source_provenance_validation.json').read_text())
provenance_commit = subprocess.check_output(
    ['git', 'log', '-1', '--format=%H', '--', str(PROVENANCE / 'source_provenance_validation.json')],
    cwd=REPO, text=True).strip()
sources = {}
for group, gate_fragment in [('five_frames', 'native_offline_five_frames'),
                             ('placement', 'placement_stride1')]:
    code = WORK / 'code' / group
    paths = list((REPO / 'conformer_rnnt').rglob('*.py'))
    paths += [REPO / name for name in ['common.py', 'phone69_tokenizer.py',
                                      'ipa_tokenizer.py', 'ipa_features.py', 'ipa_labels.py']]
    paths += list((REPO / 'configs').glob('*.json'))
    contents = {str(path.relative_to(REPO)): path.read_bytes() for path in paths}
    gate = next(g for g in provenance['gates'] if gate_fragment in g['gate'])
    for item in gate['sources']:
        rel = item['path']
        data = (REPO / rel).read_bytes()
        if sha(data) != item['sha256']:
            for commit in [provenance['base_commit'], provenance_commit]:
                data = subprocess.check_output(['git', 'show', commit + ':' + rel], cwd=REPO)
                if sha(data) == item['sha256']:
                    break
        assert sha(data) == item['sha256'], rel
        contents[rel] = data
    evaluator = (REPO / 'eval.py').read_text()
    old_hook = '''    if config.get("model", {}).get("compression", {}).get("stages") is not None:
        model.on_load_checkpoint(ckpt)
    else:
        model.validate_tokenizer_checkpoint(ckpt)'''
    assert old_hook in evaluator
    evaluator = evaluator.replace(old_hook, '    model.on_load_checkpoint(ckpt)')
    evaluator = evaluator.replace('torch.load(ckpt_path, map_location="cpu")',
                                  'torch.load(ckpt_path, map_location="cpu", weights_only=False)')
    assert evaluator.count('time.time()') == 2
    evaluator = evaluator.replace('time.time()', 'time.perf_counter()')
    evaluator = evaluator.replace('    # Config\n',
        '    torch.set_num_threads(int(os.environ["LOCAL_CPU_THREADS"]))\n'
        '    torch.set_num_interop_threads(1)\n\n    # Config\n')
    contents['eval_cpu.py'] = evaluator.encode()
    for rel, data in contents.items():
        path = code / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    sources[group] = dict(root=str(code), training_source_gate=gate['gate'],
                          sha256={rel: sha(data) for rel, data in contents.items()})

order = ['stride4_none', 'stride1_cosd_learned_five_frames', 'stride2_cosd_learned_five_frames',
         'stride4_cosd_learned_five_frames', 'placement_mel_stride1', 'placement_layer5_stride1',
         'placement_layer7_stride1', 'placement_layers2_5_n3_stride1']
models = []
for short_name in order:
    record = next(m for m in audit['models'] if m['experiment'] ==
                  'exp_phone69_460h_offline_native_' + short_name + '_s42_e50')
    group = 'placement' if short_name.startswith('placement') else 'five_frames'
    source_config = PROVENANCE / 'server_snapshot/configs' / Path(record['config']['path']).name
    raw = source_config.read_bytes()
    assert sha(raw) == record['config']['sha256'], source_config
    original_path = HERE / 'configs_original' / source_config.name
    original_path.parent.mkdir(exist_ok=True)
    original_path.write_bytes(raw)
    original = yaml.safe_load(raw)
    local = copy.deepcopy(original)
    local['paths'].update(dataset_path=str(WORK / 'librispeech'),
                          global_stats_path=str(WORK / 'assets/global_stats.json'),
                          exp_dir=str(WORK / 'models' / short_name), from_checkpoint=None)
    local['tokenizer']['vocab_path'] = str(Path(sources[group]['root']) / 'configs/phone69_vocab.json')
    local['data']['num_workers'] = 0
    local['trainer'].update(accelerator='cpu', devices=1, strategy='auto')
    config_path = HERE / 'configs_cpu' / source_config.name
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(yaml.safe_dump(local, sort_keys=False, allow_unicode=True))
    assert local['model'] == original['model']
    assert local['training'] == original['training']
    assert local['decoding'] == original['decoding']
    assert {k: v for k, v in local['data'].items() if k != 'num_workers'} == {
        k: v for k, v in original['data'].items() if k != 'num_workers'}
    models.append(dict(name=short_name, group=group, experiment=record['experiment'],
                       source_config=str(original_path), source_config_sha256=sha(raw),
                       config=str(config_path), config_sha256=sha(config_path.read_bytes()),
                       checkpoint=str(WORK / 'models' / short_name / Path(record['best']['path']).name),
                       server=record))

plan = dict(status='awaiting_user_approval', approval_scope='transfer_and_serial_cpu_eval',
            repo=str(REPO), work=str(WORK), python='/opt/anaconda3/envs/aligner/bin/python',
            dyld_library_path='/opt/anaconda3/envs/aligner/lib/python3.12/site-packages/torch/lib',
            cpu_threads=4, interop_threads=1, data_workers=0,
            dataset=audit['dataset'], sources=sources, models=models,
            expected_utterances=2620, expected_reference_phones=189023,
            expected_smoke_utterances=2, beam_width=20,
            metric='corpus_PER', timer='time.perf_counter',
            checkpoint_bytes=sum(m['best']['bytes'] for m in audit['models']),
            transfer_bytes=sum(m['best']['bytes'] for m in audit['models']) +
                           audit['dataset']['bytes'] + audit['models'][0]['stats']['bytes'],
            excluded=['unfinished nine_frames (epoch 7 at inventory)',
                      'obsolete or deleted legacy checkpoints', 'historical phone69 100h experiment'])
(HERE / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
(HERE / 'dataset_files.txt').write_text(''.join(f['relative_path'] + '\n' for f in audit['dataset']['files']))
print('Prepared', len(models), 'models; no transfer or inference performed.')
