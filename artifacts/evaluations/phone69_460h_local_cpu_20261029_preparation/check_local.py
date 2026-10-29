"""Compare local factories with audited checkpoint definitions, without inference."""

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('group', choices=['five_frames', 'placement'])
parser.add_argument('--with-checkpoints', action='store_true')
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / 'plan.json').read_text())
sys.path.insert(0, plan['sources'][args.group]['root'])

import torch
import torchaudio
import pytorch_lightning
import yaml
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule

torch.set_num_threads(plan['cpu_threads'])
torch.set_num_interop_threads(1)
reports = []
for item in plan['models']:
    if item['group'] != args.group:
        continue
    config = yaml.safe_load(Path(item['config']).read_text())
    if not args.with_checkpoints:
        # Architecture checks do not consume the untransferred training statistics.
        config = copy.deepcopy(config)
        config['paths']['global_stats_path'] = str(Path(plan['repo']) / 'assets/statistics/global_stats.json')
    else:
        expected = item['server']['stats']['sha256']
        assert hashlib.sha256(Path(config['paths']['global_stats_path']).read_bytes()).hexdigest() == expected
    model = LibriSpeechRNNTModule.from_config(config)
    assert model.offline_native_definition == item['server']['offline_native_definition'], item['name']
    assert model.tokenizer.fingerprint == item['server']['tokenizer_fingerprint'], item['name']
    layout = [(name, list(tensor.shape), str(tensor.dtype)) for name, tensor in sorted(model.state_dict().items())]
    layout_sha = hashlib.sha256(json.dumps(layout).encode()).hexdigest()
    assert layout_sha == item['server']['state_dict_layout_sha256'], item['name']
    if args.with_checkpoints:
        checkpoint = torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
        model.on_load_checkpoint(checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        assert all(torch.isfinite(t).all().item() for t in checkpoint['state_dict'].values())
        assert checkpoint['epoch'] == item['server']['best_epoch_zero_based']
        assert checkpoint['global_step'] == item['server']['best_step']
        del checkpoint
    reports.append(dict(name=item['name'], definition=True, tokenizer=True,
                        state_dict_layout=True, local_checkpoint_loaded=args.with_checkpoints))
    del model
    gc.collect()
report = dict(valid=True, inference_performed=False, group=args.group,
              training_stats_loaded=args.with_checkpoints, torch=torch.__version__,
              torchaudio=torchaudio.__version__, pytorch_lightning=pytorch_lightning.__version__,
              models=reports)
suffix = 'checkpoints' if args.with_checkpoints else 'architecture'
(HERE / ('local_' + suffix + '_' + args.group + '.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
