#!/usr/bin/env python3
"""Reproduce audit edge cases on isolated CPU models; never edit training state."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

os.environ['CUDA_VISIBLE_DEVICES'] = ''
REPO = (Path(__file__).resolve().parents[2] if Path(__file__).is_file()
        else Path('/test21/conformer-rnnt-asr'))
sys.path.insert(0, str(REPO))

import torch
import torchaudio
import yaml
from scipy.signal import find_peaks

from conformer_rnnt.model import conformer_rnnt_model_from_config
from conformer_rnnt.compression.detectors import CosineDiffPeaksDetector
from conformer_rnnt.compression.mixers import LearnedWeightMixer

torch.set_num_threads(2)
os.chdir(REPO)
padding = []
for scheme in ('stride4_none', 'stride1_cosd_learned',
               'stride2_cosd_learned', 'stride4_cosd_learned'):
    torch.manual_seed(42)
    config = yaml.safe_load((REPO / f'configs/server_phone69_460h_offline_native_{scheme}_s42_e50.yaml').read_text())
    model = conformer_rnnt_model_from_config(config).cpu().eval()
    run_path = Path(config['paths']['exp_dir']) / 'run.json'
    checkpoint_path = None
    if run_path.exists():
        run = json.loads(run_path.read_text())
        checkpoint_path = (Path(run['best_checkpoint']) if run['status'] == 'trained'
                           else run_path.parent / 'checkpoints/last.ckpt')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict({key[len('model.'):]: value
                               for key, value in checkpoint['state_dict'].items()
                               if key.startswith('model.')}, strict=True)
        del checkpoint
    short = torch.randn(81, 80)
    pair = torch.randn(2, 101, 80)
    pair[0, :81] = short
    pair[0, 81:] = 0
    intermediate = []

    def save_stage1(module, inputs, output):
        intermediate.append(output[0].detach().clone())

    handle = model.transcriber.conformer1.register_forward_hook(save_stage1)
    with torch.no_grad():
        single, single_lengths = model.transcribe(short[None], torch.tensor([81]))
        batch, batch_lengths = model.transcribe(pair, torch.tensor([81, 101]))
    handle.remove()
    n = (81 + model.transcriber.time_reduction_stride - 1) // model.transcriber.time_reduction_stride
    delta = (intermediate[0][0, :n] - intermediate[1][0, :n]).abs()
    common = min(single_lengths[0].item(), batch_lengths[0].item())
    padding.append(dict(scheme=scheme, weights='fresh' if checkpoint_path is None else str(checkpoint_path),
                        input_length=81, paired_max_input_length=101,
                        stage1_max_abs_difference=float(delta.max()),
                        stage1_mean_abs_difference=float(delta.mean()),
                        single_output_length=single_lengths[0].item(),
                        paired_output_length=batch_lengths[0].item(),
                        output_prefix_max_abs_difference=float((single[0, :common] - batch[0, :common]).abs().max())))

torch.manual_seed(42)
mixer = LearnedWeightMixer(8, window_size=5, weight_predictor_mode='diff5')
with torch.no_grad():
    for parameter in mixer.parameters():
        parameter.zero_()
    mixer.weight_predictor[-1].bias[0] = 1000
frames = torch.randn(8, 8)
diff = (frames[1:] - frames[:-1]).abs().sum(-1)
boundary = mixer.mix(frames, torch.tensor([0]), diff, 8)
interior = mixer.mix(frames, torch.tensor([4]), diff, 8)
numerical = dict(all_inputs_and_parameters_finite=True,
                 forced_finite_logits=[1000, 0, 0, 0, 0], candidate_index=0,
                 boundary_output_finite=bool(torch.isfinite(boundary).all()),
                 interior_output_finite=bool(torch.isfinite(interior).all()),
                 observed_training_failure=False,
                 scope='Constructed numerical counterexample, not evidence that live weights reached these logits')

angles = torch.tensor([0., .1, 1., .2, .8, .1, 1.1, .1]).cumsum(0)
frames = torch.stack([angles.cos(), angles.sin()], -1)
c = 1 - (frames[1:] * frames[:-1]).sum(-1)
diff = (frames[1:] - frames[:-1]).abs().sum(-1)
indices, _ = CosineDiffPeaksDetector().detect(frames, diff, diff.numpy(), len(frames))
mapping = dict(cosine_distances=c.tolist(), peaks=find_peaks(c.numpy())[0].tolist(),
               valleys=find_peaks(-c.numpy())[0].tolist(), selected_frame_indices=indices.tolist(),
               definition='unique((peaks + 1) union valleys)',
               meaning='Adjacent extrema can map to the same frame; this is the current definition')

mask = torchaudio.transforms.TimeMasking(time_mask_param=100, p=0.2)
mask_case = None
for seed in range(1000):
    torch.manual_seed(seed)
    masked = mask(torch.ones(2, 80, 200))
    removed = masked[0, 0] == 0
    short_removed = int(removed[:20].sum())
    if short_removed > 4:
        mask_case = dict(seed=seed, iid_masks=mask.iid_masks, padded_length=200,
                         short_valid_length=20, configured_p=mask.p,
                         removed_short_valid_frames=short_removed,
                         short_removed_fraction=short_removed / 20,
                         shared_mask_across_examples=bool(torch.equal(masked[0], masked[1])),
                         masked_indices=removed.nonzero().flatten().tolist())
        break
assert mask_case is not None
print(json.dumps(dict(server_utc=datetime.now(timezone.utc).isoformat(), cuda_used=False,
                      padding_dependence=padding, boundary_weight_stability=numerical,
                      peak_valley_mapping=mapping, specaugment_time_mask=mask_case), indent=2))
