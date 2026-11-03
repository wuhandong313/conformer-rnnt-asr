"""Validate the approved layer-2 filter recipe without touching experiments."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torchaudio
import yaml
from conformer_rnnt.model import conformer_rnnt_model_from_config
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule


def main():
    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = conformer_rnnt_model_from_config(config).cuda().train()
    encoder = model.transcriber
    assert encoder.input_linear.out_features == 128
    assert len(encoder.conformer1.conformer_layers) == 2
    assert len(encoder.conformer2.conformer_layers) == 8
    assert encoder.conformer1.conformer_layers[0].self_attn.embed_dim == 128
    assert encoder.conformer2.conformer_layers[0].self_attn.embed_dim == 128
    assert encoder.compression.detector.feature_smoothing_alpha == 0.25
    assert encoder.compression.execution_backend == 'reference'
    assert sum(p.numel() for p in model.parameters()) == 4236299
    features = torch.randn(2, 47, 80, device='cuda', requires_grad=True)
    lengths = torch.tensor([47, 35], device='cuda', dtype=torch.int32)
    targets = torch.tensor([[1, 2, 3], [4, 5, 69]], device='cuda', dtype=torch.int32)
    target_lengths = torch.tensor([3, 2], device='cuda', dtype=torch.int32)
    previous = torch.cat((targets.new_full((2, 1), 69), targets), dim=1)
    logits, encoded, _, _ = model(features, lengths, previous, target_lengths + 1)
    loss = torchaudio.transforms.RNNTLoss(blank=69, reduction='sum', clamp=1.0)(
        logits, targets, encoded, target_lengths)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(features.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    module = LibriSpeechRNNTModule.from_config(config)
    definition = module.offline_native_definition
    assert definition['layers'] == [2, 8]
    assert definition['window_size'] == 5 and definition['weight_predictor_mode'] == 'five_frames'
    assert definition['detector']['params']['feature_smoothing_alpha'] == 0.25
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(checkpoint)
    wrong = copy.deepcopy(checkpoint)
    del wrong['offline_native_definition']['detector']['params']
    try:
        module.on_load_checkpoint(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError('Unfiltered checkpoint definition was accepted')
    print(json.dumps({'status': 'passed', 'device': torch.cuda.get_device_name(0),
        'parameters': sum(p.numel() for p in model.parameters()),
        'loss': float(loss.detach()), 'encoded_lengths': encoded.tolist(),
        'offline_native_definition': definition}, indent=2))


if __name__ == '__main__':
    main()
