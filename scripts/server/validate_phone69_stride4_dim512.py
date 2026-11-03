"""Validate the 512-wide stride-4 no-compression baseline before GPU preflight."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torchaudio
import yaml
from conformer_rnnt.model import conformer_rnnt_model_from_config
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
m = cfg['model']
assert cfg['paths']['from_checkpoint'] is None
assert cfg['tokenizer']['type'] == 'phone69' and m['num_symbols'] == 69
assert (m['time_reduction_input_dim'], m['time_reduction_stride']) == (128, 4)
assert (m['encoding_dim'], m['num_heads'], m['ffn_dim']) == (256, 8, 512)
assert (m['num_layers_block1'], m['num_layers_block2']) == (2, 8)
assert m['sequence_mode'] == 'offline' and 'right_context_length' not in m
assert m['compression'] == {'filter': 'none'}
torch.set_num_threads(2)
torch.manual_seed(42)
model = conformer_rnnt_model_from_config(cfg).train()
assert model.transcriber.output_linear.in_features == 512
assert model.transcriber.output_linear.out_features == 256
x = torch.randn(2, 47, 80, requires_grad=True)
lengths = torch.tensor([47, 35], dtype=torch.int32)
target = torch.tensor([[1, 2, 3], [4, 5, 69]], dtype=torch.int32)
tl = torch.tensor([3, 2], dtype=torch.int32)
logits, enc, _, _ = model(
    x, lengths, torch.cat([target.new_full((2, 1), 69), target], 1), tl + 1
)
loss = torchaudio.transforms.RNNTLoss(blank=69, reduction='sum', clamp=1.0)(
    logits, target, enc, tl
)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer.step()
assert all(torch.isfinite(p).all() for p in model.parameters())
module = LibriSpeechRNNTModule.from_config(cfg)
definition = module.offline_native_definition
assert definition['stride'] == 4 and definition['input_projection'] == 128
assert definition['encoding_dim'] == 256 and definition['layers'] == [2, 8]
assert definition['filter'] == 'none'
assert definition['detector'] is None and definition['mixer'] is None
checkpoint = {}
module.on_save_checkpoint(checkpoint)
module.on_load_checkpoint(checkpoint)
wrong = dict(checkpoint)
wrong['offline_native_definition'] = dict(definition, stride=2)
try:
    module.on_load_checkpoint(wrong)
except ValueError:
    pass
else:
    raise AssertionError('Incompatible checkpoint was accepted')
print(json.dumps({
    'status': 'passed',
    'parameters': sum(p.numel() for p in model.parameters()),
    'loss': float(loss.detach()),
    'lengths': enc.tolist(),
    'definition': definition,
}, indent=2))
