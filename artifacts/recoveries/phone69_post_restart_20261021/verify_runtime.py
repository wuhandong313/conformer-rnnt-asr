import json
import math
import platform
from pathlib import Path

import numpy
import pytorch_lightning
import scipy
import soundfile
import torch
import torchaudio
import yaml

assert platform.python_version().startswith('3.10.'), platform.python_version()
assert torch.__version__.split('+')[0] == '2.6.0', torch.__version__
assert torchaudio.__version__.split('+')[0] == '2.6.0', torchaudio.__version__
assert pytorch_lightning.__version__ == '2.6.6', pytorch_lightning.__version__
assert torch.cuda.is_available(), 'CUDA unavailable'
torch.set_num_threads(1)
logits = torch.randn(1, 3, 3, 70, device='cuda', requires_grad=True)
targets = torch.tensor([[1, 2]], device='cuda', dtype=torch.int32)
input_lengths = torch.tensor([3], device='cuda', dtype=torch.int32)
target_lengths = torch.tensor([2], device='cuda', dtype=torch.int32)
loss = torchaudio.functional.rnnt_loss(logits, targets, input_lengths, target_lengths, blank=69)
assert torch.isfinite(loss).all()
loss.backward()
assert torch.isfinite(logits.grad).all()
torch.cuda.synchronize()
root = Path('/test21/conformer-rnnt-asr')
configs = {}
for scheme in ['stride4_none', 'stride1_cosd_learned', 'stride2_cosd_learned', 'stride4_cosd_learned']:
    p = root / ('configs/server_phone69_460h_offline_native_' + scheme + '_s42_e50.yaml')
    config = yaml.safe_load(p.read_text())
    assert config['data'].get('test_subset') in ['test-clean', ['test-clean']]
    configs[scheme] = config['data']
samples = list(Path('/test21/librispeech').glob('**/test-clean/*/*/*.flac'))
assert len(samples) == 2620, 'test-clean FLAC count: ' + str(len(samples))
waveform, rate = torchaudio.load(str(samples[0]))
assert waveform.numel() > 0 and torch.isfinite(waveform).all() and rate == 16000
print(json.dumps({'valid': True, 'python': platform.python_version(), 'torch': torch.__version__,
                  'torchaudio': torchaudio.__version__, 'pytorch_lightning': pytorch_lightning.__version__,
                  'numpy': numpy.__version__, 'scipy': scipy.__version__, 'soundfile': soundfile.__version__,
                  'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                  'rnnt_cuda_forward_backward_finite': True, 'test_clean_samples': len(samples),
                  'audio_decode_ok': True, 'test_data_config': configs}, ensure_ascii=False))
