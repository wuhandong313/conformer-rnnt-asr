#!/usr/bin/env python3
"""Audit actual offline phone69 factories on CPU without touching live training."""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
REPO = (Path(__file__).resolve().parents[2] if Path(__file__).is_file()
        else Path('/test21/conformer-rnnt-asr'))
sys.path.insert(0, str(REPO))

import torch
import torchaudio
import yaml

from conformer_rnnt.compression import DynamicCompression, NaturalPeakCompression
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
from conformer_rnnt.model import sequence_options_from_config

SCHEMES = ('stride4_none', 'stride1_cosd_learned',
           'stride2_cosd_learned', 'stride4_cosd_learned')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retired_keys(value, path=''):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            current = f'{path}.{key}' if path else key
            if key in ('keep_rate', 'right_context_length', 'left_context_length'):
                found.append(current)
            found.extend(retired_keys(child, current))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    os.chdir(REPO)
    torch.set_num_threads(2)
    audit = REPO / 'artifacts/runs/phone69_460h_native_offline_20261018'
    gate = json.loads((audit / 'prelaunch_validation.json').read_text())
    assert all(sha(REPO / name) == expected for name, expected in gate['source_sha256'].items())
    assert sha(REPO / gate['report']) == gate['report_sha256']
    records = []
    for scheme in SCHEMES:
        torch.manual_seed(42)
        config_path = REPO / f'configs/server_phone69_460h_offline_native_{scheme}_s42_e50.yaml'
        config = yaml.safe_load(config_path.read_text())
        expected = next(x for x in gate['models'] if x['scheme'] == scheme)
        assert sha(config_path) == expected['config_sha256']
        assert not retired_keys(config)
        options = sequence_options_from_config(config['model'])
        assert options == dict(sequence_mode='offline', right_context_length=0,
                               compression_right_context_length=None)
        for injected in ('keep_rate', 'right_context_length'):
            wrong = copy.deepcopy(config['model'])
            wrong['compression'][injected] = None if injected == 'keep_rate' else 0
            try:
                sequence_options_from_config(wrong)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{injected} was accepted')
        module = LibriSpeechRNNTModule.from_config(config).cpu().eval()
        encoder = module.model.transcriber
        compression = encoder.compression
        assert isinstance(compression, NaturalPeakCompression)
        assert not any(isinstance(x, DynamicCompression) for x in module.modules())
        assert not hasattr(compression, 'keep_rate') and not hasattr(compression, 'right_context_length')
        assert encoder.right_context_length == 0 and encoder.time_reduction.pad_to_stride
        samples = [(torch.randn(1, 16000), 16000, 'CH AE1 P'),
                   (torch.randn(1, 12800), 16000, 'AH0 T')]
        batch = module._valid_collate_fn(samples)
        augmented = module._train_collate_fn(samples)
        assert batch.feature_lengths.tolist() == [101, 81]
        assert augmented.features.shape == batch.features.shape
        assert augmented.feature_lengths.tolist() == [101, 81]
        assert batch.features.size(1) == int(batch.feature_lengths.max())
        assert torch.count_nonzero(batch.features[1, 81:]) == 0
        assert torch.count_nonzero(augmented.features[1, 81:]) == 0
        traces = []
        attention = []
        if not compression.passthrough:
            detector = compression.detector
            detect = detector.detect

            def traced(frames, diff, raw, length):
                indices, scores = detect(frames, diff, raw, length)
                traces.append(dict(valid_frames=length, detector_frames=frames.size(0),
                                   candidates=torch.unique(indices).tolist()))
                return indices, scores

            detector.detect = traced

        def attention_hook(layer, inputs, kwargs):
            attention.append(dict(attn_mask=kwargs.get('attn_mask') is not None,
                                  is_causal=kwargs.get('is_causal', False),
                                  key_padding_mask=kwargs.get('key_padding_mask') is not None))

        handle = encoder.conformer1.conformer_layers[0].self_attn.register_forward_pre_hook(
            attention_hook, with_kwargs=True)
        with patch.object(DynamicCompression, 'forward', side_effect=AssertionError('Legacy path invoked')):
            output, lengths = module.model.transcribe(batch.features, batch.feature_lengths)
            selected = traces[:]
            changed = batch.features.clone()
            changed[1, 81:] = 10000
            other, other_lengths = module.model.transcribe(changed, batch.feature_lengths)
            torch.testing.assert_close(other, output)
            assert torch.equal(lengths, other_lengths)
            if not compression.passthrough:
                assert lengths.tolist() == [len(x['candidates']) for x in selected]
                assert [x['detector_frames'] for x in selected] == [
                    (n + encoder.time_reduction_stride - 1) // encoder.time_reduction_stride
                    for n in [101, 81]]
            module.log = lambda *a, **kw: None
            loss = module._step(batch, 0, 'train')
            loss.backward()
        handle.remove()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
        assert all(not x['attn_mask'] and not x['is_causal'] for x in attention)
        assert all(x['key_padding_mask'] for x in attention)
        run_path = Path(config['paths']['exp_dir']) / 'run.json'
        run = json.loads(run_path.read_text()) if run_path.exists() else None
        log_path = run_path.parent / 'logs/train.log'
        guard_lines = ([x for x in log_path.read_text().replace('\r', '\n').splitlines()
                        if '[CustomDataset:' in x] if log_path.exists() else [])
        records.append(dict(scheme=scheme, config=str(config_path.relative_to(REPO)),
                            config_sha256=sha(config_path), no_retired_config_keys=True,
                            effective_sequence_options=options,
                            native_definition=module.offline_native_definition,
                            compression_class=type(compression).__name__,
                            detector_class=None if compression.passthrough else type(compression.detector).__name__,
                            mixer_class=None if compression.passthrough else type(compression.mixer).__name__,
                            passthrough=compression.passthrough, no_legacy_compression_instances=True,
                            total_parameters=sum(p.numel() for p in module.model.parameters()),
                            block_parameters=sum(p.numel() for block in (encoder.conformer1, encoder.conformer2) for p in block.parameters()),
                            compression_parameters=sum(p.numel() for p in compression.parameters()),
                            mel_kwargs=module._mel_kwargs, feature_lengths=batch.feature_lengths.tolist(),
                            reduced_lengths=[(n + encoder.time_reduction_stride - 1) // encoder.time_reduction_stride for n in [101, 81]],
                            compressed_lengths=lengths.tolist(), detector_calls=selected if not compression.passthrough else [],
                            first_attention_call=attention[0], padding_value_invariance=True,
                            synthetic_cpu_loss=float(loss.detach()), finite_backward=True,
                            run=run, dataset_guard_log_lines=guard_lines))
        del module
    result = dict(valid=True, server_utc=datetime.now(timezone.utc).isoformat(),
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  torch_version=torch.__version__, torchaudio_version=torchaudio.__version__,
                  cuda_used=False, prelaunch_source_and_report_hashes_match=True,
                  library_conformer_forward=inspect.getsource(torchaudio.models.Conformer.forward),
                  library_layer_forward=inspect.getsource(torchaudio.models.conformer.ConformerLayer.forward),
                  library_convolution_init=inspect.getsource(torchaudio.models.conformer._ConvolutionModule.__init__),
                  models=records)
    text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
