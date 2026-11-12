"""Collect missing stage lengths on the home CPU using frozen evaluation inputs.

Encoder forward only: no beam search, training, MAC hooks or timing benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(path, digest):
    if sha(path) != digest:
        raise ValueError('Input hash mismatch: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve the existing calibration file')
    manifest = json.loads(args.manifest.read_text())
    entry = manifest['model']
    source = Path(manifest['source'])
    for name, digest in manifest['code_sha256'].items():
        verify(source / name, digest)
    for name in ('checkpoint', 'config'):
        verify(entry[name], entry[name + '_sha256'])
    sys.path.insert(0, str(source))
    import torch
    import torchaudio
    import yaml
    from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
    from torchaudio.models.conformer import ConformerLayer
    if not Path(sys.modules[LibriSpeechRNNTModule.__module__].__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError('Model source is not the frozen evaluation source')
    if torch.__version__ != manifest['runtime']['torch'] or torchaudio.__version__ != manifest['runtime']['torchaudio']:
        raise ValueError('Use the archived runtime')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg = yaml.safe_load(Path(entry['config']).read_text())
    verify(cfg['paths']['global_stats_path'], manifest['global_stats_sha256'])
    verify(cfg['tokenizer']['vocab_path'], manifest['vocab_sha256'])
    model = LibriSpeechRNNTModule.from_config(cfg)
    ck = torch.load(entry['checkpoint'], map_location='cpu', weights_only=False)
    model.on_load_checkpoint(ck)
    model.load_state_dict(ck['state_dict'], strict=True)
    del ck
    model.cpu().eval()
    loader = model.test_dataloader()
    audio = manifest['audio']
    if (len(audio) != 2620 or len(loader) != 2620 or loader.batch_size != 1
            or loader.num_workers != 0
            or loader.dataset.datasets[0]._walker != [a['uid'] for a in audio]):
        raise ValueError('Unexpected evaluation corpus')
    root = Path(cfg['paths']['dataset_path']) / 'LibriSpeech/test-clean'
    measured_layers = []
    def record_layer(module, inputs):
        measured_layers.append(int(inputs[0].shape[0]))
    for layer in model.model.transcriber.modules():
        if isinstance(layer, ConformerLayer):
            layer.register_forward_pre_hook(record_layer)
    rows = []
    with torch.inference_mode():
        for i, (batch, _) in enumerate(loader):
            item = audio[i]
            speaker, chapter, _ = item['uid'].split('-')
            verify(root / speaker / chapter / (item['uid'] + '.flac'), item['sha256'])
            raw = int(batch.feature_lengths[0])
            if raw != item['samples'] // cfg['data']['hop_length'] + 1:
                raise ValueError('Unexpected input feature length')
            encoder = model.model.transcriber
            measured_layers.clear()
            _, output = encoder(batch.features, batch.feature_lengths)
            observed = encoder.compression.last_lengths
            by_site = {s['after_layer']: s for s in observed['stages']}
            current, layer_frames = raw, []
            for site in range(cfg['model']['num_layers'] + 1):
                if site in by_site:
                    if by_site[site]['input'] != [current]:
                        raise ValueError('Inconsistent stage input')
                    current = by_site[site]['output'][0]
                if site == 0:
                    stride = cfg['model']['time_reduction_stride']
                    current = (current + stride - 1) // stride
                if site < cfg['model']['num_layers']:
                    layer_frames.append(current)
            if current != int(output[0]) or layer_frames != measured_layers:
                raise ValueError('Inconsistent encoder output')
            rows.append(dict(uid=item['uid'], input_mel_frames=raw, layer_frames=layer_frames,
                             output_steps=current, stages=observed['stages']))
            if (i + 1) % 250 == 0:
                print(json.dumps({'calibrated_utterances': i + 1, 'total': len(audio)}), flush=True)
    result = dict(status='complete', method='encoder-only stage length calibration',
                  checkpoint_sha256=entry['checkpoint_sha256'], config_sha256=entry['config_sha256'],
                  manifest_sha256=sha(args.manifest), calibration_script_sha256=sha(__file__),
                  frozen_source_sha256=manifest['code_sha256'], utterances=len(rows), rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'utterances': len(rows), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
