#!/usr/bin/env python3
"""Compare reference and batched cosine compression on real phone69 batches.

Run only on an otherwise idle GPU. This never updates a model or checkpoint.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch, yaml
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule


def timed_compression(module, x, lengths, backend):
    module.execution_backend = backend
    module.record_indices = False
    module.zero_grad(set_to_none=True)
    xin = x.detach().requires_grad_(True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    out, out_lengths = module(xin, lengths)
    out.square().mean().backward()
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    del xin, out, out_lengths
    return duration


def compare_compression(module, x, lengths):
    records = []
    for backend in ['reference', 'batched']:
        module.execution_backend = backend
        module.record_indices = True
        module.zero_grad(set_to_none=True)
        xin = x.detach().requires_grad_(True)
        out, out_lengths = module(xin, lengths)
        out.square().mean().backward()
        torch.cuda.synchronize()
        records.append({'indices': module.last_indices['retained'], 'out': out.detach(),
                        'lengths': out_lengths.detach(), 'input_grad': xin.grad.detach(),
                        'param_grads': [p.grad.detach().clone() for p in module.mixer.parameters()]})
    a, b = records
    if a['indices'] != b['indices']:
        raise AssertionError('Reference and batched frame indices differ')
    torch.testing.assert_close(a['lengths'], b['lengths'], rtol=0, atol=0)
    torch.testing.assert_close(a['out'], b['out'], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(a['input_grad'], b['input_grad'], rtol=5e-4, atol=1e-6)
    for ga, gb in zip(a['param_grads'], b['param_grads']):
        torch.testing.assert_close(ga, gb, rtol=5e-4, atol=1e-6)
    return {'indices_match': True, 'output_max_abs_diff': (a['out']-b['out']).abs().max().item(),
            'input_grad_max_abs_diff': (a['input_grad']-b['input_grad']).abs().max().item()}


def full_step(module, batch, backend, initial_state, record_indices=False):
    module.load_state_dict(initial_state, strict=True)
    module.model.transcriber.compression.execution_backend = backend
    module.model.transcriber.compression.record_indices = record_indices
    module.train()
    module.zero_grad(set_to_none=True)
    torch.manual_seed(20261029)
    torch.cuda.manual_seed_all(20261029)
    features = batch.features.to('cuda'); lengths = batch.feature_lengths.to('cuda')
    targets = batch.targets.to('cuda'); target_lengths = batch.target_lengths.to('cuda')
    previous = torch.cat([targets.new_full((targets.shape[0], 1), module.blank_idx), targets], dim=1)
    torch.cuda.synchronize(); start = time.perf_counter()
    output, encoded_lengths, _, _ = module.model(features, lengths, previous, target_lengths + 1)
    loss = module._rnnt_loss(output, targets, encoded_lengths.to(torch.int32), target_lengths)
    loss.backward(); torch.cuda.synchronize()
    duration = time.perf_counter() - start
    result = {'seconds': duration, 'loss': float(loss.detach()), 'indices': module.model.transcriber.compression.last_indices if record_indices else None, 'encoded_lengths': encoded_lengths.detach().cpu().tolist(),
              'grads': {name: p.grad.detach().cpu().clone() for name, p in module.model.named_parameters() if p.grad is not None}}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args=parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA unavailable'
    assert args.repeats >= 3
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    server=Path('/test21/conformer-rnnt-asr')
    config=server/'configs/server_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_s42_e50.yaml'
    experiment=server/'exp_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_s42_e50'
    run=json.loads((experiment/'run.json').read_text())
    assert run['status']=='trained' and run['epoch']==50 and (experiment/'TRAINING_COMPLETE').exists(), 'Training not complete'
    cfg=yaml.safe_load(config.read_text())
    assert hashlib.sha256(config.read_bytes()).hexdigest()==run['config_sha256']
    model=LibriSpeechRNNTModule.from_config(cfg)
    checkpoint=torch.load(run['epoch_boundary_checkpoint'],map_location='cpu',weights_only=False)
    model.on_load_checkpoint(checkpoint)
    assert checkpoint.get('optimizer_states') and checkpoint.get('lr_schedulers')
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    del checkpoint
    model=model.to('cuda').eval()
    original_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    loader=model.train_dataloader(); dataset=loader.dataset
    probe=json.loads((server/'artifacts/runs/phone69_460h_nine_frames_20261029/memory_probe.json').read_text())
    cases=[r for r in probe['selected_batches'] if r['dataset_index'] in (0,1)][:3]
    assert len(cases)>=2
    report={'experiment':str(experiment),'checkpoint':run['epoch_boundary_checkpoint'],
            'config_sha256':run['config_sha256'],'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(0),'device':'cuda','repeats':args.repeats,
            'note':'Compression timings include forward, backward and batch-level sync; no optimizer or data loading.',
            'cases':[]}
    first_samples=None
    for case in cases:
        ds=dataset.datasets[case['dataset_index']]
        samples=ds[case['batch_index']]
        if first_samples is None:first_samples=samples[:8]
        torch.manual_seed(42);batch=model._train_collate_fn(samples)
        features=batch.features.to('cuda');lengths=batch.feature_lengths.to('cuda')
        encoder=model.model.transcriber
        model.eval()
        with torch.no_grad():
            valid=torch.arange(features.size(1),device='cuda')[None,:] < lengths[:,None]
            x, xlen=encoder.time_reduction(encoder.input_linear(features.masked_fill(~valid[:,:,None],0.0)),lengths)
            x,xlen=encoder.conformer1(x,xlen)
        comp=encoder.compression
        comparison=compare_compression(comp,x,xlen)
        for backend in ['reference','batched']:timed_compression(comp,x,xlen,backend)
        measurements={'reference':[],'batched':[]};peaks={}
        for i in range(args.repeats):
            for backend in (['reference','batched'] if i%2==0 else ['batched','reference']):
                torch.cuda.reset_peak_memory_stats()
                measurements[backend].append(timed_compression(comp,x,xlen,backend))
                peaks[backend]=max(peaks.get(backend,0),torch.cuda.max_memory_allocated()/1024**3)
        old=statistics.median(measurements['reference']);new=statistics.median(measurements['batched'])
        report['cases'].append({'subset':case['subset'],'batch_index':case['batch_index'],
            'utterances':len(samples),'encoded_shape':list(x.shape),'comparison':comparison,
            'seconds':measurements,'median_seconds':{'reference':old,'batched':new},
            'speedup':old/new,'peak_allocated_gib':peaks})
        print(json.dumps(report['cases'][-1]),flush=True)
        del features,lengths,x,xlen,batch,samples
        gc.collect()
    # Full forward/backward on real samples. Restore BN state and RNG before
    # every paired run; model weights are never updated.
    torch.manual_seed(42);batch=model._train_collate_fn(first_samples)
    model.to('cuda');times={'reference':[],'batched':[]}
    outputs={backend:full_step(model,batch,backend,original_state,record_indices=True)
             for backend in ['reference','batched']}
    for i in range(3):
        for backend in (['reference','batched'] if i%2==0 else ['batched','reference']):
            row=full_step(model,batch,backend,original_state)
            times[backend].append(row['seconds'])
    x,y=outputs['reference'],outputs['batched']
    assert x['indices']==y['indices'], 'Full model selected frame indices differ'
    assert x['encoded_lengths']==y['encoded_lengths'], 'Full model lengths differ'
    torch.testing.assert_close(torch.tensor(x['loss']),torch.tensor(y['loss']),rtol=5e-5,atol=1e-4)
    assert x['grads'].keys()==y['grads'].keys()
    for name in x['grads']:
        torch.testing.assert_close(x['grads'][name],y['grads'][name],rtol=1e-3,atol=2e-5,msg=lambda message: name+': '+message)
    old=statistics.median(times['reference']);new=statistics.median(times['batched'])
    report['full_model_smoke']={'utterances':len(first_samples),'training_mode':True,'forward_backward_only':True,
        'loss_reference':x['loss'],'loss_batched':y['loss'],'loss_and_gradients_match':True,
        'seconds':times,'median_seconds':{'reference':old,'batched':new},'speedup':old/new}
    report['correctness_passed']=True
    report['all_compression_cases_faster']=all(row['speedup']>1 for row in report['cases'])
    report['full_model_smoke_faster']=report['full_model_smoke']['speedup']>1
    with args.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report['full_model_smoke']),flush=True)

if __name__=='__main__':main()
