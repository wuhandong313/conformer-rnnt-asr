import json,statistics,time,hashlib
from pathlib import Path
import torch,yaml
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
from scripts.server.benchmark_phone69_batched_cosine import compare_compression,timed_compression
torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(0.04,0)
root=Path('/test21/conformer-rnnt-asr');exp=root/'exp_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_s42_e50'
run=json.loads((exp/'run.json').read_text());cfgpath=Path(run['config']);assert hashlib.sha256(cfgpath.read_bytes()).hexdigest()==run['config_sha256']
cfg=yaml.safe_load(cfgpath.read_text());model=LibriSpeechRNNTModule.from_config(cfg)
ck=torch.load(run['epoch_boundary_checkpoint'],map_location='cpu',weights_only=False);assert ck.get('optimizer_states') and ck.get('lr_schedulers');model.on_load_checkpoint(ck);model.load_state_dict(ck['state_dict'],strict=True);del ck
model=model.cuda().eval();dataset=model.train_dataloader().dataset
pre=json.loads((root/'artifacts/runs/phone69_460h_nine_frames_20261029/memory_probe.json').read_text());cases=[c for c in pre['selected_batches'] if c['dataset_index'] in (0,1)][:2]
assert len(cases)==2;rows=[]
for case in cases:
 samples=dataset.datasets[case['dataset_index']][case['batch_index']][:4]
 torch.manual_seed(42);batch=model._train_collate_fn(samples);features=batch.features.cuda();lengths=batch.feature_lengths.cuda();enc=model.model.transcriber
 with torch.no_grad():
  valid=torch.arange(features.size(1),device='cuda')[None,:]<lengths[:,None];x,l=enc.time_reduction(enc.input_linear(features.masked_fill(~valid[:,:,None],0.0)),lengths);x,l=enc.conformer1(x,l)
 c=compare_compression(enc.compression,x,l)
 for b in ['reference','batched']:timed_compression(enc.compression,x,l,b)
 times={'reference':[],'batched':[]}
 for i in range(3):
  for b in (['reference','batched'] if i%2==0 else ['batched','reference']):times[b].append(timed_compression(enc.compression,x,l,b))
 med={k:statistics.median(v) for k,v in times.items()};rows.append({'subset':case['subset'],'batch_index':case['batch_index'],'shape':list(x.shape),'comparison':c,'median_ms':{k:1000*v for k,v in med.items()},'speedup':med['reference']/med['batched']})
print('PHONE69_JSON:'+json.dumps({'samples_per_case':4,'checkpoint':run['epoch_boundary_checkpoint'],'model_eval_first_two_layers':True,'compression_forward_backward':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'cases':rows},indent=2),flush=True)
