import json, statistics, time
import torch
from conformer_rnnt.compression.natural import NaturalPeakCompression
torch.set_num_threads(1)
torch.cuda.set_per_process_memory_fraction(0.02, 0)
torch.manual_seed(20261029)
rows=[]
for window in (5,9):
 for batch, frames in ((4,128),(8,384)):
  module=NaturalPeakCompression(detector={'type':'cosine_diff'},mixer={'type':'learned'},feature_dim=128,window_size=window,weight_predictor_mode={5:'five_frames',9:'nine_frames'}[window]).cuda()
  x=torch.randn(batch,frames,128,device='cuda'); lengths=torch.tensor([frames-i*(frames//(batch*3)) for i in range(batch)],device='cuda',dtype=torch.int32)
  def run(backend,trace):
   module.execution_backend=backend;module.record_indices=trace;module.zero_grad(set_to_none=True)
   v=x.detach().clone().requires_grad_(True);torch.cuda.synchronize();t=time.perf_counter()
   y,l=module(v,lengths);y.square().mean().backward();torch.cuda.synchronize();elapsed=time.perf_counter()-t
   if not trace:return elapsed
   return {'y':y.detach(),'lengths':l.detach(),'indices':module.last_indices,'gx':v.grad.detach(),'gp':[p.grad.detach().clone() for p in module.mixer.parameters()]}
  a,b=run('reference',True),run('batched',True)
  assert a['indices']==b['indices'];torch.testing.assert_close(a['lengths'],b['lengths'],rtol=0,atol=0)
  torch.testing.assert_close(a['y'],b['y'],rtol=3e-5,atol=3e-6);torch.testing.assert_close(a['gx'],b['gx'],rtol=5e-4,atol=1e-6)
  for g,h in zip(a['gp'],b['gp']):torch.testing.assert_close(g,h,rtol=5e-4,atol=1e-6)
  for backend in ('reference','batched'):run(backend,False)
  times={'reference':[],'batched':[]}
  for i in range(5):
   for backend in (('reference','batched') if i%2==0 else ('batched','reference')):times[backend].append(run(backend,False))
  m={k:statistics.median(v) for k,v in times.items()}
  rows.append({'window':window,'shape':[batch,frames,128],'indices_match':True,'output_max_abs_diff':(a['y']-b['y']).abs().max().item(),'input_grad_max_abs_diff':(a['gx']-b['gx']).abs().max().item(),'weight_grad_max_abs_diff':max((g-h).abs().max().item() for g,h in zip(a['gp'],b['gp'])),'median_ms':{k:1000*v for k,v in m.items()},'speedup':m['reference']/m['batched']})
print(json.dumps({'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'data':'synthetic','repeats':5,'timing':'synchronized forward and backward','process_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'results':rows},indent=2),flush=True)
