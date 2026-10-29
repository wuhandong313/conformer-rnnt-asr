import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch, torchaudio, yaml
from conformer_rnnt.model import conformer_rnnt_model_from_config
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
p=Path(sys.argv[1]); c=yaml.safe_load(p.read_text()); torch.set_num_threads(2); torch.manual_seed(42)
m=conformer_rnnt_model_from_config(c).train()
x=torch.randn(2,47,80,requires_grad=True); lengths=torch.tensor([47,35],dtype=torch.int32)
target=torch.tensor([[1,2,3],[4,5,69]],dtype=torch.int32); tl=torch.tensor([3,2],dtype=torch.int32)
y,enc,_,_=m(x,lengths,torch.cat([target.new_full((2,1),69),target],1),tl+1)
loss=torchaudio.transforms.RNNTLoss(blank=69,reduction='sum',clamp=1.0)(y,target,enc,tl)
loss.backward(); assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
opt=torch.optim.Adam(m.parameters(),lr=.0005); opt.step(); assert all(torch.isfinite(p).all() for p in m.parameters())
module=LibriSpeechRNNTModule.from_config(c); definition=module.offline_native_definition
assert definition['window_size']==9 and definition['weight_predictor_mode']=='nine_frames' and definition['layers']==[2,8]
ckpt={};module.on_save_checkpoint(ckpt);module.on_load_checkpoint(ckpt)
wrong=dict(ckpt); wrong['offline_native_definition']=dict(definition,window_size=5)
try: module.on_load_checkpoint(wrong)
except ValueError: pass
else: raise AssertionError('Mismatched checkpoint accepted')
print(json.dumps(dict(status='passed', parameters=sum(p.numel() for p in m.parameters()), loss=float(loss.detach()), lengths=enc.tolist(), definition=definition),indent=2))
