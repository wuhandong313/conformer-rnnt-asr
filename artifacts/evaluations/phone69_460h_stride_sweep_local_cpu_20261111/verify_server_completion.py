import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

root=Path('/test21/conformer-rnnt-asr')
sys.path.insert(0,str(root))
from conformer_rnnt.checkpoint_utils import phone69_resume_metadata

out=root/'output/phone69_460h_stride_sweep_recovery_20261111'
original=root/'output/phone69_460h_stride_sweep_proj128_20261108'
def read(path):
    return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

state=read(out/'recovery_state.json')
completion=read(out/'completion.json')
assert state['status']=='complete'
assert completion['models']==state['completed']
assert sorted(g['stride'] for g in completion['models'])==[1,2,8]
source=state['identity']['source_sha256']
assert len(source)==29
for relative,digest in source.items():
    assert sha(root/relative)==digest, relative
records=[]
for group in completion['models']:
    stride=group['stride']
    exp=root/('exp_phone69_460h_offline_native_stride%d_none_proj128_s42_e50'%stride)
    run=read(exp/'run.json')
    gate=read(original/('stride%d_proj128_selected.json'%stride))
    assert gate['valid']
    cfg_path=root/gate['effective_config']
    assert sha(cfg_path)==gate['effective_config_sha256']==group['original_config_sha256']
    cfg=yaml.safe_load(cfg_path.read_text())
    assert run['status']=='trained' and run['epoch']==50 and run['epochs']==50
    assert run['max_token_limit']==group['budget']==gate['budget']
    assert run['internal_dim']==128*stride and run['time_reduction_stride']==stride
    assert cfg['seed']==42 and cfg['training']['epochs']==50
    allowed={gate['effective_config_sha256']}
    only_checkpoint_changed=True
    if stride==8:
        resume_path=Path(state['config'])
        assert sha(resume_path)==state['config_sha256']
        resume=yaml.safe_load(resume_path.read_text())
        expected=copy.deepcopy(cfg)
        expected['paths']['from_checkpoint']=state['resume_checkpoint']
        only_checkpoint_changed=resume==expected
        assert only_checkpoint_changed
        allowed.add(state['config_sha256'])
    assert run['config_sha256'] in allowed
    best=Path(run['best_checkpoint'])
    assert best.is_file() and best==Path(group['best_checkpoint'])
    assert (exp/'best_checkpoint.txt').read_text().strip()==str(best)
    assert sha(best)==group['best_checkpoint_sha256']
    boundary=Path(run['epoch_boundary_checkpoint'])
    assert boundary.is_file() and str(boundary)==group['epoch_boundary_checkpoint']
    checkpoint=torch.load(boundary,map_location='cpu',weights_only=False)
    metadata=phone69_resume_metadata(checkpoint,group['budget'])
    assert metadata['resume_completed_epochs']==50 and metadata['resume_at_epoch_boundary']
    assert metadata['resume_global_step']==run['global_step']==group['global_step']
    assert metadata==group['epoch_boundary_metadata']
    assert checkpoint.get('state_dict') and checkpoint.get('optimizer_states') and checkpoint.get('lr_schedulers')
    records.append(dict(stride=stride,status=run['status'],epoch=run['epoch'],global_step=run['global_step'],
        budget=group['budget'],best_checkpoint=str(best),best_checkpoint_sha256=group['best_checkpoint_sha256'],
        best_exists=True,best_marker_matches=True,epoch_boundary_checkpoint=str(boundary),
        epoch_boundary_exists=True,epoch_boundary_metadata=metadata,optimizer_states_present=True,
        scheduler_states_present=True,original_config_sha256=group['original_config_sha256'],
        actual_run_config_sha256=run['config_sha256'],only_resume_checkpoint_changed=only_checkpoint_changed,
        actual_training_commit=run['code_commit']))
    del checkpoint
print(json.dumps(dict(valid=True,server_utc=datetime.now(timezone.utc).isoformat(),
    recovery_status=state['status'],source_files_verified=len(source),completion=completion,
    completion_sha256=sha(out/'completion.json'),models=records),ensure_ascii=False))
