#!/usr/bin/env python3
from datetime import datetime, timezone
from pathlib import Path
import csv, json, math, os, subprocess, sys
root=Path('/test21/conformer-rnnt-asr'); out=root/'artifacts/evaluations/phone69_460h_placement_stride1_testclean_cpu_20261025'
plan=json.loads((out/'run_plan.json').read_text()); state_path=out/'eval_state.json'
def now(): return datetime.now(timezone.utc).isoformat()
def save_state(state):
    tmp=state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n'); os.replace(tmp,state_path)
state={'status':'running','started_at_utc':now(),'device':'cpu','current':None,'completed':[],'experiments':{x['scheme']:{'status':'queued'} for x in plan['experiments']}}
save_state(state)
references=None; summary={'status':'running','device':'cpu','test_subset':'test-clean','metric':'per','beam_width':20,'experiments':[]}
try:
  for item in plan['experiments']:
    state['current']=item['scheme']; state['experiments'][item['scheme']]={'status':'evaluating','started_at_utc':now()}; save_state(state)
    env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
    cmd=['/root/anaconda3/envs/whd/bin/python','-u','eval.py',item['eval_config'],'--checkpoint',item['checkpoint'],'--metric','per','--out',item['raw_csv'],'--cpu']
    log_path=root/item['log']
    with log_path.open('w') as log:
      proc=subprocess.Popen(cmd,cwd=str(root),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
      state['experiments'][item['scheme']]['pid']=proc.pid; save_state(state)
      rc=proc.wait()
    if rc != 0: raise RuntimeError(item['scheme']+' eval.py exited '+str(rc))
    with (root/item['raw_csv']).open(newline='') as f: rows=list(csv.DictReader(f))
    if len(rows) != plan['expected_samples']: raise RuntimeError(item['scheme']+' sample count mismatch: '+str(len(rows)))
    refs=[(int(row['idx']),row['ref']) for row in rows]
    if references is None: references=refs
    elif refs != references: raise RuntimeError('reference sequence differs for '+item['scheme'])
    errors=sum(int(row['phone_errors']) for row in rows); phones=sum(int(row['reference_phones']) for row in rows)
    if phones != plan['expected_reference_phones']: raise RuntimeError(item['scheme']+' reference-phone count mismatch: '+str(phones))
    rtfs=[float(row['RTF']) for row in rows]
    if not all(math.isfinite(x) and x >= 0 for x in rtfs): raise RuntimeError(item['scheme']+' invalid RTF values')
    result={'scheme':item['scheme'],'experiment':item['name'],'status':'complete','checkpoint':item['checkpoint'],
            'num_samples':len(rows),'phone_errors':errors,'reference_phones':phones,'PER':errors/phones,
            'mean_utterance_RTF':sum(rtfs)/len(rtfs),'raw_csv':item['raw_csv'],'log':item['log'],
            'eval_config':item['eval_config'],'eval_config_sha256':item['eval_config_sha256']}
    summary['experiments'].append(result); summary['status']='running'
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    state['completed'].append(item['scheme']); state['experiments'][item['scheme']].update({'status':'complete','result':result,'finished_at_utc':now()}); state['current']=None; save_state(state)
  summary['status']='complete'; summary['completed_at_utc']=now(); (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
  (out/'RAW_EVALUATIONS_COMPLETE').write_text(now()+'\n')
  state.update({'status':'complete','current':None,'completed_at_utc':now()}); save_state(state)
except BaseException as e:
  state.update({'status':'failed','failed_at_utc':now(),'error':repr(e)}); save_state(state)
  summary.update({'status':'failed','error':repr(e),'failed_at_utc':now()}); (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
  raise
