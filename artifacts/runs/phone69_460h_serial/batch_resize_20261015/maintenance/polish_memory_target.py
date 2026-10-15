import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

repo=Path('/test21/conformer-rnnt-asr')
os.chdir(repo)
state_path=repo/'output/phone69_460h/batch_resize_state.json'
while True:
    state=json.loads(state_path.read_text())
    if state['status']=='target_budgets_ready': break
    time.sleep(10)
budgets=state['target_budgets']
state.update(status='final_memory_checks',pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat())
state_path.write_text(json.dumps(state,indent=2)+'\n')
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
for config,budget in [
 ('configs/server_phone69_460h_stride4_none_s42_e50.yaml',22000),
 ('configs/server_phone69_460h_stride1_cosd_learned_s42_e50.yaml',15000),
]:
    if budgets[config]['peak_reserved_GiB']>=62: continue
    path=repo/'output/phone69_460h/resize_probes'/(Path(config).stem+'_target80_b'+str(budget)+'.json')
    with path.with_suffix('.log').open('w') as log:
        result=subprocess.run(['/root/anaconda3/envs/whd/bin/python','-u',str(repo/'output/phone69_460h/probe_resize_config.py'),config,'--budget',str(budget),'--keep-cache','--out',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT)
    report=json.loads(path.read_text())
    peak=max((row['peak_reserved_GiB'] for row in report['results']),default=0)
    if result.returncode==0 and report['status']=='passed' and peak<=66:
        budgets[config]={'budget':budget,'peak_reserved_GiB':peak,'report':str(path),'keep_allocator_cache':True}
    print(config,budget,report['status'],peak,flush=True)
state=json.loads(state_path.read_text())
state.update(status='budgets_finalized',target_budgets=budgets,updated_at=datetime.now(timezone.utc).isoformat())
state_path.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state,indent=2),flush=True)
