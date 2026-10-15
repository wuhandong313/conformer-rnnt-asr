import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

repo=Path('/test21/conformer-rnnt-asr')
os.chdir(repo)
state_path=repo/'output/phone69_460h/batch_resize_state.json'
out_dir=repo/'output/phone69_460h/resize_probes'
python='/root/anaconda3/envs/whd/bin/python'
probe=repo/'output/phone69_460h/probe_resize_config.py'
def status(**updates):
    state=json.loads(state_path.read_text())
    state.update(updates,updated_at=datetime.now(timezone.utc).isoformat())
    temp=state_path.with_suffix('.target.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n')
    temp.replace(state_path)

while True:
    state=json.loads(state_path.read_text())
    if state['status']=='probes_complete': break
    print('waiting for initial memory probes: '+state['status'],flush=True)
    time.sleep(15)
initial=state['selected_budgets']
environment=os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
selected={}
status(status='tuning_memory_target',pid=os.getpid(),target_gpu_memory_fraction=.80)
for config,base in initial.items():
    budget=max(500,int(base['budget']*62.5/base['peak_reserved_GiB']/500)*500)
    lower=0
    upper=None
    best=None
    for attempt in range(6):
        name=Path(config).stem+'_target80_b'+str(budget)
        report_path=out_dir/(name+'.json')
        print('target 80%: {} budget={}'.format(config,budget),flush=True)
        status(active_probe=config,probe_budget=budget)
        with (out_dir/(name+'.log')).open('w') as log:
            result=subprocess.run([python,'-u',str(probe),config,'--budget',str(budget),'--keep-cache','--out',str(report_path)],env=environment,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads(report_path.read_text())
        peak=max((row['peak_reserved_GiB'] for row in report['results']),default=0)
        passed=result.returncode==0 and report['status']=='passed'
        print('status={} peak={:.2f} GiB'.format(report['status'],peak),flush=True)
        if passed:
            item={'budget':budget,'peak_reserved_GiB':peak,'report':str(report_path),'keep_allocator_cache':True}
            if best is None or abs(peak-62.5)<abs(best['peak_reserved_GiB']-62.5): best=item
            if 60<=peak<=65.5: break
        if not passed or peak>65.5:
            upper=budget
        else:
            lower=budget
        if upper is not None:
            next_budget=max(500,((lower+upper)//2//500)*500)
        else:
            multiplier=min(2,max(1.1,62.5/max(peak,1)))
            next_budget=max(budget+500,int(budget*multiplier/500)*500)
        if next_budget==budget or (upper is not None and upper-lower<=500): break
        budget=next_budget
    if best is None:
        status(status='failed',error='No safe memory target for '+config)
        raise SystemExit(2)
    selected[config]=best
    status(target_budgets=selected)
status(status='target_budgets_ready',target_budgets=selected)
print(json.dumps(selected,indent=2),flush=True)
