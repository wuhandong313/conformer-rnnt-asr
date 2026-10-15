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
    tmp=state_path.with_suffix('.target90.tmp')
    tmp.write_text(json.dumps(state,indent=2)+'\n')
    tmp.replace(state_path)

while True:
    state=json.loads(state_path.read_text())
    if state['status']=='budgets_finalized': break
    print('waiting for previous checks: '+state['status'],flush=True)
    time.sleep(10)
initial=state['target_budgets']
environment=os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
selected={}
status(status='tuning_90_percent',pid=os.getpid(),target_gpu_memory_fraction=.90)
for config,base in initial.items():
    budget=max(500,round(base['budget']*70.5/base['peak_reserved_GiB']/500)*500)
    lower=0
    upper=None
    best=None
    for attempt in range(7):
        name=Path(config).stem+'_target90_b'+str(budget)
        report_path=out_dir/(name+'.json')
        print('target 90%: {} budget={}'.format(config,budget),flush=True)
        status(active_probe=config,probe_budget=budget)
        with (out_dir/(name+'.log')).open('w') as log:
            result=subprocess.run([python,'-u',str(probe),config,'--budget',str(budget),'--keep-cache','--max-reserved-fraction','0.92','--out',str(report_path)],env=environment,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads(report_path.read_text())
        peak=max((row['peak_reserved_GiB'] for row in report['results']),default=0)
        passed=result.returncode==0 and report['status']=='passed'
        print('status={} peak={:.2f} GiB'.format(report['status'],peak),flush=True)
        if passed:
            item={'budget':budget,'peak_reserved_GiB':peak,'report':str(report_path),'keep_allocator_cache':True}
            if best is None or abs(peak-70.5)<abs(best['peak_reserved_GiB']-70.5): best=item
            if 69<=peak<=72: break
        if not passed or peak>72:
            upper=budget
        else:
            lower=budget
        if upper is not None:
            next_budget=max(500,((lower+upper)//2//500)*500)
        else:
            multiplier=min(2,max(1.05,70.5/max(peak,1)))
            next_budget=max(budget+500,round(budget*multiplier/500)*500)
        if next_budget==budget or (upper is not None and upper-lower<=500): break
        budget=next_budget
    if best is None:
        status(status='failed',error='No safe 90% memory target for '+config)
        raise SystemExit(2)
    selected[config]=best
    status(final_budgets=selected)
status(status='ready_to_apply_budgets',final_budgets=selected)
print(json.dumps(selected,indent=2),flush=True)
