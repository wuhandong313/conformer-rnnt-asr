import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

repo=Path('/test21/conformer-rnnt-asr')
os.chdir(repo)
state_path=repo/'output/phone69_460h/batch_resize_state.json'
metadata_path=repo/'output/phone69_460h/resize_metadata_report.json'
python='/root/anaconda3/envs/whd/bin/python'
probe=repo/'output/phone69_460h/probe_resize_config.py'
out_dir=repo/'output/phone69_460h/resize_probes'
out_dir.mkdir(parents=True,exist_ok=True)

def status(**updates):
    state=json.loads(state_path.read_text())
    state.update(updates,updated_at=datetime.now(timezone.utc).isoformat())
    temporary=state_path.with_suffix('.probe.tmp')
    temporary.write_text(json.dumps(state,indent=2)+'\n')
    temporary.replace(state_path)

while True:
    state=json.loads(state_path.read_text())
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if state['status']=='failed':
        raise SystemExit('Epoch finishing failed; refusing memory probes')
    if state['status']=='boundary_saved' and metadata.get('status')=='metadata_cached':
        break
    print('waiting: epoch={}, metadata={}'.format(state['status'],metadata.get('status')),flush=True)
    time.sleep(15)

status(status='probing',pid=os.getpid())
recipes=[
 ('configs/server_phone69_460h_stride4_none_s42_e50.yaml',24000,16000),
 ('configs/server_phone69_460h_stride1_cosd_learned_s42_e50.yaml',6000,3000),
 ('configs/server_phone69_460h_stride4_cosd_learned_s42_e50.yaml',48000,16000),
]
selected={}
environment=os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
for config,initial,minimum in recipes:
    budget=initial
    accepted=None
    previous_peak=None
    for trial in range(4):
        name=Path(config).stem+'_b'+str(budget)
        report_path=out_dir/(name+'.json')
        log_path=out_dir/(name+'.log')
        print('probing {} budget={}'.format(config,budget),flush=True)
        status(active_probe=config,probe_budget=budget)
        with log_path.open('w') as log:
            result=subprocess.run([python,'-u',str(probe),config,'--budget',str(budget),'--out',str(report_path)],env=environment,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        passed=result.returncode==0 and report.get('status')=='passed'
        peak=max((row['peak_reserved_GiB'] for row in report.get('results',[])),default=0)
        print('result status={} peak={:.2f} GiB'.format(report.get('status'),peak),flush=True)
        if not passed:
            if accepted:
                break
            if budget<=minimum:
                raise SystemExit('No safe increased batch budget for {}'.format(config))
            budget=max(minimum,budget*2//3)
            continue
        accepted={'budget':budget,'peak_reserved_GiB':peak,'report':str(report_path)}
        if peak>=45 or (previous_peak is not None and peak<previous_peak*1.1):
            break
        previous_peak=peak
        budget=budget*3//2
    assert accepted
    selected[config]=accepted
    status(selected_budgets=selected)
status(status='probes_complete',selected_budgets=selected)
print(json.dumps(selected,indent=2),flush=True)
