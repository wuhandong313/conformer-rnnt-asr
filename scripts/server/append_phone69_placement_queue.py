#!/usr/bin/env python3
"""Append the Mel placement runs after the existing local L3/L9 queue."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.server.phone69_placement_supplement import PlacementQueue, load_plan, sha, save, verify_completed

PLAN=ROOT/'configs/phone69_placement_current_server_three_20261115.json'
PREDECESSOR=ROOT/'output/phone69_460h_placement_local_handoff_20261114'
LAYER9=ROOT/'output/phone69_460h_placement_current_server_layer9_20261114'
OUT=ROOT/'output/phone69_460h_placement_append_20261115'
NAMES=['mel_layer3','mel_layer7']


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def matches(pid,filename):
    path=Path('/proc')/str(pid)
    try:
        return ((path/'stat').read_text().split()[2]!='Z' and filename in
                {Path(a).name for a in (path/'cmdline').read_bytes().decode(errors='replace').split('\0') if a})
    except (OSError,IndexError):
        return False


def predecessor_done(state,alive):
    if not state:
        raise ValueError('Existing local queue state is missing')
    if state.get('status')=='failed':
        raise RuntimeError('Existing local queue failed; no subsequent experiments will start')
    if alive:
        return False
    if state.get('status')!='complete':
        raise RuntimeError('Existing local queue exited without verified completion')
    return True


def verify_predecessor():
    if not read(LAYER9/'completion.json'):
        raise ValueError('L9 completion record is missing')
    original=read(ROOT/'configs/phone69_placement_supplement_20261112.json')
    results=[]
    for name in ['layer3','layer9']:
        item=copy.deepcopy(next(i for i in original['experiments'] if i['name']==name))
        item['action']='reuse'
        budget=8800
        if name=='layer9':
            gate=read(LAYER9/'layer9_selected.json')
            if not gate or not gate.get('valid'):
                raise ValueError('L9 selected budget is not verified')
            if sha(ROOT/gate['effective_config'])!=gate['effective_config_sha256']:
                raise ValueError('L9 selected configuration changed')
            budget=gate['budget']
        result=verify_completed(item,budget=budget)
        if name=='layer9' and result['config_sha256']!=gate['effective_config_sha256']:
            raise ValueError('L9 actual recipe differs from selected recipe')
        results.append(result)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['validate','run','status'],required=True)
    args=parser.parse_args()
    _,plan=load_plan(PLAN)
    if [i['name'] for i in plan['experiments']]!=NAMES or any(i['action']!='train' for i in plan['experiments']):
        raise ValueError('Append manifest must contain the fresh Mel+L3 and Mel+L7 runs')
    if args.mode=='validate':
        print(json.dumps(dict(valid=True,after=['layer3','layer9'],append=NAMES)))
        return
    if args.mode=='status':
        print(json.dumps(dict(state=read(OUT/'append_state.json'),request=read(OUT/'append_request.json'),
            predecessor=read(PREDECESSOR/'handoff_state.json'),layer9_queue=read(LAYER9/'queue_state.json'),
            new_queue=read(ROOT/plan['runtime_dir']/'queue_state.json')),ensure_ascii=False))
        return
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    OUT.mkdir(parents=True,exist_ok=False)
    def state(status,**fields):
        value=dict(status=status,controller_pid=os.getpid(),server_utc=datetime.now(timezone.utc).isoformat(),**fields)
        save(OUT/'append_state.json',value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
    try:
        prior=read(PREDECESSOR/'handoff_state.json')
        if not prior:
            raise ValueError('Existing L3/L9 supervisor is missing')
        predecessor_pid=prior['controller_pid']
        request=read(PREDECESSOR/'handoff_request.json')
        sources=dict(request['source_sha256'])
        for relative in ['scripts/server/append_phone69_placement_queue.py',str(PLAN.relative_to(ROOT))]:
            sources[relative]=sha(ROOT/relative)
        def unchanged():
            changed=[p for p,h in sources.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
            if changed:
                raise RuntimeError('Source/configuration changed: '+', '.join(changed))
        unchanged()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        save(OUT/'append_request.json',dict(code_commit=commit,source_sha256=sources,
            predecessor_controller_pid=predecessor_pid,plan=str(PLAN.relative_to(ROOT)),
            execution_order=['layer3','layer9']+NAMES,training_processes_untouched=True))
        while True:
            prior=read(PREDECESSOR/'handoff_state.json')
            if prior.get('controller_pid')!=predecessor_pid:
                raise ValueError('Predecessor supervisor identity changed')
            if predecessor_done(prior,matches(predecessor_pid,'continue_phone69_placement_local.py')):
                break
            layer9=read(LAYER9/'queue_state.json') or {}
            state('waiting_for_existing_l3_l9',predecessor_controller_pid=predecessor_pid,
                predecessor_phase=prior['status'],layer9_phase=layer9.get('status'),append=NAMES)
            time.sleep(20)
        unchanged()
        with open('/tmp/phone69_460h_serial.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            results=verify_predecessor()
            save(OUT/'predecessor_verified.json',results)
        state('starting_returned_placements',completed_predecessor=results,append=NAMES)
        queue=PlacementQueue(PLAN)
        queue.run()
        state('complete',completed=results+queue.completed)
    except BaseException as error:
        state('failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
