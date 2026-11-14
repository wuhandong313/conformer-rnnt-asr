#!/usr/bin/env python3
"""Retire only the old queue parent, retain its L3 worker, then train L9.

The relocated Mel+L3/Mel+L7/L2+L7 recipes are never launched here. The old
deployment/manifest stays immutable; handoff identities and progress are audited.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.server.phone69_placement_supplement import PlacementQueue, load_plan, sha, save, verify_completed

PLAN = ROOT / 'configs/phone69_placement_current_server_layer9_20261114.json'
ORIGINAL = ROOT / 'output/phone69_460h_placement_recovery_20261114'
OUT = ROOT / 'output/phone69_460h_placement_local_handoff_20261114'


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def process_matches(pid, filename):
    path = Path('/proc') / str(pid)
    try:
        if (path/'stat').read_text().split()[2] == 'Z':
            return False
        names = {Path(a).name for a in (path/'cmdline').read_bytes().decode(errors='replace').split('\0') if a}
        return filename in names
    except (OSError,IndexError):
        return False


def retire_queue_parent(controller_pid, training_pid):
    if controller_pid == training_pid or not process_matches(training_pid,'train_phone69_460h_one.py'):
        raise ValueError('The active training worker identity is not intact')
    if not process_matches(controller_pid,'recover_phone69_placement_supplement.py'):
        raise ValueError('The old queue parent identity changed; do not signal an unrelated process')
    os.kill(controller_pid,signal.SIGTERM)  # Single PID, never process group or training PID.
    for _ in range(50):
        if not process_matches(controller_pid,'recover_phone69_placement_supplement.py'):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError('Old queue parent did not exit; preserve the active training worker')
    if not process_matches(training_pid,'train_phone69_460h_one.py'):
        raise RuntimeError('Training worker is no longer alive; inspect rather than restarting')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['validate','run'],required=True)
    args=parser.parse_args()
    _,plan=load_plan(PLAN)
    if [x['name'] for x in plan['experiments']] != ['layer9']:
        raise ValueError('Current-server continuation must contain only L9')
    if args.mode=='validate':
        print(json.dumps(dict(valid=True,after_running_layer3=['layer9'])))
        return
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
    OUT.mkdir(parents=True,exist_ok=False)
    def state(status,**fields):
        value=dict(status=status,controller_pid=os.getpid(),server_utc=now(),**fields)
        save(OUT/'handoff_state.json',value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
    try:
        old=read(ORIGINAL/'recovery_state.json')
        if old['status']!='training' or old['experiment']!='layer3':
            raise ValueError('Expected the existing L3 training job; inspect current state first')
        old_deployment=read(ORIGINAL/'deployment.json')
        fingerprints=dict(old_deployment['source_sha256'])
        for relative in ['scripts/server/continue_phone69_placement_local.py',
                         'configs/phone69_placement_current_server_layer9_20261114.json']:
            fingerprints[relative]=sha(ROOT/relative)
        for relative,digest in fingerprints.items():
            if sha(ROOT/relative)!=digest:
                raise ValueError('Frozen source/configuration changed: '+relative)
        run_path=Path('/test21/conformer-rnnt-asr/exp_phone69_460h_offline_native_placement_layer3_stride1_s42_e50_supp20261112/run.json')
        run=read(run_path)
        training_pid=old['training_pid']
        if run['pid']!=training_pid or run['status']!='training' or run['max_token_limit']!=8800:
            raise ValueError('Running L3 identity differs from the recovery record')
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        save(OUT/'handoff_request.json',dict(code_commit=source_commit,original_state=old,
            original_run_config_sha256=run['config_sha256'],training_pid_preserved=training_pid,
            old_controller_pid=old['controller_pid'],deferred_to_other_server=['mel_layer3','mel_layer7','layers2_7_n3'],
            current_server_next=['layer9'],source_sha256=fingerprints))
        state('retiring_old_queue_parent',old_controller_pid=old['controller_pid'],training_pid=training_pid)
        retire_queue_parent(old['controller_pid'],training_pid)
        with open('/tmp/phone69_460h_serial.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with (OUT/'layer3_gpu_usage.csv').open('x') as gpu:
                while process_matches(training_pid,'train_phone69_460h_one.py'):
                    current=read(run_path)
                    state('waiting_for_running_layer3',training_pid=training_pid,
                        epoch=current.get('epoch'),global_step=current.get('global_step'),budget=8800,
                        remaining_here=['layer9'],deferred_to_other_server=['mel_layer3','mel_layer7','layers2_7_n3'])
                    subprocess.run(['nvidia-smi','-i','0','--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader'],stdout=gpu,stderr=subprocess.DEVNULL)
                    gpu.flush()
                    time.sleep(15)
            for relative,digest in fingerprints.items():
                if sha(ROOT/relative)!=digest:
                    raise ValueError('Frozen source/configuration changed: '+relative)
            current=read(run_path)
            if current.get('config_sha256')!=run['config_sha256'] or current.get('pid')!=training_pid:
                raise ValueError('L3 worker/configuration changed during handoff')
            original_plan=read(ROOT/'configs/phone69_placement_supplement_20261112.json')
            item=next(x for x in original_plan['experiments'] if x['name']=='layer3')
            item=dict(item,action='reuse')
            result=verify_completed(item,budget=8800)
            save(OUT/'layer3_complete.json',result)
        state('starting_layer9_only',completed=[result],remaining_here=['layer9'])
        queue=PlacementQueue(PLAN)
        queue.run()
        state('complete',completed=[result]+queue.completed,deferred_to_other_server=['mel_layer3','mel_layer7','layers2_7_n3'])
    except BaseException as error:
        state('failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
