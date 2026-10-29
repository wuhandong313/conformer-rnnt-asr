#!/usr/bin/env python3
"""Run a fresh stride-1 nine-frame experiment after immutable preflight."""
import fcntl, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
AUDIT=ROOT/'artifacts/runs/phone69_460h_nine_frames_20261029'
CONFIG=ROOT/'configs/server_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_s42_e50.yaml'
EXP=ROOT/'exp_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_s42_e50'
AUDIT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
def state(status, **fields):
    x=dict(status=status,server_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**fields); save(AUDIT/'queue_state.json',x);print(json.dumps(x),flush=True)
def run(args,name):
    with (AUDIT/name).open('x') as f:
        return subprocess.run([sys.executable,'-u',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT).returncode
def gpu_idle():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if result: raise RuntimeError('GPU occupied: '+result)
def main():
    with open('/tmp/phone69_460h_serial.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not EXP.exists(), 'Experiment already exists; refusing to overwrite or restart'
        c=yaml.safe_load(CONFIG.read_text()); m=c['model']; comp=m['compression']
        assert c['paths']['from_checkpoint'] is None and c['paths']['exp_dir']==str(EXP)
        assert m['sequence_mode']=='offline' and m['time_reduction_stride']==1 and (m['num_layers_block1'],m['num_layers_block2'])==(2,8)
        assert comp['weight_predictor_mode']=='nine_frames' and comp['hanning_window_size']==9 and 'keep_rate' not in comp and 'right_context_length' not in m
        gpu_idle();state('preflight',stage='regression_tests')
        assert run(['-m','unittest','discover','-s','tests'],'tests.log')==0, 'Tests failed; see tests.log'
        state('preflight',stage='synthetic_rnnt')
        assert run(['scripts/server/validate_phone69_nine_frames.py',CONFIG],'synthetic.log')==0, 'Synthetic validation failed'
        gpu_idle();state('preflight',stage='real_batch_memory',budget=c['data']['max_token_limit'])
        report=AUDIT/'memory_probe.json'
        assert run(['scripts/server/probe_phone69_training_memory.py',CONFIG,'--out',report,'--keep-cache','--max-reserved-fraction','0.9'],'memory_probe.log')==0, 'Memory preflight failed'
        result=json.loads(report.read_text()); assert result['status']=='passed' and result['config_sha256']==sha(CONFIG)
        assert all(sha(ROOT/p)==digest for p,digest in result['source_sha256'].items())
        sources=[*sorted((ROOT/'conformer_rnnt').rglob('*.py')),*[ROOT/p for p in ['common.py','train.py','scripts/server/train_phone69_460h_one.py','scripts/server/phone69_epoch_checkpoint.py','scripts/server/phone69_nine_frames_pipeline.py']]]
        gate=dict(valid=True,config=str(CONFIG),config_sha256=sha(CONFIG),sources={str(p.relative_to(ROOT)):sha(p) for p in sources},probe=str(report),server_utc=datetime.now(timezone.utc).isoformat())
        save(AUDIT/'prelaunch_validation.json',gate)
        assert all(sha(ROOT/p)==h for p,h in gate['sources'].items()) and sha(CONFIG)==gate['config_sha256']
        gpu_idle();assert not EXP.exists()
        (EXP/'logs').mkdir(parents=True)
        with (EXP/'logs/train.log').open('x') as log,(EXP/'logs/gpu_usage.csv').open('x') as gpu:
            child=subprocess.Popen([sys.executable,'-u','scripts/server/train_phone69_460h_one.py',str(CONFIG)],cwd=ROOT,env=dict(os.environ,RESUME_LAST='0'),stdout=log,stderr=subprocess.STDOUT)
            state('training',training_pid=child.pid,experiment=str(EXP),budget=c['data']['max_token_limit'])
            while child.poll() is None:
                subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader','-i','0'],stdout=gpu,stderr=subprocess.DEVNULL);gpu.flush()
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:pass
        assert child.returncode==0, 'Training exited unsuccessfully; preserved logs'
        import torch
        sys.path.insert(0,str(ROOT))
        from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
        runstate=json.loads((EXP/'run.json').read_text()); assert runstate['status']=='trained' and runstate['epoch']==50
        definition=LibriSpeechRNNTModule.from_config(c).offline_native_definition
        for key in ['best_checkpoint','epoch_boundary_checkpoint']:
            ck=torch.load(runstate[key],map_location='cpu',weights_only=False)
            assert ck.get('state_dict') and ck.get('optimizer_states') and ck.get('lr_schedulers') and ck.get('offline_native_definition')==definition
            if key=='epoch_boundary_checkpoint':
                cur=ck['loops']['fit_loop']['epoch_progress']['current'];assert cur['ready']==cur['completed'] and ck['phone69_epoch_boundary']['resume_completed_epochs']==50
        (EXP/'TRAINING_COMPLETE').touch(exist_ok=False)
        state('complete',experiment=str(EXP),global_step=runstate['global_step'],best_checkpoint=runstate['best_checkpoint'])
if __name__=='__main__':
    try:main()
    except BaseException as e:
        state('failed',error=repr(e),message='Logs and checkpoints preserved; no automatic restart');raise
