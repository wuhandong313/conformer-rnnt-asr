#!/usr/bin/env python3
"""Run the approved isolated GPU benchmark after the active training completes."""
import fcntl,hashlib,json,os,subprocess,sys,tarfile,tempfile,time
from datetime import datetime,timezone
from pathlib import Path
root=Path('/test21/conformer-rnnt-asr');audit=root/'artifacts/runs/phone69_batched_gpu_check_20261029';bundle=audit/'benchmark_bundle.tar.gz'
train=root/'exp_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_s42_e50'
queue=root/'artifacts/runs/phone69_460h_nine_frames_20261029/queue_state.json'

def state(status,**extra):
 data={'status':status,'server_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),**extra};p=audit/'state.json';tmp=audit/'state.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p);print(json.dumps(data),flush=True)
def gpu_apps():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return p.stdout.strip()
def main():
 metadata=json.loads((audit/'bundle_manifest.json').read_text());assert hashlib.sha256(bundle.read_bytes()).hexdigest()==metadata['bundle_sha256']
 state('waiting_for_training')
 while True:
  q=json.loads(queue.read_text()) if queue.exists() else {}
  d=json.loads((train/'run.json').read_text()) if (train/'run.json').exists() else {}
  if q.get('status')=='failed' or d.get('status') in ('failed','interrupted'):
   state('blocked_training_failed',message='Training failed or interrupted; benchmark was not started');return 1
  if q.get('status')=='complete' and d.get('status')=='trained' and d.get('epoch')==50 and (train/'TRAINING_COMPLETE').exists():
   with open('/tmp/phone69_460h_serial.lock','a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:time.sleep(30);continue
    if gpu_apps():
     fcntl.flock(lock,fcntl.LOCK_UN);time.sleep(30);continue
    state('gpu_tests',training_completed=True)
    with tempfile.TemporaryDirectory(prefix='phone69_gpu_benchmark_') as temp:
     with tarfile.open(bundle,'r:gz') as archive:archive.extractall(temp,filter='data')
     env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
     with (audit/'gpu_tests.log').open('x') as out:
      p=subprocess.run([sys.executable,'-u','-m','unittest','discover','-s','tests','-p','test_batched_cosine.py'],cwd=temp,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=300)
     if p.returncode:
      state('failed_gpu_equivalence',test_exit=p.returncode,log=str(audit/'gpu_tests.log'));return 1
     for attempt in range(3):
      if not gpu_apps():break
      time.sleep(2)
     else:
      state('blocked_gpu_occupied_after_test',message='Another process acquired GPU; no benchmark started');return 1
     state('benchmarking',training_completed=True,tests_passed=True)
     with (audit/'benchmark.log').open('x') as out:
      p=subprocess.run([sys.executable,'-u','scripts/server/benchmark_phone69_batched_cosine.py','--out',str(audit/'benchmark_report.json'),'--repeats','5'],cwd=temp,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=3600)
     if p.returncode:
      state('failed_benchmark',benchmark_exit=p.returncode,log=str(audit/'benchmark.log'));return 1
     report=json.loads((audit/'benchmark_report.json').read_text())
     state('complete',correctness_passed=report['correctness_passed'],all_compression_cases_faster=report['all_compression_cases_faster'],full_model_smoke_faster=report['full_model_smoke_faster'],report=str(audit/'benchmark_report.json'))
     return 0
  time.sleep(30)
if __name__=='__main__':
 try:sys.exit(main())
 except BaseException as error:
  state('failed',error=repr(error));raise
