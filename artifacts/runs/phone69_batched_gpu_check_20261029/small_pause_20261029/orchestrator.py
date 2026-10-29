import json, os, signal, subprocess, sys, tarfile, tempfile, time, traceback
from datetime import datetime, timezone
from pathlib import Path
root=Path('/test21/conformer-rnnt-asr')
base=root/'artifacts/runs/phone69_batched_gpu_check_20261029'
out=base/'small_pause_20261029'
queue=root/'artifacts/runs/phone69_460h_nine_frames_20261029/queue_state.json'
run=root/'exp_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_s42_e50/run.json'
def status(state,**kwargs):
 p=out/'state.json';t=out/'state.tmp';t.write_text(json.dumps({'status':state,'server_utc':datetime.now(timezone.utc).isoformat(),**kwargs},indent=2)+'\n');t.replace(p)
def sigterm(signum,frame):raise RuntimeError('orchestrator received signal '+str(signum))
signal.signal(signal.SIGTERM,sigterm)
pid=None;paused=False;resumed=False
try:
 q=json.loads(queue.read_text());r=json.loads(run.read_text());pid=int(q['training_pid'])
 assert q['status']=='training' and r['status']=='training' and r['pid']==pid
 cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes();assert b'train_phone69_460h_one.py' in cmd
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]);assert free>3500, 'Insufficient free GPU memory: '+str(free)+' MiB'
 with tempfile.TemporaryDirectory(prefix='phone69_small_gpu_') as temp:
  with tarfile.open(base/'benchmark_bundle.tar.gz','r:gz') as a:a.extractall(temp,filter='data')
  status('preparing',training_pid=pid,free_mib=free)
  os.kill(pid,signal.SIGSTOP);paused=True
  for _ in range(100):
   if (Path('/proc')/str(pid)/'status').read_text().split('State:')[1].lstrip().startswith('T'):break
   time.sleep(.05)
  else:raise RuntimeError('Training did not enter stopped state')
  status('testing',training_pid=pid,free_mib=free,step_before=r.get('global_step'))
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=temp,PYTHONDONTWRITEBYTECODE='1')
  with (out/'gpu_small.log').open('x') as log:
   p=subprocess.run([sys.executable,'-u',str(out/'gpu_small_test.py')],cwd=temp,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
  if p.returncode:raise RuntimeError('GPU small test failed, exit '+str(p.returncode))
  report=json.loads((out/'gpu_small.log').read_text());(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
except BaseException as error:
 (out/'error.log').write_text(traceback.format_exc())
 failure=repr(error)
else:failure=None
finally:
 if paused:
  try:os.kill(pid,signal.SIGCONT);resumed=True
  except BaseException as error:failure=(failure or '')+' resume error: '+repr(error)
status('failed' if failure or (paused and not resumed) else 'complete',training_pid=pid,training_resumed=resumed,error=failure)
