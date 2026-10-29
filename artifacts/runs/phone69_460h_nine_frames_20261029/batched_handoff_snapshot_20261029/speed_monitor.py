from pathlib import Path
from datetime import datetime,timezone
import json,os,time,traceback
r=Path('/test21/conformer-rnnt-asr');e=r/'exp_phone69_460h_offline_native_stride1_cosd_learned_nine_frames_s42_e50';a=r/'artifacts/runs/phone69_460h_nine_frames_20261029/batched_handoff_20261029';pid=35568;start=None
try:
 deadline=time.monotonic()+900
 while time.monotonic()<deadline:
  d=json.loads((e/'run.json').read_text());step=d.get('global_step',0)
  if d.get('pid')!=pid or d.get('status')!='training':raise RuntimeError('training process or state changed')
  if start is None and step>=25500:
   if step!=25500:raise RuntimeError('missed requested start step')
   start={'global_step':step,'monotonic_seconds':time.monotonic(),'observed_utc':datetime.now(timezone.utc).isoformat()}
  if start and step>=25700:
   if step!=25700:raise RuntimeError('missed requested end step')
   end={'global_step':step,'monotonic_seconds':time.monotonic(),'observed_utc':datetime.now(timezone.utc).isoformat()};elapsed=end['monotonic_seconds']-start['monotonic_seconds']
   out={'status':'complete','backend':'batched','pid':pid,'start':start,'end':end,'elapsed_seconds':elapsed,'steps_per_second':200/elapsed,'interval_seconds':0.25,'validation':'one scheduled validation expected between steps 25500 and 25700','excludes':'startup, preflight, and checkpoint loading'}
   (a/'speed_monotonic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True);break
  time.sleep(.25)
 else:raise TimeoutError('speed measurement timed out')
except BaseException:
 (a/'speed_monotonic_error.log').write_text(traceback.format_exc());raise
