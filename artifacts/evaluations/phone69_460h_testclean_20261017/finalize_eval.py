from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import time

out = Path('/test21/conformer-rnnt-asr/artifacts/evaluations/phone69_460h_testclean_20261017')
state_path = out / 'eval_state.json'
def write_state(status, **extra):
    state = {'status': status, 'updated_at_utc': datetime.now(timezone.utc).isoformat(), **extra}
    tmp = state_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(state_path)

while not (out / 'RAW_EVALUATIONS_COMPLETE').exists():
    active = subprocess.run(['tmux', 'has-session', '-t', '=phone69_460h_eval_20261017'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not active:
        write_state('failed', reason='Raw evaluation session ended before all three evaluations completed; preserve logs for diagnosis.')
        raise SystemExit(1)
    write_state('evaluating', aggregation='S/D/I will be computed from every full reference/hypothesis pair after evaluation.')
    time.sleep(30)

write_state('aggregating')
with (out / 'aggregation.log').open('a') as log:
    completed = subprocess.run(['/root/anaconda3/envs/whd/bin/python', '-u', str(out / 'aggregate_phone_edits.py')], stdout=log, stderr=subprocess.STDOUT)
if completed.returncode:
    write_state('failed', reason='S/D/I aggregation failed', returncode=completed.returncode)
    raise SystemExit(completed.returncode)
write_state('complete', summary=str(out / 'summary.json'))
