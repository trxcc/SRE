"""Capture the exact shell-built CLI privately without starting a Ray job."""
from pathlib import Path
import json,os,sys,time
assert os.environ['RESEARCHCLAW_RENDER_ONLY']=='1'
args=sys.argv[1:]
runtime=Path(next(x.split('=',1)[1]for x in args if x.startswith('--runtime-env='))).read_text()
entry=args[args.index('--')+1:]
p=Path(os.environ['LOG_DIR'])/'captured-cli.json'
with p.open('x')as f:json.dump({'at':time.time(),'runtime_env':json.loads(runtime),'entry':entry,'training_launched':False},f,indent=2)
p.chmod(0o600)
print('RESEARCHCLAW_CLI_CAPTURED_WITHOUT_JOB_LAUNCH')
