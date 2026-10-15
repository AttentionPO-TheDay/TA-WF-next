import json,os,subprocess,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent; C=json.loads((R/'config.json').read_text())
with (R/'artifacts/launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX); assert C['status']=='frozen' and not (R/'artifacts/launch.json').exists()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(C['physical_gpu']),CUBLAS_WORKSPACE_CONFIG=':4096:8')
 with (R/'logs/runner.out').open('ab',buffering=0) as out: p=subprocess.Popen([C['python'],'-u',str(R/'runner.py')],cwd=R.parents[1],env=env,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 d={'pid':p.pid,'physical_gpu':C['physical_gpu'],'conditions':C['conditions']}; (R/'artifacts/launch.json').write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d))
