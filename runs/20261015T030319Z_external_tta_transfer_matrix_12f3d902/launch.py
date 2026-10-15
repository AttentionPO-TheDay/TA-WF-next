import fcntl,json,os,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent; C=json.loads((R/'config.json').read_text())
with (R/'artifacts/launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX); assert C['status']=='frozen'; p=json.loads((R/'artifacts/preflight.json').read_text()); assert p['passed']; assert p['config_sha']==hashlib.sha256((R/'config.json').read_bytes()).hexdigest()
 for n,d in p['code_sha'].items(): assert hashlib.sha256((R/n).read_bytes()).hexdigest()==d,n
 assert not (R/'artifacts/launch.json').exists()
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(C['physical_gpu']),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=os.pathsep.join(['/home/rbf/.cache/uv/archive-v0/T5i6lSLqi6GiQIJQ:/home/rbf/.cache/uv/archive-v0/oTKuGrE9yEg84v3r',os.environ.get('PYTHONPATH','')]))
 cmd=[C['python'],'-u',str(R/'runner.py')]
 with (R/'logs/runner.out').open('ab',buffering=0) as out: proc=subprocess.Popen(cmd,cwd=R.parents[1],env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 data={'pid':proc.pid,'gpu':C['physical_gpu'],'new_tasks':255,'command':cmd}; (R/'artifacts/launch.json').write_text(json.dumps(data,indent=2)+'\n'); print(json.dumps(data))
