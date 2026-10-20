import hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
C=json.loads((R/'config.json').read_text())
assert C['status']=='frozen'
F=json.loads((R/'artifacts/freeze.json').read_text())
for relative,expected in F.items():
 h=hashlib.sha256((R.parents[1]/relative).read_bytes()).hexdigest()
 assert h==expected,relative
assert not (R/'artifacts/launch.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(C['budget']['physical_gpu']),CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
cmd=['/home/rbf/TA-WF-local-backup/.venv/bin/python3','-u',str(R/'train.py')]
with (R/'logs/train.out').open('ab',buffering=0) as out:
 proc=subprocess.Popen(cmd,cwd=R.parents[1],env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
(R/'artifacts/launch.json').write_text(json.dumps({'pid':proc.pid,'command':cmd,'physical_gpu':C['budget']['physical_gpu'],'seeds':C['seeds']},indent=2)+'\n')
print(proc.pid)
