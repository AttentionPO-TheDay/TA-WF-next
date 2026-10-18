import json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent; C=json.loads((R/"config.json").read_text())
assert C["status"]=="frozen" and json.loads((R/"artifacts/preflight.json").read_text())["passed"]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(C["physical_gpu"]),CUBLAS_WORKSPACE_CONFIG=":4096:8")
with (R/"lgs/runner.out").open("ab",buffering=0) as out:
 p=subprocess.Popen([C["python"],"-u",str(R/"runner.py")],cwd=R.parents[1],env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
(R/"artifacts/launch.json").write_text(json.dumps({"pid":p.pid,"gpu":C["physical_gpu"],"new_tasks":9},indent=2)+"\n")
print(p.pid)
