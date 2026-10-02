"""Start six bounded CPU workers, then replay predictions and close the registry."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
RUN=Path(__file__).resolve().parent
import train


def main():
    with (RUN/'artifacts/pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config,_=train.load_config()
        assert config['concurrent_workers']==6 and config['threads_per_worker']==2
        environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        workers=[]
        for seed in config['seeds']:
            for condition in config['conditions']:
                key=f'{condition}_{seed}'
                log=(RUN/'logs'/f'{key}.log').open('a')
                worker=subprocess.Popen([sys.executable,'-u',str(RUN/'train.py'),'--condition',condition,'--seed',str(seed)],
                                        cwd=RUN.parents[1],env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                workers.append((key,worker,log,time.monotonic()))
        train.atomic_json(RUN/'artifacts/processes.json',{'supervisor_pid':os.getpid(),'workers':{key:p.pid for key,p,_,_ in workers},'device':'cpu'})
        codes={}
        while len(codes)<len(workers):
            for key,worker,log,started in workers:
                if key in codes:continue
                code=worker.poll()
                # Allow setup/serialization plus an epoch-boundary stop; never add optimizer budget.
                if code is None and time.monotonic()-started>config['time_limit_seconds_per_seed']+180:
                    worker.terminate()
                    try:code=worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:worker.kill();code=worker.wait()
                if code is not None:
                    codes[key]=code;log.close()
                    train.atomic_json(RUN/'artifacts/worker_exit_codes.json',codes)
            time.sleep(2)
        with (RUN/'logs/verify.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',str(RUN/'verify_report.py')],env=environment,stdout=log,stderr=subprocess.STDOUT)
        train.atomic_json(RUN/'artifacts/pipeline_result.json',{'worker_exit_codes':codes,'verification_exit_code':result.returncode,'finished_unix':time.time()})
        if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
