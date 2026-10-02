"""Three-slot bounded CPU diagnostic supervisor."""
from collections import deque
import fcntl
import subprocess
import time
from diagnose import *


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:p.kill();p.wait()

def main():
    with (RUN/'artifacts/pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=config_read();started=time.monotonic();pending=deque((name,seed) for seed in c['seeds'] for name in c['conditions'])
        active={};codes={};reasons={};cancelled=[]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
        while pending or active:
            elapsed=c.get('previous_execution_seconds',0)+time.monotonic()-started
            if elapsed>c['pipeline_time_limit_seconds']:
                cancelled=list(pending);pending.clear()
                for key,entry in active.items():stop(entry['p']);reasons[key]='global budget'
            while pending and len(active)<c['concurrent_workers']:
                name,seed=pending.popleft();key=f'{name}_{seed}';log=(RUN/'logs'/f'{key}.log').open('w')
                p=subprocess.Popen([sys.executable,'-u',str(RUN/'diagnose.py'),'--condition',name,'--seed',str(seed)],
                                   stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
                active[key]={'p':p,'log':log,'started':time.monotonic()}
            for key,entry in list(active.items()):
                p=entry['p']
                if p.poll() is None:
                    if time.monotonic()-entry['started']>c['time_limit_seconds_per_job']+30:stop(p);reasons[key]='job timeout'
                    else:
                        try:
                            rss=next(int(line.split()[1])*1024 for line in Path(f'/proc/{p.pid}/status').read_text().splitlines() if line.startswith('VmRSS:'))
                            if rss>c['max_rss_gib_per_worker']*1024**3:stop(p);reasons[key]='memory budget'
                        except (FileNotFoundError,StopIteration):pass
                if p.poll() is not None:
                    codes[key]=p.returncode;entry['log'].close();del active[key]
            write_json(RUN/'artifacts/progress.json',{'elapsed_seconds':elapsed,'active':{k:v['p'].pid for k,v in active.items()},'pending':list(pending),'exit_codes':codes,'stop_reasons':reasons})
            if active:time.sleep(2)
        with (RUN/'logs/summarize.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',str(RUN/'summarize.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
        write_json(RUN/'artifacts/pipeline_result.json',{'exit_codes':codes,'stop_reasons':reasons,'cancelled':cancelled,'summary_exit_code':result.returncode,'elapsed_seconds':time.monotonic()-started})
        if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
