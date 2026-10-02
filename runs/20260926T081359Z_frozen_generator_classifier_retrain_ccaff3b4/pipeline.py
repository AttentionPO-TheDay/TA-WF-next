"""Six CPU slots shared by training and immediate per-job verification."""
from __future__ import annotations
from collections import deque
import fcntl
import subprocess
import time
from common import *
atomic_json=write_json


def rss_bytes(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    except FileNotFoundError:return 0
    return 0


def stop_process(p):
    if p.poll() is not None:return
    p.terminate()
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait()


def main():
    with (RUN/'artifacts/pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=config_read();resource=config['resource']
        queue=deque(('train',c,seed) for seed in config['seeds'] for c in config['conditions'])
        active={};exit_codes={};processes={};reasons={};cancelled=[];started=time.monotonic()
        slots=resource['workers']
        assert slots*resource['threads_per_worker']<=12
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS=str(resource['threads_per_worker']),
                 MKL_NUM_THREADS=str(resource['threads_per_worker']),OPENBLAS_NUM_THREADS='1')
        while queue or active:
            elapsed=time.monotonic()-started
            if elapsed>resource['pipeline_seconds']:
                cancelled=list(queue);queue.clear()
                for task,entry in active.items():reasons[task]='global time budget';stop_process(entry['process'])
            while queue and len(active)<slots:
                kind,name,seed=queue.popleft();key=f'{kind}_{name}_{seed}'
                log=(RUN/'logs'/f'{key}.log').open('a')
                script='train.py' if kind=='train' else 'verify_one.py'
                p=subprocess.Popen([sys.executable,'-u',str(RUN/script),'--condition',name,'--seed',str(seed)],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                active[key]={'process':p,'log':log,'started':time.monotonic(),'kind':kind,'name':name,'seed':seed}
                processes[key]=p.pid
                atomic_json(RUN/'artifacts/processes.json',{'supervisor_pid':os.getpid(),'workers':processes,'device':'cpu'})
            usage={key:rss_bytes(entry['process'].pid) for key,entry in active.items()}
            total=sum(usage.values())
            for key,entry in list(active.items()):
                p=entry['process'];kind=entry['kind'];lived=time.monotonic()-entry['started']
                limit=resource['job_seconds'] if kind=='train' else resource['verify_seconds']
                if p.poll() is None:
                    if lived>limit:reasons[key]='job wall limit';stop_process(p)
                    elif usage[key]>resource['rss_gib_per_worker']*1024**3:reasons[key]='per-process RSS limit';stop_process(p)
                    elif total>resource['total_rss_gib']*1024**3:
                        reasons[key]='total RSS limit';stop_process(p);total-=usage[key]
                code=p.poll()
                if code is None:continue
                exit_codes[key]=code;entry['log'].close();del active[key]
                if kind=='train' and code==0 and elapsed<=resource['pipeline_seconds']:
                    queue.appendleft(('verify',entry['name'],entry['seed']))
                atomic_json(RUN/'artifacts/worker_exit_codes.json',{'codes':exit_codes,'stop_reasons':reasons})
            atomic_json(RUN/'artifacts/queue_progress.json',{'elapsed_seconds':time.monotonic()-started,
                        'active':{k:v['process'].pid for k,v in active.items()},'queued':list(queue),
                        'finished':len(exit_codes),'active_rss_bytes':usage,'cancelled':cancelled})
            if active:time.sleep(2)
        with (RUN/'logs/finalize.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',str(RUN/'finalize.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(RUN/'artifacts/pipeline_result.json',{'exit_codes':exit_codes,'stop_reasons':reasons,'cancelled':cancelled,
                     'finalize_exit_code':result.returncode,'finished_unix':time.time(),'elapsed_seconds':time.monotonic()-started})
        if result.returncode and not (RUN/'artifacts/integrity.json').exists():
            summary='CPU并行实验结束但汇总器异常，需检查logs/finalize.log；不能称核验完成。'
            subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state','stopped','--summary',summary],check=True)
            (RUN/'RESULTS.md').write_text('# 实验结果\n\n'+summary+'\n')
        if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
