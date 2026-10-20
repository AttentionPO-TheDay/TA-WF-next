"""Source preflight, diagnostic manifest, input freeze, two GPU queues, audit, report."""
import fcntl
import json
import os
import subprocess
import time
import traceback
from runtime import RUN,ROOT,C,sha,save,status,verify

def call(name,timeout):subprocess.run([os.sys.executable,'-u',str(RUN/name)],check=True,timeout=timeout)

def main():
    with (ROOT/'.multifactor_diagnostic_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify();status('running','协议/机制已冻结；source-only预检及逐行诊断清单构建中，尚未目标评分。')
        call('preflight.py',600)
        if not (RUN/'artifacts/data_audit.json').exists():call('prepare.py',2400)
        audit=json.loads((RUN/'artifacts/data_audit.json').read_text());assert audit['passed']
        freeze=json.loads((RUN/'artifacts/freeze.json').read_text())
        for p in [RUN/'artifacts/diagnostic_rows.csv',RUN/'artifacts/data_audit.json',RUN/'artifacts/diagnostic_gate.json']+list((RUN/'artifacts').glob('data_*.pt')):
            freeze[str(p.relative_to(ROOT))]=sha(p)
        save('freeze.json',freeze);verify()
        status('running','诊断协议已解锁并冻结逐行清单；GPU0/2运行三seed×十切片×七条件210单元，外部v4仅引用。')
        pending=list(C['seeds']);active={};handles=[];start=time.monotonic()
        try:
            while pending or active:
                for seed in list(pending):
                    gpu=C['budget']['seed_gpu_map'][str(seed)]
                    if gpu in active:continue
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),DIAGNOSTIC_SEED=str(seed))
                    h=(RUN/'logs'/f'worker_{seed}.out').open('ab',buffering=0);handles.append(h)
                    p=subprocess.Popen([os.sys.executable,'-u',str(RUN/'worker.py')],env=env,stdout=h,stderr=subprocess.STDOUT)
                    active[gpu]=(seed,p);pending.remove(seed)
                for gpu,(seed,p) in list(active.items()):
                    if p.poll() is not None:
                        if p.returncode:raise RuntimeError(f'worker {seed} failed: {p.returncode}')
                        del active[gpu]
                if time.monotonic()-start>C['budget']['max_wall_seconds']:raise TimeoutError('matrix wall budget')
                if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())>C['budget']['max_output_gib']*1024**3:raise TimeoutError('output budget')
                time.sleep(3)
        finally:
            for _,p in active.values():
                if p.poll() is None:p.terminate();p.wait(timeout=30)
            for h in handles:h.close()
        scores=[]
        for seed in C['seeds']:scores.extend(json.loads((RUN/'artifacts'/f'scores_{seed}.json').read_text()))
        assert len(scores)==210;save('scores.json',scores)
        call('audit.py',2400);call('report.py',600)
        status('completed','210单元全部完成且checkpoint重载/指标复算通过；Table1/2/3、雷达图及两组共享坐标t-SNE已导出，正负结果如实保留。')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('controller_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        status('failed',f'诊断流水线失败：{type(e).__name__}: {e}；保留全部产物。');raise
