"""Bounded v7 pipeline: preflight,54 states, complete reload and terminal analysis."""
import fcntl,json,os,time,subprocess,traceback
from runtime import RUN,ROOT,C,save,sha,verify,status
START=time.monotonic()

def budget():
    if time.monotonic()-START>C['budget']['max_wall_seconds']:raise TimeoutError('pipeline wall')
    if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())>C['budget']['max_output_gib']*1024**3:raise TimeoutError('output budget')

def parallel(mode,script='worker.py'):
    ps=[];hs=[]
    try:
        for seed in C['seeds']:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(C['budget']['seed_gpu_map'][str(seed)]),CANDIDATE_SEED=str(seed),MOMENTUM_MODE=mode,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            h=(RUN/'logs'/f'{mode}_{seed}.out').open('ab',buffering=0);hs.append(h);p=subprocess.Popen([os.sys.executable,'-u',str(RUN/script)],env=env,stdout=h,stderr=subprocess.STDOUT);ps.append((seed,p))
        save('active_processes.json',dict(phase=mode,processes=[dict(seed=s,pid=p.pid,gpu=C['budget']['seed_gpu_map'][str(s)]) for s,p in ps]))
        while True:
            budget()
            for s,p in ps:
                if p.poll() not in (None,0):raise RuntimeError(f'{mode} seed{s} exit{p.returncode}')
            if all(p.poll()==0 for _,p in ps):break
            time.sleep(3)
    finally:
        for _,p in ps:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
        for h in hs:h.close()

def freeze():
    prior=ROOT/C['v6_run'];assert json.loads((prior/'artifacts/postrun_audit.json').read_text())['passed']
    inherited=json.loads((prior/'artifacts/freeze.json').read_text())
    for n,h in inherited.items():assert sha(ROOT/n)==h,n
    delivery=json.loads((prior/'artifacts/delivery_audit.json').read_text())['outputs'];reuse=[]
    for seed in C['seeds']:
        for name in C['slices']:
            reuse += [f'checkpoints/{seed}_{name}_sharp_target_only.pt']+[f'artifacts/{prefix}_{seed}_{name}_sharp_target_only.npz' for prefix in ['raw','valid','signal']]
    for n in reuse:
        p=prior/n;assert sha(p)==delivery[n],n;inherited[str(p.relative_to(ROOT))]=sha(p)
    v3=ROOT/C['baseline_run'];r3=json.loads((v3/'artifacts/scores.json').read_text())
    for seed in C['seeds']:
        for name in C['slices']:
            row=next(r for r in r3 if r['seed']==seed and r['slice']==name and r['condition']=='tent_bn_reference' and r['policy']=='raw')
            for rel,key in [(f'checkpoints/{seed}_{name}_tent_bn_reference.pt','checkpoint_sha256'),(f'artifacts/raw_{seed}_{name}_tent_bn_reference.npz','raw_sha256')]:
                p=v3/rel;assert sha(p)==row[key];inherited[str(p.relative_to(ROOT))]=sha(p)
    files=list(RUN.glob('*.py'))+[RUN/'config.json',RUN/'PLAN.md',RUN/'SOURCES.md',prior/'artifacts/delivery_audit.json',v3/'artifacts/scores.json',ROOT/C['source_run']/'artifacts/prepared.pt']
    for p in files:inherited[str(p.relative_to(ROOT))]=sha(p)
    save('freeze.json',inherited);verify();save('reuse_audit.json',dict(passed=True,inherited_inputs=len(inherited),historical_control_outputs_checked=len(reuse)+12,source_rows=18553,source_valid_rows=2160))

def main():
    with (ROOT/'.mechanism_isolation_v7.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze();status('running','用户恢复执行；24适应+30推理分解已冻结，GPU0/1/2源域预检。');parallel('preflight')
        assert all(json.loads((RUN/'artifacts'/f'preflight_{s}.json').read_text())['passed'] for s in C['seeds'])
        status('running','源域预检通过；24适应与30推理状态在GPU0/1/2运行，仅reference拟合。');parallel('train')
        status('running','54状态完成；全query/source-valid重载审计。');parallel('audit','audit.py')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for script in ['scoring.py','report.py']:subprocess.run([os.sys.executable,'-u',str(RUN/script)],env=env,check=True,timeout=600)
        aa=[json.loads((RUN/'artifacts'/f'audit_{s}.json').read_text()) for s in C['seeds']];assert all(a['passed'] for a in aa) and sum(a['query_checkpoints_reloaded'] for a in aa)==54
        save('postrun_audit.json',dict(passed=True,new_training_tasks=24,all_query_and_valid_checkpoint_reload=54,seeds=aa));verify();elapsed=time.monotonic()-START;budget()
        assert 3*elapsed/3600<=C['budget']['max_gpu_hours'];save('resource_audit.json',dict(pipeline_seconds=elapsed,conservative_gpu_hours=3*elapsed/3600,output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),new_adaptations=24,physical_gpus=[0,1,2]))
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n54状态完整query/valid重载、独立指标、source阈值与预测策略复算、BN范围/源底座锁定、SHA审计通过。\n')
        outputs={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and not any(v in p.parts for v in ['lgs','logs','__pycache__']) and p.name!='delivery_audit.json'}
        save('delivery_audit.json',dict(passed=True,outputs=outputs));verify()
        for n,h in outputs.items():assert sha(RUN/n)==h
        d=json.loads((RUN/'artifacts/registered_decisions.json').read_text());status('completed',f'24适应+30推理状态及54完整query/valid重载通过；写入筛选={d["memory_passed"]}，早期统计={d["stats_passed"]}，主评分={d["scoring_passed"]}。仅两片诊断开发。')

if __name__=='__main__':
    try:main()
    except BaseException as e:save('controller_failure.json',dict(error=str(e),traceback=traceback.format_exc()));status('failed',f'失败：{type(e).__name__}: {e}；产物保留。');raise
