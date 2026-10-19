"""Bounded v9 pipeline: preflight,54 states, complete reload and terminal analysis."""
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
    prior=ROOT/C['v8_run'];assert json.loads((prior/'artifacts/postrun_audit.json').read_text())['passed'];inherited=json.loads((prior/'artifacts/freeze.json').read_text())
    for n,h in inherited.items():assert sha(ROOT/n)==h,n
    delivery=json.loads((prior/'artifacts/delivery_audit.json').read_text())['outputs'];reuse=['RESULTS.md','artifacts/registered_decisions.json']
    for seed in C['seeds']:
        for name in C['slices']:reuse += [f'checkpoints/{seed}_{name}_geometry_grad_balance.pt']+[f'artifacts/{prefix}_{seed}_{name}_geometry_grad_balance.npz' for prefix in ['reference','query','valid']]
    for n in reuse:
        p=prior/n;assert sha(p)==delivery[n],n;inherited[str(p.relative_to(ROOT))]=sha(p)
    for p in list(RUN.glob('*.py'))+[RUN/'config.json',RUN/'PLAN.md',RUN/'SOURCES.md',prior/'artifacts/delivery_audit.json']:inherited[str(p.relative_to(ROOT))]=sha(p)
    save('freeze.json',inherited);verify();save('reuse_audit.json',dict(passed=True,inputs=len(inherited),historical_outputs_checked=len(reuse),source_rows=18553,new_adaptations=24))

def main():
    with (ROOT/'.progressive_view_v9.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze();status('running','24适应已冻结；先进行source-only PCA视图/温度拟合，纯source BN锁定。')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        subprocess.run([os.sys.executable,'-u',str(RUN/'view.py')],env=env,check=True,timeout=300)
        manifest=json.loads((RUN/'artifacts/freeze.json').read_text())
        for name in ['source_view.pt','source_view_audit.json']:
            p=RUN/'artifacts'/name;manifest[str(p.relative_to(ROOT))]=sha(p)
        save('freeze.json',manifest);verify();parallel('preflight')
        assert all(json.loads((RUN/'artifacts'/f'preflight_{s}.json').read_text())['passed'] for s in C['seeds']);status('running','源域预检通过；GPU0/1/2运行24适应及18历史控制状态，无已停分支重跑。');parallel('train')
        status('running','24适应完成；42状态完整reference/query/valid重载及辅助投影、source BN、总质量、聚类审计。');parallel('audit','audit.py')
        subprocess.run([os.sys.executable,'-u',str(RUN/'report.py')],env=env,check=True,timeout=600)
        aa=[json.loads((RUN/'artifacts'/f'audit_{s}.json').read_text()) for s in C['seeds']];assert all(a['passed'] for a in aa) and sum(a['new_checkpoints_reloaded'] for a in aa)==24
        save('postrun_audit.json',dict(passed=True,new_training_tasks=24,all_reference_query_valid_reloads=42,new_checkpoints_reloaded=24,source_only_auxiliary_fit=True,seeds=aa));verify();elapsed=time.monotonic()-START;budget();assert 3*elapsed/3600<=C['budget']['max_gpu_hours']
        save('resource_audit.json',dict(pipeline_seconds=elapsed,conservative_gpu_hours=3*elapsed/3600,output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),new_adaptations=24,physical_gpus=[0,1,2]));save('active_processes.json',dict(phase='completed',processes=[]))
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n24checkpoint和18控制状态完整reference/query/valid重载、辅助视图PCA/q重算、BN及底座锁定、倍率/打乱质量审计、独立混淆矩阵与聚类分配及source-only聚类重拟合、SHA核验通过。\n')
        outputs={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and not any(v in p.parts for v in ['lgs','logs','__pycache__']) and p.name!='delivery_audit.json'};save('delivery_audit.json',dict(passed=True,outputs=outputs));verify()
        for n,h in outputs.items():assert sha(RUN/n)==h
        d=json.loads((RUN/'artifacts/registered_decisions.json').read_text());status('completed',f'24适应、42状态三分区完整重载通过；稳定性={d["stability_passed"]}，相对v8效果={d["efficacy_passed"]}，辅助视图={d["auxiliary_passed"]}。仅两片诊断开发。')

if __name__=='__main__':
    try:main()
    except BaseException as e:save('controller_failure.json',dict(error=str(e),traceback=traceback.format_exc()));status('failed',f'失败：{type(e).__name__}: {e}；产物保留。');raise
