"""Thirty-six bounded diagnostic adaptations, source preflight and full reload."""
import fcntl,json,os,time,subprocess,traceback
from runtime import RUN,ROOT,C,save,sha,verify,status
START=time.monotonic()
N=len(C['conditions'])*len(C['seeds'])*len(C['slices'])

def budget():
    if time.monotonic()-START>C['budget']['max_wall_seconds']:raise TimeoutError('pipeline wall')
    if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())>C['budget']['max_output_gib']*1024**3:raise TimeoutError('output budget')

def parallel(mode,script='worker.py'):
    ps=[];hs=[]
    try:
        for seed in C['seeds']:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(C['budget']['seed_gpu_map'][str(seed)]),CANDIDATE_SEED=str(seed),MOMENTUM_MODE=mode,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            h=(RUN/'logs'/f'{mode}_{seed}.out').open('ab',buffering=0);hs.append(h)
            p=subprocess.Popen([os.sys.executable,'-u',str(RUN/script)],env=env,stdout=h,stderr=subprocess.STDOUT);ps.append((seed,p))
        save('active_processes.json',{'phase':mode,'processes':[{'seed':s,'pid':p.pid,'gpu':C['budget']['seed_gpu_map'][str(s)]} for s,p in ps]})
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
    assert C['status']=='frozen'
    prior=ROOT/C['v5_run']
    assert json.loads((prior/'artifacts/postrun_audit.json').read_text())['passed']
    inherited=json.loads((prior/'artifacts/freeze.json').read_text())
    for n,h in inherited.items():assert sha(ROOT/n)==h,n
    delivery=json.loads((prior/'artifacts/delivery_audit.json').read_text())['outputs']
    reuse=['artifacts/scores.json','artifacts/summary.json','artifacts/postrun_audit.json']
    for seed in C['seeds']:
        for name in C['slices']:
            for arm in ['full_static','batch_momentum','momentum_proto']:
                reuse += [f'artifacts/raw_{seed}_{name}_{arm}.npz',f'artifacts/signal_{seed}_{name}_{arm}.npz']
            reuse += [f'checkpoints/{seed}_{name}_momentum_proto.pt',f'artifacts/updates_{seed}_{name}_momentum_proto.json']
    for n in reuse:
        p=prior/n;assert sha(p)==delivery[n],n;inherited[str(p.relative_to(ROOT))]=sha(p)
    files=list(RUN.glob('*.py'))+[RUN/'config.json',RUN/'PLAN.md',RUN/'SOURCES.md',RUN/'REPAIR_HISTORY.md',prior/'artifacts/delivery_audit.json']
    for p in files:inherited[str(p.relative_to(ROOT))]=sha(p)
    save('freeze.json',inherited);verify()
    save('reuse_audit.json',{'passed':True,'source_rows':18553,'source_valid_rows':2160,'inherited_inputs':len(inherited),'historical_control_outputs_checked':len(reuse),'source_seeds':C['seeds'],'slices':C['slices']})

def main():
    with (ROOT/'.prototype_refinement_v6.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze()
        status('running','六组原型细化、两切片、三源seed共36任务已冻结；GPU0/1/2源域预检。');parallel('preflight')
        assert all(json.loads((RUN/'artifacts'/f'preflight_{s}.json').read_text())['passed'] for s in C['seeds'])
        status('running','源域等价性和六组开关预检通过；GPU0/1/2并行36任务，仅reference更新。');parallel('train')
        status('running','36任务完成，进行全query/source-valid checkpoint重载及独立指标审计。');parallel('audit','audit.py')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        subprocess.run([os.sys.executable,'-u',str(RUN/'frontier.py')],env=env,check=True,timeout=600)
        aa=[json.loads((RUN/'artifacts'/f'audit_{s}.json').read_text()) for s in C['seeds']]
        assert all(a['passed'] for a in aa) and sum(a['query_checkpoints_reloaded'] for a in aa)==N
        save('postrun_audit.json',{'passed':True,'new_training_tasks':N,'all_query_and_valid_checkpoint_reload':N,'seeds':aa});verify()
        elapsed=time.monotonic()-START;budget();assert 3*elapsed/3600<=C['budget']['max_gpu_hours']
        save('resource_audit.json',{'pipeline_seconds':elapsed,'conservative_gpu_hours':3*elapsed/3600,'output_bytes':sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),'new_adaptations':N,'physical_gpus':[0,1,2]})
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n36个checkpoint的完整query和source-valid重载、逐指标独立复算、纯净底座锁定、reference状态范围及输入/输出SHA核验通过。\n')
        outputs={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and not any(v in p.parts for v in ['lgs','logs','__pycache__']) and p.name!='delivery_audit.json'}
        save('delivery_audit.json',{'passed':True,'outputs':outputs});verify()
        for n,h in outputs.items():assert sha(RUN/n)==h
        d=json.loads((RUN/'artifacts/registered_decisions.json').read_text())
        status('completed',f'36任务与全query/valid重载审计完成；主候选配对改进={d["primary_passed"]}，扩展资格={d["expansion_eligible"]}。仅两切片诊断开发。')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('controller_failure.json',{'error':str(e),'traceback':traceback.format_exc()});status('failed',f'失败：{type(e).__name__}: {e}；产物保留。');raise
