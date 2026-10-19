import os,time,json,fcntl,subprocess,traceback
from pathlib import Path
from runtime import RUN,ROOT,C,sha,save,verify,status
START=time.monotonic()
def budget():
    if time.monotonic()-START>C['budget']['max_wall_seconds']:raise TimeoutError('pipeline wall')
    if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())>C['budget']['max_output_gib']*1024**3:raise TimeoutError('output budget')
def freeze():
    v10=ROOT/C['v10_run'];v9=ROOT/C['v9_run'];d10=json.loads((v10/'artifacts/delivery_audit.json').read_text());d9=json.loads((v9/'artifacts/delivery_audit.json').read_text());assert d10['passed'] and d9['passed'];old=json.loads((v10/'artifacts/freeze.json').read_text());manifest={}
    for p in [ROOT/C['source_run']/'artifacts/prepared.pt']+[ROOT/C['diagnostic_run']/'artifacts'/f'data_{n}.pt' for n in C['slices']]:
        rel=str(p.relative_to(ROOT));assert sha(p)==old[rel];manifest[rel]=sha(p)
    reuse10=['artifacts/frontier_cells.json','artifacts/registered_decisions.json','RESULTS.md','model.py']
    reuse9=['artifacts/frontier_cells.json','RESULTS.md']
    for seed in C['seeds']:
        reuse10.append(f'artifacts/train_summary_{seed}.json')
        for arm in C['arms']:
            reuse10 += [f'checkpoints/{seed}_{arm}.pt',f'artifacts/sourcevalid_{seed}_{arm}.npz']+[f'artifacts/{role}_{seed}_{name}_{arm}.npz' for name in C['slices'] for role in ['reference','query']]
        for name in C['slices']:
            reuse9 += [f'artifacts/base_{seed}_{name}.npz']+[f'artifacts/{partition}_{seed}_{name}_{condition}.npz' for partition in ['query','valid'] for condition in ['frozen','smooth_balance']]
    for parent,delivery,reuse in [(v10,d10,reuse10),(v9,d9,reuse9)]:
        for rel in reuse:
            p=parent/rel;assert sha(p)==delivery['outputs'][rel],p;manifest[str(p.relative_to(ROOT))]=sha(p)
    assert sha(RUN/'model.py')==sha(v10/'model.py')
    for p in list(RUN.glob('*.py'))+[RUN/'PLAN.md',RUN/'config.json',RUN/'SOURCES.md',ROOT/'configs/datasets.json',v10/'artifacts/delivery_audit.json',v9/'artifacts/delivery_audit.json']:
        manifest[str(p.relative_to(ROOT))]=sha(p)
    save('freeze.json',manifest);verify();ds=json.loads((ROOT/C['dataset_config']).read_text());assert (Path(ds['data_root'])/ds['datasets']['proteus_temporal']['path']).is_dir();save('reuse_audit.json',dict(passed=True,explicit_checkpoints=9,reused_source_training=True,new_training_steps=0,inputs=len(manifest),v10_output_checks=len(reuse10),v9_output_checks=len(reuse9)))
def parallel(mode):
    ps=[];hs=[]
    try:
        for seed in C['seeds']:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(C['budget']['seed_gpu_map'][str(seed)]),VIEW_SEED=str(seed),VIEW_MODE=mode,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            h=(RUN/'logs'/f'{mode}_{seed}.out').open('ab',buffering=0);hs.append(h);p=subprocess.Popen([os.sys.executable,'-u',str(RUN/'worker.py')],env=env,stdout=h,stderr=subprocess.STDOUT);ps.append((seed,p))
        save('active_processes.json',dict(phase=mode,processes=[dict(seed=s,pid=p.pid,gpu=C['budget']['seed_gpu_map'][str(s)]) for s,p in ps]))
        while True:
            budget()
            for seed,p in ps:
                if p.poll() not in (None,0):raise RuntimeError(f'{mode} seed{seed} exit{p.returncode}')
            if all(p.poll()==0 for _,p in ps):break
            time.sleep(3)
    finally:
        for _,p in ps:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
        for h in hs:h.close()
def main():
    with (ROOT/'.cross_view_stability_v11.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze();status('running','已冻结两候选池/五评分；零训练，先核验扰动及原始logits。');parallel('preflight');assert all(json.loads((RUN/'artifacts'/f'preflight_{s}.json').read_text())['passed'] for s in C['seeds'])
        status('running','预检通过；GPU0/1/2生成180个新扰动logit状态，不更新权重或统计。');parallel('infer');status('running','180个新扰动状态完成；全量重载核验，另核对45个原始状态。');parallel('audit')
        audits=[json.loads((RUN/'artifacts'/f'audit_{s}.json').read_text()) for s in C['seeds']];assert all(a['passed'] for a in audits) and sum(a['augmented_states'] for a in audits)==180 and sum(a['original_identity_states'] for a in audits)==45
        status('running','180+45状态重载通过；计算固定候选的稳定性评分、打乱对照与诊断前沿。');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');subprocess.run([os.sys.executable,'-u',str(RUN/'analysis.py')],env=env,check=True,timeout=300)
        verify();budget();elapsed=time.monotonic()-START;assert 3*elapsed/3600<=C['budget']['max_gpu_hours'];save('postrun_audit.json',dict(passed=True,reloaded_augmented_states=180,original_identity_states=45,unchanged_model_checkpoints=9,metric_audit=json.loads((RUN/'artifacts/metric_audit.json').read_text()),source_query_fit=False));save('resource_audit.json',dict(pipeline_seconds=elapsed,conservative_gpu_hours=3*elapsed/3600,output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),new_training_steps=0));save('active_processes.json',dict(phase='completed',processes=[]))
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n180个新扰动状态全量重载、45个原始logits精确一致、9模型锁定、360个终端评分状态独立混淆与tie-safe前沿核验通过；输入SHA与历史基线前沿精确一致，结果已封存。\n')
        out={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and not any(x in p.parts for x in ['logs','lgs','__pycache__']) and p.name!='delivery_audit.json'};save('delivery_audit.json',dict(passed=True,outputs=out))
        for p,h in out.items():assert sha(RUN/p)==h
        d=json.loads((RUN/'artifacts/registered_decisions.json').read_text());status('completed',f'180扰动/45原始状态全重载通过；主判据={d["primary_passed"]}，实际可利用性={d["utility_passed"]}。零训练、未整合TTA，GPU空闲。')
if __name__=='__main__':
    try:main()
    except BaseException as e:save('controller_failure.json',dict(error=str(e),traceback=traceback.format_exc()));status('failed',f'失败{type(e).__name__}: {e}；保留失败日志与产物。');raise
