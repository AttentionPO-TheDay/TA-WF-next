import os,json,time,subprocess,traceback,fcntl
import torch
from runtime import RUN,ROOT,C,save,sha,status,verify,source
START=time.monotonic()
def budget():
    if time.monotonic()-START>C['budget']['max_wall_seconds']:raise TimeoutError('pipeline wall')
    if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())>C['budget']['max_output_gib']*1024**3:raise TimeoutError('storage budget')
def freeze():
    v9=ROOT/C['v9_run'];old=json.loads((v9/'artifacts/freeze.json').read_text());delivery=json.loads((v9/'artifacts/delivery_audit.json').read_text());assert delivery['passed']
    inputs={};p=ROOT/C['source_run']/'artifacts/prepared.pt';assert sha(p)==old[str(p.relative_to(ROOT))];inputs[str(p.relative_to(ROOT))]=sha(p)
    audit=json.loads((ROOT/C['diagnostic_run']/'artifacts/data_audit.json').read_text());assert audit['passed']
    for name in C['slices']:
        p=ROOT/C['diagnostic_run']/'artifacts'/f'data_{name}.pt';assert sha(p)==audit['slices'][name]['cache_sha256'];inputs[str(p.relative_to(ROOT))]=sha(p)
    reuse=['artifacts/frontier_cells.json','artifacts/registered_decisions.json','RESULTS.md']
    for seed in C['seeds']:
        for name in C['slices']:reuse += [f'artifacts/base_{seed}_{name}.npz',f'artifacts/view_{seed}_{name}.npz',f'artifacts/valid_{seed}_{name}_frozen.npz']
    for rel in reuse:
        p=v9/rel;assert sha(p)==delivery['outputs'][rel];inputs[str(p.relative_to(ROOT))]=sha(p)
    for p in list(RUN.glob('*.py'))+[RUN/'PLAN.md',RUN/'config.json',RUN/'SOURCES.md',ROOT/'configs/datasets.json',ROOT/C['diagnostic_run']/'artifacts/data_audit.json',v9/'artifacts/delivery_audit.json']:
        inputs[str(p.relative_to(ROOT))]=sha(p)
    save('freeze.json',inputs);verify()
    ds=json.loads((ROOT/C['dataset_config']).read_text());assert (__import__('pathlib').Path(ds['data_root'])/ds['datasets']['proteus_temporal']['path']).is_dir()
    d=source();diagnostics={}
    for role,dat in d.items():
        t=dat['timestamps'];valid=(t[:,1:]!=0)&(t[:,:-1]!=0);negative=((t[:,1:].abs()<t[:,:-1].abs())&valid)
        diagnostics[role]=dict(rows=len(t),negative_adjacent_fraction=float(negative.sum()/valid.sum()),zero_padding_fraction=float((t==0).float().mean()))
    save('input_diagnostics.json',diagnostics);save('reuse_audit.json',dict(passed=True,inputs=len(inputs),new_source_models=9,old_outputs_reused=len(reuse)))
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
    with (ROOT/'.independent_view_v10.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze();status('running','预检9个独立源视图，目标评分关闭；纯source18553/valid2160固定。');parallel('preflight')
        assert all(json.loads((RUN/'artifacts'/f'preflight_{s}.json').read_text())['passed'] for s in C['seeds']);status('running','预检通过；GPU0/1/2训练9个独立源视图，30轮固定，选模仅源valid。');parallel('train')
        rows=[r for s in C['seeds'] for r in json.loads((RUN/'artifacts'/f'train_summary_{s}.json').read_text())];assert len(rows)==9 and all(not r['query_seen'] for r in rows);save('all_source_training_complete.json',dict(passed=True,models=9,seconds=time.monotonic()-START,query_scoring_started=False))
        status('running','9个source-only模型训练与valid重载完成；开始两个观察切片的终端预测。');parallel('evaluate');status('running','终端预测完成；独立重载全部9个checkpoint与36个reference/query预测状态。');parallel('audit')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');subprocess.run([os.sys.executable,'-u',str(RUN/'report.py')],env=env,check=True,timeout=300)
        audits=[json.loads((RUN/'artifacts'/f'reload_audit_{s}.json').read_text()) for s in C['seeds']];assert all(a['passed'] for a in audits) and sum(a['models_reloaded'] for a in audits)==9 and sum(a['target_states'] for a in audits)==36
        verify();budget();elapsed=time.monotonic()-START;assert 3*elapsed/3600<=C['budget']['max_gpu_hours'];save('postrun_audit.json',dict(passed=True,source_models_reloaded=9,target_prediction_states_reloaded=36,metric_audit=json.loads((RUN/'artifacts/metric_audit.json').read_text())));save('resource_audit.json',dict(pipeline_seconds=elapsed,conservative_gpu_hours=3*elapsed/3600,output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())));save('active_processes.json',dict(phase='completed',processes=[]))
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n9个checkpoint、源valid与source探针，以及36个reference/query状态完整独立重载通过；逐seed初始权重、训练行顺序、选模日程、独立混淆指标、冻结输入SHA核验通过。\n')
        outputs={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and not any(x in p.parts for x in ['lgs','logs','__pycache__']) and p.name!='delivery_audit.json'};save('delivery_audit.json',dict(passed=True,outputs=outputs))
        for n,h in outputs.items():assert sha(RUN/n)==h
        decisions=json.loads((RUN/'artifacts/registered_decisions.json').read_text());status('completed',f'9源模型/36目标状态完整重载审计通过，主候选门槛={decisions["primary_passed"]}；只做两片诊断，未接入TTA，无新GPU任务。')
if __name__=='__main__':
    try:main()
    except BaseException as e:save('controller_failure.json',dict(error=str(e),traceback=traceback.format_exc()));status('failed',f'失败{type(e).__name__}: {e}；失败日志与产物保留。');raise
