import os,time,json,subprocess,traceback,fcntl
from pathlib import Path
from runtime import RUN,ROOT,C,save,sha,verify,status
START=time.monotonic()
def budget():
    if time.monotonic()-START>C['budget']['max_wall_seconds']:raise TimeoutError('pipeline wall')
    if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())>C['budget']['max_output_gib']*1024**3:raise TimeoutError('output budget')
def freeze():
    manifest={};parents={key:ROOT/C[key] for key in ['v9_run','v10_run','v11_run']};deliveries={k:json.loads((p/'artifacts/delivery_audit.json').read_text()) for k,p in parents.items()};assert all(d['passed'] for d in deliveries.values());reuse={k:[] for k in parents}
    reuse['v11_run']=['artifacts/cells.json','artifacts/registered_decisions.json','RESULTS.md']
    for seed in C['seeds']:
        reuse['v10_run'].append(f'artifacts/train_summary_{seed}.json')
        for role in ['valid']+[n+'_query' for n in C['slices']]:
            reuse['v10_run'].append(f'artifacts/sourcevalid_{seed}_direction_iat.npz' if role=='valid' else f'artifacts/query_{seed}_{role.removesuffix("_query")}_direction_iat.npz')
            for view in C['views'][1:]:reuse['v11_run'].append(f'artifacts/logits_{seed}_direction_iat_{role}_{view}.npz')
        for name in C['slices']:
            for partition in ['valid','query']:
                reuse['v11_run'].append(f'artifacts/scores_{partition}_{seed}_{name}_direction_iat_v9_smooth.npz')
                reuse['v9_run'] += [f'artifacts/{partition}_{seed}_{name}_{condition}.npz' for condition in ['frozen','smooth_balance']]
    for key,rels in reuse.items():
        for rel in rels:
            p=parents[key]/rel;h=sha(p);assert h==deliveries[key]['outputs'][rel],p;manifest[str(p.relative_to(ROOT))]=h
    for p in list(RUN.glob('*.py'))+[RUN/'config.json',RUN/'PLAN.md',RUN/'SOURCES.md',ROOT/'configs/datasets.json']+[p/'artifacts/delivery_audit.json' for p in parents.values()]:manifest[str(p.relative_to(ROOT))]=sha(p)
    save('freeze.json',manifest);verify();d=json.loads((ROOT/C['dataset_config']).read_text());assert (Path(d['data_root'])/d['datasets']['proteus_temporal']['path']).is_dir();save('reuse_audit.json',dict(passed=True,files=len(manifest),explicit_reuse_counts={k:len(v) for k,v in reuse.items()},new_training_steps=0,new_forward_passes=0,query_labels_not_used_for_fitting=True))
def call(phase):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    save('active_processes.json',dict(phase=phase,compute='CPU2 cached logits only'))
    with (RUN/'logs'/f'{phase}.out').open('ab',buffering=0) as f:subprocess.run([os.sys.executable,'-u',str(RUN/'experiment.py'),phase],stdout=f,stderr=subprocess.STDOUT,env=env,check=True,timeout=300)
    budget()
def main():
    with (ROOT/'.low_nar_soft_v12.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze();status('running','配置已冻结；以FPR=NAR<=1%考核，源valid拟合固定软证据权重与尺度，零训练/前向。');call('run')
        status('running','96个缓存评分状态完成；独立重载重算pAUC、实际阈值和无效改写计数。');call('audit');assert json.loads((RUN/'artifacts/reload_audit.json').read_text())['passed'];call('report');verify();budget()
        elapsed=time.monotonic()-START;save('postrun_audit.json',dict(passed=True,states=96,candidate_pool_unchanged=True,pauc_full_negative_denominator=True,sklearn_partial_roc_checked=True,source_calibration_frozen_before_query=True));save('resource_audit.json',dict(pipeline_seconds=elapsed,output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),gpu_hours=0,new_training_steps=0,new_forward_passes=0));save('active_processes.json',dict(phase='completed',compute='CPU2 cached logits only'))
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n全部96状态评分逐位重载一致、局部ROC独立积分/混淆矩阵/实际阈值及无效改写重算通过；24历史基线前沿精确一致，候选类别与source校准不变，输入输出SHA封存。\n')
        outputs={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and not any(x in p.parts for x in ['logs','lgs','__pycache__']) and p.name!='delivery_audit.json'};save('delivery_audit.json',dict(passed=True,outputs=outputs));verify()
        for p,h in outputs.items():assert sha(RUN/p)==h
        d=json.loads((RUN/'artifacts/registered_decisions.json').read_text());status('completed',f'96状态重载/局部ROC审计通过，主判据={d["passed"]}；零训练/新前向，未部署或整合TTA。')
if __name__=='__main__':
    try:main()
    except BaseException as e:save('controller_failure.json',dict(error=str(e),traceback=traceback.format_exc()));status('failed',f'失败{type(e).__name__}: {e}；保留全部日志和产物。');raise
