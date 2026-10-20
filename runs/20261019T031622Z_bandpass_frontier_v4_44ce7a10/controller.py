"""Registered bounded pipeline. Never changes frozen method after feedback."""
import fcntl,json,os,subprocess,time,traceback
from runtime import RUN,ROOT,C,save,sha,status,verify

START=time.monotonic()
def check_budget():
    if time.monotonic()-START>C['budget']['max_wall_seconds']:raise TimeoutError('total wall budget')
    if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())>C['budget']['max_output_gib']*1024**3:raise TimeoutError('output budget')

def parallel(mode,script='worker.py'):
    ps=[];hs=[]
    try:
        for seed in C['seeds']:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(C['budget']['seed_gpu_map'][str(seed)]),CANDIDATE_SEED=str(seed),BANDPASS_MODE=mode,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            h=(RUN/'logs'/f'{mode}_{seed}.out').open('ab',buffering=0);hs.append(h);p=subprocess.Popen([os.sys.executable,'-u',str(RUN/script)],env=env,stdout=h,stderr=subprocess.STDOUT);ps.append((seed,p))
        save('active_processes.json',{'phase':mode,'processes':[{'seed':s,'pid':p.pid,'gpu':C['budget']['seed_gpu_map'][str(s)]} for s,p in ps]})
        while True:
            check_budget()
            for s,p in ps:
                if p.poll() not in (None,0):raise RuntimeError(f'{mode} seed{s} exit{p.returncode}')
            if all(p.poll()==0 for _,p in ps):break
            time.sleep(3)
    finally:
        for _,p in ps:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
        for h in hs:h.close()

def command(script,*args):
    check_budget();subprocess.run([os.sys.executable,'-u',str(RUN/script),*args],check=True,timeout=max(1,C['budget']['max_wall_seconds']-(time.monotonic()-START)),env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))

def freeze():
    assert C['status']=='frozen';old=ROOT/C['baseline_run'];prior=ROOT/C['previous_run'];diag=ROOT/C['diagnostic_run'];assert json.loads((old/'artifacts/delivery_audit.json').read_text())['passed']
    files=list(RUN.glob('*.py'))+[RUN/'config.json',RUN/'PLAN.md',RUN/'SOURCES.md',RUN/'REPAIR_HISTORY.md',ROOT/'PROTOCOL.md',ROOT/'MULTI_FACTOR_DIAGNOSTIC_PROTOCOL.md',ROOT/'configs/datasets.json',old/'config.json',old/'artifacts/delivery_audit.json',old/'artifacts/postrun_audit.json',diag/'artifacts/data_audit.json',ROOT/C['source_run']/'artifacts/prepared.pt',ROOT/C['source_run']/'artifacts/reports.json']
    olddelivery=json.loads((old/'artifacts/delivery_audit.json').read_text());oldfreeze=json.loads((old/'artifacts/freeze.json').read_text());records=[]
    for name in C['slices']:files.append(diag/'artifacts'/f'data_{name}.pt')
    for seed in C['seeds']:
        files += [ROOT/C['source_run']/'checkpoints'/f'pure_source_wide_bn_{seed}.pth',prior/'artifacts'/f'source_gmm_{seed}.pt',old/'artifacts'/f'source_calibration_{seed}.pt']
        for name in C['slices']:
            for f in [old/'artifacts'/f'base_{seed}_{name}.npz',old/'artifacts'/f'evidence_{seed}_{name}.npz']+[old/'artifacts'/f'raw_{seed}_{name}_{m}.npz' for m in ['soft_neighbor_lora','soft_neighbor_residual','tent_bn_reference']]+[old/'checkpoints'/f'{seed}_{name}_{m}.pt' for m in ['soft_neighbor_lora','soft_neighbor_residual','tent_bn_reference']]:
                h=sha(f);rel=str(f.relative_to(old));expected=olddelivery['outputs'].get(rel)
                # Some raw/checkpoint files are covered by v3 per-cell SHA rather than delivery manifest.
                if expected is not None:assert h==expected
                records.append({'file':str(f.relative_to(ROOT)),'sha256':h,'delivery_listed':expected is not None});files.append(f)
        for f in [old/'artifacts'/f'source_calibration_{seed}.pt',prior/'artifacts'/f'source_gmm_{seed}.pt']:
            assert sha(f)==oldfreeze[str(f.relative_to(ROOT))]
    # All three raw controls must match original audited per-cell SHA.
    previous=json.loads((old/'artifacts/scores.json').read_text())
    for seed in C['seeds']:
        for name in C['slices']:
            for condition in ['soft_neighbor_lora','soft_neighbor_residual','tent_bn_reference']:
                row=next(r for r in previous if r['seed']==seed and r['slice']==name and r['condition']==condition and r['policy']=='raw')
                assert sha(old/'artifacts'/f'raw_{seed}_{name}_{condition}.npz')==row['raw_sha256'];assert sha(old/'checkpoints'/f'{seed}_{name}_{condition}.pt')==row['checkpoint_sha256']
    files.append(old/'artifacts/scores.json');save('reuse_audit.json',{'passed':True,'records':records,'controls_checked_against_original_cell_sha':90})
    save('freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(files)});verify()

def main():
    with (ROOT/'.bandpass_frontier_v4.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze();status('running','方案/代码/数据冻结；三GPU源域一致性预检，随后90次带通路由适应及安全—纠错前沿分析。')
        parallel('preflight')
        assert all(json.loads((RUN/'artifacts'/f'preflight_{s}.json').read_text())['passed'] for s in C['seeds'])
        status('running','三seed源域/无路由损失与梯度预检通过；先完成历史候选与统一接受评分诊断。');parallel('controls');command('frontier.py','controls')
        status('running','历史候选诊断完成；GPU0/1/2并行90次硬带通、软带通、软带通×几何适应。');parallel('adapted');command('frontier.py','adapted')
        trigger=json.loads((RUN/'artifacts/shuffle_trigger.json').read_text())
        if trigger['enabled']:
            status('running','触发预定软路由权重打乱归因对照，追加30次；全部属于诊断开发。');parallel('shuffle')
        status('running','适应完成，正在全query重载及独立指标/路由/分数审计。');parallel('audit','audit.py');command('frontier.py','final')
        audits=[json.loads((RUN/'artifacts'/f'audit_{s}.json').read_text()) for s in C['seeds']];assert all(a['passed'] for a in audits);expected=630+(90 if trigger['enabled'] else 0);assert sum(a['states'] for a in audits)==expected
        verify();save('postrun_audit.json',{'passed':True,'states':expected,'full_query_checkpoints':sum(a['new_checkpoints_full_query_reloaded'] for a in audits),'seeds':audits})
        save('resource_audit.json',{'pipeline_seconds':time.monotonic()-START,'conservative_gpu_hours':3*(time.monotonic()-START)/3600,'output_bytes':sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),'new_training_tasks':90+(30 if trigger['enabled'] else 0),'gpus':[0,1,2]})
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n完整query重载、冻结输入SHA、独立指标/权重/分数审计通过。资源与交付散列见artifacts/resource_audit.json、delivery_audit.json。\n')
        outputs={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and not ('lgs' in p.parts or 'logs' in p.parts or '__pycache__' in p.parts) and p.name!='delivery_audit.json'}
        save('delivery_audit.json',{'passed':True,'outputs':outputs});verify()
        for rel,h in outputs.items():assert sha(RUN/rel)==h
        decisions=json.loads((RUN/'artifacts/registered_decisions.json').read_text());primary=next(r for r in decisions if r['primary']);status('completed',f'{90+(30 if trigger["enabled"] else 0)}新适应、{expected}评分状态及全query重载审计完成；主诊断判据{primary["passed"]}，ΔRecovery@NAR1={primary["delta_rec_at_nar1_pp"]:+.3f}pp。仅诊断开发证据。')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('controller_failure.json',{'error':str(e),'traceback':traceback.format_exc()});status('failed',f'运行失败：{type(e).__name__}: {e}；产物保留，未改阈值。');raise
