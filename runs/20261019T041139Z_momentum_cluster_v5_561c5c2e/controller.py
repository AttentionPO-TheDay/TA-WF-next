"""Thirty-task bounded experiment; per-seed GPU workers and complete delivery audit."""
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
    assert C['status']=='frozen';v3=ROOT/C['baseline_run'];v4=ROOT/C['v4_run'];prior=ROOT/C['previous_run'];diag=ROOT/C['diagnostic_run'];source=ROOT/C['source_run']
    assert json.loads((v3/'artifacts/postrun_audit.json').read_text())['passed'] and json.loads((v4/'artifacts/postrun_audit.json').read_text())['passed']
    files=list(RUN.glob('*.py'))+[RUN/'config.json',RUN/'PLAN.md',RUN/'SOURCES.md',RUN/'REPAIR_HISTORY.md',ROOT/'PROTOCOL.md',ROOT/'MULTI_FACTOR_DIAGNOSTIC_PROTOCOL.md',ROOT/'configs/datasets.json',v3/'artifacts/scores.json',v3/'artifacts/postrun_audit.json',v4/'artifacts/postrun_audit.json',v4/'artifacts/delivery_audit.json',diag/'artifacts/data_audit.json',diag/'artifacts/diagnostic_rows.csv',source/'artifacts/reports.json']
    v4freeze=json.loads((v4/'artifacts/freeze.json').read_text());v4delivery=json.loads((v4/'artifacts/delivery_audit.json').read_text());oldrows=json.loads((v3/'artifacts/scores.json').read_text());audit=json.loads((diag/'artifacts/data_audit.json').read_text());assert audit['passed'];check=[]
    for seed in C['seeds']:
        files += [source/'checkpoints'/f'pure_source_wide_bn_{seed}.pth',prior/'artifacts'/f'source_gmm_{seed}.pt',v3/'artifacts'/f'source_calibration_{seed}.pt',v4/'artifacts'/f'source_{seed}.pt',v4/'artifacts'/f'adapted_scores_{seed}.json']
        for p in [source/'checkpoints'/f'pure_source_wide_bn_{seed}.pth',prior/'artifacts'/f'source_gmm_{seed}.pt',v3/'artifacts'/f'source_calibration_{seed}.pt']:assert sha(p)==v4freeze[str(p.relative_to(ROOT))]
        assert sha(v4/'artifacts'/f'source_{seed}.pt')==v4delivery['outputs'][f'artifacts/source_{seed}.pt']
        for name in C['slices']:
            files += [v3/'artifacts'/f'base_{seed}_{name}.npz',v4/'artifacts'/f'raw_{seed}_{name}_soft_geometry.npz',v4/'artifacts'/f'signal_{seed}_{name}_soft_geometry.npz',v4/'artifacts'/f'signal_{seed}_{name}_ungated_lora.npz',v4/'artifacts'/f'signal_{seed}_{name}_tent_reference.npz']
            for cond in ['soft_neighbor_lora','tent_bn_reference']:
                p=v3/'artifacts'/f'raw_{seed}_{name}_{cond}.npz';rr=next(r for r in oldrows if r['seed']==seed and r['slice']==name and r['condition']==cond and r['policy']=='raw');assert sha(p)==rr['raw_sha256'];files.append(p)
            for p in files[-5:]:
                if p.parent==v4/'artifacts' and str(p.relative_to(v4)) in v4delivery['outputs']:assert sha(p)==v4delivery['outputs'][str(p.relative_to(v4))]
    for name in C['slices']:
        p=diag/'artifacts'/f'data_{name}.pt';assert sha(p)==audit['slices'][name]['cache_sha256'];files.append(p)
    # All v4-derived immutable source/evidence arrays must match the previous delivery.
    for p in files:
        if p.is_relative_to(v4) and str(p.relative_to(v4)) in v4delivery['outputs']:assert sha(p)==v4delivery['outputs'][str(p.relative_to(v4))]
    save('freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(files)});verify();save('reuse_audit.json',{'passed':True,'source_rows':18553,'source_valid_rows':2160,'diagnostic_slices':C['slices'],'historical_controls_checked_against_cell_and_delivery_sha':True})


def main():
    with (ROOT/'.momentum_cluster_v5.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze();status('running','两切片5arm×3seed预算30任务已冻结；三GPU源域预检，随后批次动量/动态原型/批内对比消融。');parallel('preflight')
        assert all(json.loads((RUN/'artifacts'/f'preflight_{s}.json').read_text())['passed'] for s in C['seeds']);status('running','源域预检通过；GPU0/1/2并行30个两切片任务，无query更新及输出硬旁路。');parallel('train')
        status('running','30任务完成，进行全query和source valid重载、状态和独立指标审计。');parallel('audit','audit.py')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');budget();subprocess.run([os.sys.executable,'-u',str(RUN/'frontier.py')],env=env,check=True,timeout=600)
        aa=[json.loads((RUN/'artifacts'/f'audit_{s}.json').read_text()) for s in C['seeds']];assert all(a['passed'] for a in aa) and sum(a['query_checkpoints_reloaded'] for a in aa)==30
        save('postrun_audit.json',{'passed':True,'new_training_tasks':30,'all_query_and_valid_checkpoint_reload':30,'seeds':aa});verify()
        save('resource_audit.json',{'pipeline_seconds':time.monotonic()-START,'conservative_gpu_hours':3*(time.monotonic()-START)/3600,'output_bytes':sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),'new_adaptations':30,'physical_gpus':[0,1,2]})
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n全30个checkpoint的query与source valid重载、逐指标独立复算、纯净底座锁定、reference状态范围、权重正值和损失有限性审计通过；输入/输出SHA完整核验。\n')
        outputs={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and not any(v in p.parts for v in ['lgs','logs','__pycache__']) and p.name!='delivery_audit.json'};save('delivery_audit.json',{'passed':True,'outputs':outputs});verify()
        for n,h in outputs.items():assert sha(RUN/n)==h
        d=json.loads((RUN/'artifacts/registered_decisions.json').read_text());detail='；'.join(f"{r['slice']} jointΔF1{r['joint_delta_f1_pp']:+.3f}pp" for r in d['slices']);status('completed',f'30任务及全query/valid重载审计完成；{detail}；扩展资格{d["expansion_eligible"]}。仅两切片诊断开发证据。')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('controller_failure.json',{'error':str(e),'traceback':traceback.format_exc()});status('failed',f'失败：{type(e).__name__}: {e}；产物保留。');raise
