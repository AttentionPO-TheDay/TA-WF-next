"""Freeze source-calibrated methods, three-GPU execution, full audit and delivery."""
import fcntl,json,os,subprocess,time,traceback
from runtime import RUN,ROOT,C,sha,save,status,verify

def parallel(script):
    jobs=[];handles=[];begin=time.monotonic()
    try:
        for seed in C['seeds']:
            gpu=C['budget']['seed_gpu_map'][str(seed)];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CANDIDATE_SEED=str(seed),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            h=(RUN/'logs'/f'{script}_{seed}.out').open('ab',buffering=0);handles.append(h)
            p=subprocess.Popen([os.sys.executable,'-u',str(RUN/script)],env=env,stdout=h,stderr=subprocess.STDOUT);jobs.append((seed,p))
        save('worker_processes.json',[{'seed':seed,'pid':p.pid,'gpu':C['budget']['seed_gpu_map'][str(seed)]} for seed,p in jobs])
        while True:
            for seed,p in jobs:
                if p.poll() not in (None,0):raise RuntimeError(f'{script} seed{seed} exit{p.returncode}')
            if all(p.poll()==0 for _,p in jobs):break
            if time.monotonic()-begin>C['budget']['max_wall_seconds']:raise TimeoutError('matrix wall')
            if sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file())>C['budget']['max_output_gib']*1024**3:raise TimeoutError('storage')
            time.sleep(3)
    finally:
        for _,p in jobs:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
        for h in handles:h.close()

def main():
    begin=time.time()
    with (ROOT/'.candidate_accept_v3.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pre=[]
        for seed in C['seeds']:
            p=json.loads((RUN/'artifacts'/f'preflight_{seed}.json').read_text());assert p['passed'];pre.append(p)
        files=[RUN/'config.json',RUN/'PLAN.md',RUN/'SOURCES.md',ROOT/'PROTOCOL.md',ROOT/'MULTI_FACTOR_DIAGNOSTIC_PROTOCOL.md',ROOT/'configs/datasets.json',ROOT/C['source_run']/'artifacts/prepared.pt',ROOT/C['source_run']/'artifacts/reports.json']+list(RUN.glob('*.py'))
        prior=ROOT/C['previous_run'];diag=ROOT/C['diagnostic_run'];data=json.loads((diag/'artifacts/data_audit.json').read_text());assert data['passed']
        files.extend([diag/'artifacts/data_audit.json',diag/'artifacts/diagnostic_rows.csv',diag/'DIAGNOSTIC_PROTOCOL.md',prior/'artifacts/postrun_audit.json'])
        for seed in C['seeds']:
            files.extend([RUN/'artifacts'/f'preflight_{seed}.json',RUN/'artifacts'/f'source_calibration_{seed}.pt',prior/'artifacts'/f'source_gmm_{seed}.pt',prior/'artifacts'/f'source_preflight_{seed}.json',ROOT/C['source_run']/'checkpoints'/f'pure_source_wide_bn_{seed}.pth'])
        for name in C['slices']:
            p=diag/'artifacts'/f'data_{name}.pt';assert sha(p)==data['slices'][name]['cache_sha256'];files.append(p)
        save('freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in files});verify()
        save('source_gate.json',{'passed':True,'source_only':True,'source_valid_labels_used_for_threshold_only':True,'acceptance_enabled':{str(p['seed']):p['acceptance_enabled'] for p in pre},'unavailable_branch':'No feasible neighbor-evidence acceptance threshold in any seed; registered fallback is frozen. No threshold relaxation, no target feedback used.','time_unix':time.time()})
        status('running','源域校准与代码/数据已冻结；GPU0/1/2运行硬/软教师/软邻居×Residual/LoRA及同预算TENT，630评分状态；邻域验收源校准不可行，保留预定回退。')
        parallel('worker.py');scores=[]
        for seed in C['seeds']:scores.extend(json.loads((RUN/'artifacts'/f'scores_{seed}.json').read_text()))
        assert len(scores)==630;save('scores.json',scores)
        for script,timeout in [('audit.py',2400),('report.py',600)]:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            subprocess.run([os.sys.executable,'-u',str(RUN/script)],env=env,check=True,timeout=timeout)
        save('resource_audit.json',{'pipeline_seconds':time.time()-begin,'conservative_gpu_hours':3*(time.time()-begin)/3600,'output_bytes':sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),'physical_gpus':[0,1,2],'source_preflight_seconds_by_seed':{str(p['seed']):p['seconds'] for p in pre}})
        verify();d=json.loads((RUN/'artifacts/delivery_audit.json').read_text());assert d['passed']
        for rel,h in d['outputs'].items():assert sha(RUN/rel)==h
        status('completed','630评分状态完成（210适应训练、270原始状态及360验收状态）；全量checkpoint/指标/旁路审计通过，完整主表及五组图已导出；仅诊断开发证据。')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('controller_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});status('failed',f'候选纠错实验失败：{type(e).__name__}: {e}；产物保留。');raise
