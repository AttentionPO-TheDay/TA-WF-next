import fcntl,json,os,subprocess,time,traceback
from runtime import RUN,ROOT,C,sha,save,status,verify

def parallel(script):
    jobs=[];hs=[];start=time.monotonic()
    try:
        for seed in C['seeds']:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(C['budget']['seed_gpu_map'][str(seed)]),COMBO_SEED=str(seed),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            h=(RUN/'logs'/f'{script}_{seed}.out').open('ab',buffering=0);hs.append(h)
            p=subprocess.Popen([os.sys.executable,'-u',str(RUN/script)],env=env,stdout=h,stderr=subprocess.STDOUT);jobs.append((seed,p))
        save('processes_'+script+'.json',[{'seed':seed,'pid':p.pid,'physical_gpu':C['budget']['seed_gpu_map'][str(seed)]} for seed,p in jobs])
        while True:
            for seed,p in jobs:
                if p.poll() not in (None,0):raise RuntimeError(f'{script} seed{seed} exit{p.returncode}')
            if all(p.poll()==0 for _,p in jobs):break
            limit=1200 if script=='prepare_source.py' else C['budget']['max_wall_seconds']
            if time.monotonic()-start>limit:raise TimeoutError(script)
            if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())>C['budget']['max_output_gib']*1024**3:raise TimeoutError('storage')
            time.sleep(3)
    finally:
        for _,p in jobs:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
        for h in hs:h.close()

def main():
    begin=time.time()
    with (ROOT/'.gmm_routing_lora_v2.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status('running','已登记8组合×3seed×10切片240单元；GPU0/1/2进行纯源GMM/门控校准与LoRA机械预检，尚未新目标评分。')
        parallel('prepare_source.py')
        frozen={}
        files=[RUN/'config.json',RUN/'PLAN.md',RUN/'SOURCES.md',ROOT/'PROTOCOL.md',ROOT/'MULTI_FACTOR_DIAGNOSTIC_PROTOCOL.md',ROOT/C['source_run']/'artifacts/prepared.pt',ROOT/C['source_run']/'artifacts/reports.json']+list(RUN.glob('*.py'))
        for seed in C['seeds']:
            pre=json.loads((RUN/'artifacts'/f'source_preflight_{seed}.json').read_text());assert pre['passed']
            files.extend([RUN/'artifacts'/f'source_gmm_{seed}.pt',RUN/'artifacts'/f'source_preflight_{seed}.json',ROOT/C['source_run']/'checkpoints'/f'pure_source_wide_bn_{seed}.pth'])
        prior=ROOT/C['diagnostic_run'];data=json.loads((prior/'artifacts/data_audit.json').read_text());assert data['passed']
        files.extend([prior/'artifacts/data_audit.json',prior/'artifacts/diagnostic_rows.csv',prior/'artifacts/postrun_audit.json',prior/'DIAGNOSTIC_PROTOCOL.md'])
        for name in C['slices']:
            p=prior/'artifacts'/f'data_{name}.pt';assert sha(p)==data['slices'][name]['cache_sha256'];files.append(p)
        for p in files:frozen[str(p.relative_to(ROOT))]=sha(p)
        save('freeze.json',frozen);verify()
        save('preflight.json',{'passed':True,'source_only':True,'seeds':C['seeds'],'frozen_files':len(frozen),'time_unix':time.time()})
        status('running','源域GMM/门控阈值和Q/V LoRA检查通过并冻结；GPU0/1/2各一seed运行240单元，8组合匹配400步，无外部新评分。')
        parallel('worker.py')
        scores=[]
        for seed in C['seeds']:scores+=json.loads((RUN/'artifacts'/f'scores_{seed}.json').read_text())
        assert len(scores)==240;save('scores.json',scores)
        # Audit/report scripts are frozen before target scoring like method code.
        for script,timeout in [('audit.py',2400),('report.py',600)]:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            subprocess.run([os.sys.executable,'-u',str(RUN/script)],env=env,check=True,timeout=timeout)
        save('resource_audit.json',{'pipeline_seconds':time.time()-begin,'conservative_gpu_hours':3*(time.time()-begin)/3600,'output_bytes':sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),'physical_gpus':[0,1,2]})
        status('completed','240单元全部完成；checkpoint重载/冻结参数/推理旁路/指标复算审计通过，完整组合表与因子效应已导出；仅已观察诊断证据。')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('controller_failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});status('failed',f'组合实验失败：{type(e).__name__}: {e}，产物保留。');raise
