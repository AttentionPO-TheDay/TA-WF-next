import json, os, signal, subprocess, sys, time, traceback
import numpy as np
from common import RUN, ROOT, config, verify_freeze, save

def update(state, summary): subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
def main():
    c=config(); verify_freeze(); assert __import__('torch').cuda.is_available()
    todo=list(c['seeds']); active=[]; done=[]; start=time.monotonic(); update('running','SWA GPU实验启动；三seed共享预注册后段平均窗口，未来关闭。')
    try:
        while todo or active:
            if time.monotonic()-start>c['pipeline_seconds']: raise TimeoutError('whole pipeline budget exceeded')
            while todo and len(active)<c['max_gpu_parallel']:
                seed=todo.pop(0); log=(RUN/'logs'/f'ce_trajectory_{seed}.log').open('x'); env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'); p=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),'--seed',str(seed),'--device','cuda'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); active.append((seed,p,log,time.monotonic()))
            for item in list(active):
                seed,p,log,launched=item
                if p.poll() is not None:
                    log.close(); active.remove(item)
                    if p.returncode!=0: raise RuntimeError(f'seed {seed} failed with {p.returncode}')
                    assert (RUN/'artifacts'/f'ce_trajectory_{seed}'/'report.json').exists(); done.append(seed)
                elif time.monotonic()-launched>c['gpu_job_seconds']+300: raise TimeoutError(f'seed {seed} watchdog exceeded')
            save(RUN/'artifacts/progress.json',{'status':'running','active':[x[0] for x in active],'completed':done,'pending':todo,'elapsed_seconds':time.monotonic()-start}); time.sleep(5)
        reports=[json.loads((RUN/'artifacts'/f'ce_trajectory_{s}'/'report.json').read_text()) for s in c['seeds']]
        means={k:{m:float(np.mean([r[k]['valid'][m] for r in reports])) for m in ('accuracy','macro_f1')} for k in ('ce_final','swa')}
        da=[(r['swa']['valid']['accuracy']-r['ce_final']['valid']['accuracy'])*100 for r in reports]; df=[(r['swa']['valid']['macro_f1']-r['ce_final']['valid']['macro_f1'])*100 for r in reports]
        comp={'accuracy_mean_delta_pp':float(np.mean(da)),'accuracy_per_seed_pp':da,'macro_f1_mean_delta_pp':float(np.mean(df)),'positive_seed_count':int(sum(x>0 for x in da)),'pass':bool(np.mean(da)>=1.0 and min(da)>0 and np.mean(df)>=0)}
        summary={'means':means,'comparison_swa_minus_ce_final':comp,'reports':reports,'future_access':False,'training_tasks_completed':len(reports),'swa_steps':c['swa_steps']}; save(RUN/'artifacts/summary.json',summary)
        rows=['# SWA结果','','三条 CE 轨迹完成；SWA 使用预注册后段七个 checkpoint 的固定平均。','','| 条件 | valid accuracy | valid Macro-F1 |','|---|---:|---:|']
        for k in ('ce_final','swa'): rows.append(f"| {k} | {means[k]['accuracy']*100:.3f}% | {means[k]['macro_f1']*100:.3f}% |")
        rows += ['', '```json', json.dumps({'swa - ce_final':comp},ensure_ascii=False,indent=2), '```', '', '未访问未来日期、WTT-Time、AWF 或 TTA。']; (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n'); save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'elapsed_seconds':time.monotonic()-start}); update('completed','SWA三seed训练与固定窗口平均完成；结果见RESULTS.md，未来关闭。')
    except BaseException:
        for _,p,log,_ in active:
            if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
            log.close()
        err=traceback.format_exc(); (RUN/'logs/error.txt').write_text(err); save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err}); update('failed','SWA训练中止，部分产物保留，见logs/error.txt。'); raise
if __name__=='__main__': main()
