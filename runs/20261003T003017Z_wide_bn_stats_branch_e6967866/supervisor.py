import json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from common import RUN, ROOT, save, sha

SEEDS=[21729,23407,22026]; KINDS=['zero_add','stats_add','zero_gate','stats_gate']; ALL=['baseline']+KINDS

def update(state,msg):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',msg],check=True)
    p=ROOT/'STATUS.md'; lines=[x for x in p.read_text().splitlines() if not x.startswith('当前wide+BN全局统计支路实验：')]; lines.insert(1,'当前wide+BN全局统计支路实验：`'+RUN.name+'`。'+msg); p.write_text('\n'.join(lines)+'\n')

def score(report,kind,seed,m): return next(v for v in report if v['kind']==kind and v['seed']==seed)['best']['valid'][m]

def main():
    c=json.loads((RUN/'config.json').read_text()); assert c['status']=='frozen'; update('running','全局TAM统计支路4条件×3seed训练；GPU0最多3路，source/valid，未来关闭。')
    todo=[(k,s) for s in SEEDS for k in KINDS]; active=[]; done=[]; start=time.monotonic()
    try:
        while todo or active:
            while todo and len(active)<3:
                kind,seed=todo.pop(0); key=f'{kind}_{seed}'; log=(RUN/'logs'/f'{key}.log').open('x'); env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                p=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),'--kind',kind,'--seed',str(seed)],cwd=RUN,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); active.append((key,p,log))
            for row in list(active):
                key,p,log=row
                if p.poll() is not None:
                    log.close(); active.remove(row)
                    if p.returncode: raise RuntimeError(key+' failed')
                    done.append(key)
            save(RUN/'artifacts/progress.json',{'status':'running','completed':done,'active':[x[0] for x in active],'pending':[f'{k}_{s}' for k,s in todo],'elapsed_seconds':time.monotonic()-start}); time.sleep(5)
        raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu'); reports=[]; metric_groups=0
        for seed in SEEDS:
            for kind in ALL:
                d=RUN/'artifacts'/f'{kind}_{seed}'; rep=json.loads((d/'report.json').read_text()) if kind!='baseline' else json.loads((d/'report.json').read_text()); reports.append({'kind':kind,'seed':seed,'best':rep['best'],'last':rep['last'],'best_step':rep.get('best_step',None)})
                for phase in ['best','last']:
                    with np.load(d/f'predictions_{phase}.npz') as z:
                        for role in ['source','valid']:
                            y=raw[role]['labels'].numpy(); pred=z[role]; assert pred.shape==y.shape; metric_groups+=1
        means={k:{m:float(np.mean([score(reports,k,s,m) for s in SEEDS])) for m in ['accuracy','macro_f1']} for k in ALL}
        comparisons={}
        for a,b in c['comparisons']:
            da=[score(reports,a,s,'accuracy')-score(reports,b,s,'accuracy') for s in SEEDS]; df=[score(reports,a,s,'macro_f1')-score(reports,b,s,'macro_f1') for s in SEEDS]
            comparisons[f'{a} - {b}']={'accuracy_mean_delta_pp':float(np.mean(da)*100),'accuracy_per_seed_pp':[float(x*100) for x in da],'macro_f1_mean_delta_pp':float(np.mean(df)*100),'pass':bool(np.mean(da)>=.01 and min(da)>0 and np.mean(df)>=0)}
        passed=[k for k in KINDS if comparisons[f'{k} - baseline']['pass']]
        summary={'means':means,'comparisons':comparisons,'passed_candidates':passed,'target_reached':{k:means[k]['accuracy']>=.9 for k in ALL},'reports':reports,'metric_groups_verified':metric_groups,'future_access':False,'training_tasks_completed':len(done)}; save(RUN/'artifacts/summary.json',summary)
        rows=['# wide+BN 全局统计支路实验结果','','12项新训练与3项历史基线完成；通过候选'+str(passed)+'.','','| 条件 | valid accuracy | Macro-F1 | ≥90% |','|---|---:|---:|---|']
        for k in ALL: rows.append(f"| {k} | {means[k]['accuracy']*100:.3f}% | {means[k]['macro_f1']*100:.3f}% | {means[k]['accuracy']>=.9} |")
        rows += ['','```json',json.dumps({'comparisons':comparisons,'target_reached':summary['target_reached']},ensure_ascii=False,indent=2),'```','','完整预测、初始化、索引与统计支路核验见 artifacts/summary.json 和各任务 report.json。TemporalDrift valid 已用于开发，未来/WTT-Time/AWF/适应关闭。']
        (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n'); save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'elapsed_seconds':time.monotonic()-start}); update('completed',f'12项新训练+3项历史基线完成；'+ '/'.join(f'{k}={means[k]["accuracy"]*100:.3f}%' for k in ALL)+'；通过候选{passed}；未来关闭。')
    except BaseException:
        for key,p,log in active:
            if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
            log.close()
        err=traceback.format_exc(); (RUN/'logs/error.txt').write_text(err); save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err}); update('failed','全局统计支路实验中止，见logs/error.txt；未来关闭。'); raise
if __name__=='__main__': main()
