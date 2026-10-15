"""Run and aggregate two attribution conditions; previous results are references."""
import json, os, signal, subprocess, sys, time, traceback
import numpy as np
import torch
from common import RUN, ROOT, config, verify_freeze, save

def update(state, summary): subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)

def main():
    c=config();verify_freeze();assert torch.cuda.is_available();todo=[(k,s) for s in c['seeds'] for k in c['conditions']];active=[];done=[];start=time.monotonic();update('running','GPU0归因训练启动；primary CE-only-joint/primary CE+peer×3seed，上一轮双视图结果只读参考。')
    try:
        while todo or active:
            if time.monotonic()-start>c['pipeline_seconds']:raise TimeoutError('whole pipeline budget exceeded')
            while todo and len(active)<c['max_gpu_parallel']:
                kind,seed=todo.pop(0);key=f'{kind}_{seed}';log=(RUN/'logs'/f'{key}.log').open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');p=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),'--kind',kind,'--seed',str(seed),'--device','cuda'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append((key,p,log,time.monotonic()))
            for item in list(active):
                key,p,log,launched=item
                if p.poll() is not None:
                    log.close();active.remove(item)
                    if p.returncode!=0:raise RuntimeError(f'{key} failed with {p.returncode}')
                    assert (RUN/'artifacts'/key/'report.json').exists();done.append(key)
                elif time.monotonic()-launched>c['gpu_job_seconds']+300:raise TimeoutError(f'{key} watchdog exceeded')
            save(RUN/'artifacts/progress.json',{'status':'running','active':[x[0] for x in active],'completed':done,'pending':[f'{k}_{s}' for k,s in todo],'elapsed_seconds':time.monotonic()-start});time.sleep(5)
        refs=[];reports=[]
        for s in c['seeds']:
            r=json.loads((RUN/'artifacts'/f'ce_only_{s}'/'report.json').read_text());r['kind']='ce_only';r['seed']=s;refs.append(r)
            for base in ('dual_mask_ce','mask_peer'):
                r=json.loads((RUN/'artifacts'/f'{base}_{s}'/'report.json').read_text());r['kind']=base;r['seed']=s;refs.append(r)
        for key in done:reports.append(json.loads((RUN/'artifacts'/key/'report.json').read_text()))
        allr=refs+reports;names=['ce_only','dual_mask_ce','mask_peer','primary_ce_only_joint','primary_ce_peer'];means={k:{m:float(np.mean([r['best']['valid'][m] for r in allr if r['kind']==k])) for m in ('accuracy','macro_f1')} for k in names};comparisons={}
        for k,b in [('primary_ce_peer','primary_ce_only_joint'),('primary_ce_only_joint','ce_only'),('primary_ce_peer','ce_only'),('mask_peer','dual_mask_ce')]:
            da=[next(r['best']['valid']['accuracy'] for r in allr if r['kind']==k and r['seed']==s)-next(r['best']['valid']['accuracy'] for r in allr if r['kind']==b and r['seed']==s) for s in c['seeds']];df=[next(r['best']['valid']['macro_f1'] for r in allr if r['kind']==k and r['seed']==s)-next(r['best']['valid']['macro_f1'] for r in allr if r['kind']==b and r['seed']==s) for s in c['seeds']];comparisons[f'{k} - {b}']={'accuracy_mean_delta_pp':float(np.mean(da)*100),'accuracy_per_seed_pp':[float(x*100) for x in da],'macro_f1_mean_delta_pp':float(np.mean(df)*100),'positive_seed_count':int(sum(x>0 for x in da)),'pass':bool(np.mean(da)>=.01 and min(da)>0 and np.mean(df)>=0)}
        summary={'means':means,'comparisons':comparisons,'reports':allr,'future_access':False,'training_tasks_completed':len(reports),'historical_reference_tasks':len(refs),'single_network_peer':True,'fixed_teacher_loaded':False,'ema_loaded':False,'valid_selection':'predeclared checkpoint selection only'};save(RUN/'artifacts/summary.json',summary);rows=['# Peer归因实验结果','','GPU0 两个新条件×三 seed完成；上一轮结果只读参考。','','| 条件 | valid accuracy | valid Macro-F1 |','|---|---:|---:|']
        for k in names:rows.append(f"| {k} | {means[k]['accuracy']*100:.3f}% | {means[k]['macro_f1']*100:.3f}% |")
        rows += ['', '```json',json.dumps(comparisons,ensure_ascii=False,indent=2),'```','','主要比较 primary_ce_peer 与 primary_ce_only_joint；历史 dual_mask_ce/mask_peer/ce_only 仅用于解释。未来/WTT-Time/AWF/TTA关闭。'];(RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'elapsed_seconds':time.monotonic()-start});update('completed','6项归因GPU训练完成；primary CE-only-joint/primary CE+peer及历史参考汇总见RESULTS.md；未来关闭。')
    except BaseException:
        for key,p,log,_ in active:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            log.close()
        err=traceback.format_exc();(RUN/'logs/error.txt').write_text(err);save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err});update('failed','GPU0归因训练中止，部分产物保留，见logs/error.txt。');raise

if __name__=='__main__':main()
