import os,sys,time,subprocess,json,signal,traceback
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from common import RUN,ROOT,config,verify_freeze,save

def update(state,message):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',message],check=True)
    p=ROOT/'STATUS.md';lines=[v for v in p.read_text().splitlines() if not v.startswith('当前GPU卷积分类器对照：')]
    lines.insert(1,'当前GPU卷积分类器对照：`'+RUN.name+'`。'+message);p.write_text('\n'.join(lines)+'\n')

def finalize(c):
    verify_freeze();torch.set_num_threads(2)
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');reports=[];checks=0
    for seed in c['seeds']:
        refdir=RUN/'artifacts'/f'baseline_{seed}'
        ref=torch.load(refdir/'initial_state.pt',weights_only=True,map_location='cpu');stream=torch.load(refdir/'index_stream.pt',weights_only=True,map_location='cpu')
        for kind in ['baseline','temporal_cnn']:
            out=RUN/'artifacts'/f'{kind}_{seed}';rep=json.loads((out/'report.json').read_text())
            assert rep['device']=='cuda' and rep['steps_completed']==12800 and rep['validation_opportunities']==20
            assert rep['inactive_parameters_unchanged'] and rep['verified_roles']==['source','valid']
            history=json.loads((out/'history.json').read_text());assert [v['step'] for v in history]==c['eval_steps']
            assert max(history,key=lambda v:v['valid']['accuracy'])['step']==rep['best_step']
            assert torch.equal(torch.load(out/'index_stream.pt',weights_only=True,map_location='cpu'),stream)
            state=torch.load(out/'initial_state.pt',weights_only=True,map_location='cpu')
            for k,v in ref.items():
                if kind=='temporal_cnn' and k.startswith('head.'):continue
                assert torch.equal(v,state[k]),k
            for ck in ['best','last']:
                with np.load(out/f'predictions_{ck}.npz') as f:
                    for role in ['source','valid']:
                        y=raw[role]['labels'].numpy();pred=f[role]
                        assert pred.shape==y.shape and np.isin(pred,np.arange(102)).all()
                        assert abs(accuracy_score(y,pred)-rep[ck][role]['accuracy'])<1e-12
                        assert abs(f1_score(y,pred,labels=np.arange(102),average='macro',zero_division=0)-rep[ck][role]['macro_f1'])<1e-12;checks+=1
            reports.append(rep)
    means={k:{m:float(np.mean([v['best']['valid'][m] for v in reports if v['kind']==k])) for m in ['accuracy','macro_f1']} for k in ['baseline','temporal_cnn']}
    ds=[];fs=[]
    for seed in c['seeds']:
        a=next(v for v in reports if v['kind']=='temporal_cnn' and v['seed']==seed);b=next(v for v in reports if v['kind']=='baseline' and v['seed']==seed)
        ds.append(a['best']['valid']['accuracy']-b['best']['valid']['accuracy']);fs.append(a['best']['valid']['macro_f1']-b['best']['valid']['macro_f1'])
    passed=bool(np.mean(ds)>=.01 and min(ds)>0 and np.mean(fs)>=0)
    out={'reports':reports,'means':means,'accuracy_mean_delta_pp':float(np.mean(ds)*100),'accuracy_per_seed_pp':[float(v*100) for v in ds],'macro_f1_mean_delta_pp':float(np.mean(fs)*100),'pass':passed,'metric_groups_verified':checks,'target_reached':means['temporal_cnn']['accuracy']>=.9,'historical_baseline_seeds':3,'cpu_history_used_as_training':False}
    save(RUN/'artifacts/summary.json',out)
    msg=f"3项GPU CNN完成；accuracy {means['temporal_cnn']['accuracy']*100:.3f}% vs baseline {means['baseline']['accuracy']*100:.3f}%；增量{np.mean(ds)*100:+.3f}pp PASS={passed}；24组指标和冻结hash通过。"
    rows=['# GPU保序卷积分类器对照结果','',msg,'','|模型|seed|best步数|source accuracy|valid accuracy|valid F1|','|---|---:|---:|---:|---:|---:|']
    for v in reports:rows.append(f"|{v['kind']}|{v['seed']}|{v['best_step']}|{v['best']['source']['accuracy']*100:.3f}%|{v['best']['valid']['accuracy']*100:.3f}%|{v['best']['valid']['macro_f1']*100:.3f}%|")
    rows+=['','```json',json.dumps({k:v for k,v in out.items() if k!='reports'},indent=2),'```','','新CNN三seed全部GPU从头初始化；baseline三seed明确历史GPU复用，不是新重复；CPU部分与已完成单seed保留在前run，不混入均值或用于warm start。此设备切换经用户授权，有既往valid反馈暴露，属于开发而非独立确认。','结构、参数和计算量不同，其他共享初始state、数据、索引、步数、选模机会一致；不能称纯attention消融或计算量匹配。候选门槛与90%分别判断，无未来日期/适应或抗漂移结论。']
    (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');update('completed',msg)

def main():
    c=config();verify_freeze();active=[];done=[];start=time.monotonic()
    update('running','用户授权CPU转GPU；3seed保序卷积从头训练，复用3seed GPU Transformer对照。GPU0三路、12800步/20次选模；原CPU产物保留且不混入均值，未来关闭。')
    signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(RuntimeError('supervisor terminated')))
    try:
        for seed in c['seeds']:
            key=f'temporal_cnn_{seed}';log=(RUN/'logs'/f'{key}.log').open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            p=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),'--kind','temporal_cnn','--seed',str(seed),'--device','cuda'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append((key,p,log))
        while active:
            if time.monotonic()-start>min(c['pipeline_seconds'],c['gpu_job_seconds']+120):raise TimeoutError('GPU queue budget')
            for key,p,log in list(active):
                if p.poll() is not None:
                    log.close();active.remove((key,p,log))
                    if p.returncode:raise RuntimeError(key+' failed')
                    assert (RUN/'artifacts'/key/'report.json').exists();done.append(key)
            save(RUN/'artifacts/progress.json',{'status':'running','active':[{'task':k,'pid':p.pid,'device':'cuda'} for k,p,l in active],'completed':done,'new_total':3,'historical_reused':3,'elapsed_seconds':time.monotonic()-start});time.sleep(5)
        finalize(c);save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'historical_reused':3,'elapsed_seconds':time.monotonic()-start})
    except BaseException:
        for key,p,log in active:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
            log.close()
        err=traceback.format_exc();(RUN/'logs/error.txt').write_text(err)
        save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err})
        update('failed','GPU对照执行中止，保留产物，见logs/error.txt；未追加预算。');raise

if __name__=='__main__':main()
