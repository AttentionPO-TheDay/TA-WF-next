"""Twelve predeclared GPU0 tasks, three historical baseline seeds."""
import os,sys,time,subprocess,json,signal,traceback
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from common import RUN,ROOT,config,verify_freeze,save

def update(state,message):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',message],check=True)
    p=ROOT/'STATUS.md';lines=[v for v in p.read_text().splitlines() if not v.startswith('当前网站指纹生成器跨头对照：')]
    lines.insert(1,'当前网站指纹生成器跨头对照：`'+RUN.name+'`。'+message);p.write_text('\n'.join(lines)+'\n')

def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def finalize(c):
    verify_freeze();torch.set_num_threads(2)
    raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu');reports=[];checks=0
    kinds=c['reported_conditions']
    for seed in c['seeds']:
        base=RUN/'artifacts'/f'shallow_transformer_{seed}'
        ref=torch.load(base/'initial_state.pt',weights_only=True,map_location='cpu');stream=torch.load(base/'index_stream.pt',weights_only=True,map_location='cpu')
        for kind in kinds:
            out=RUN/'artifacts'/f'{kind}_{seed}';rep=json.loads((out/'report.json').read_text())
            assert rep['device']=='cuda' and rep['steps_completed']==12800 and rep['validation_opportunities']==20
            assert rep['inactive_parameters_unchanged'] and rep['verified_roles']==['source','valid']
            assert rep['parameters_total']==c['parameter_counts'][kind]['total'] and rep['parameters_trainable']==c['parameter_counts'][kind]['trainable']
            history=json.loads((out/'history.json').read_text());assert [v['step'] for v in history]==c['eval_steps']
            assert max(history,key=lambda v:v['valid']['accuracy'])['step']==rep['best_step']
            assert torch.equal(torch.load(out/'index_stream.pt',weights_only=True,map_location='cpu'),stream)
            state=torch.load(out/'initial_state.pt',weights_only=True,map_location='cpu')
            if kind!='rf_matched':
                for k,v in ref.items():
                    if not k.startswith(('generator.','head.')):assert torch.equal(v,state[k]),(kind,k)
                if kind.startswith('shallow'):assert all(torch.equal(v,state[k]) for k,v in ref.items() if k.startswith('generator.'))
                if kind.endswith('transformer'):assert all(torch.equal(v,state[k]) for k,v in ref.items() if k.startswith('head.'))
            if kind=='progressive_mlp':
                other=torch.load(RUN/'artifacts'/f'progressive_transformer_{seed}'/'initial_state.pt',weights_only=True)
                assert all(torch.equal(v,other[k]) for k,v in state.items() if k.startswith('generator.'))
                other=torch.load(RUN/'artifacts'/f'shallow_mlp_{seed}'/'initial_state.pt',weights_only=True)
                assert all(torch.equal(v,other[k]) for k,v in state.items() if k.startswith('head.'))
            for checkpoint in ['best','last']:
                with np.load(out/f'predictions_{checkpoint}.npz') as f:
                    for role in c['roles']:
                        y=raw[role]['labels'].numpy();pred=f[role]
                        assert pred.shape==y.shape and np.isin(pred,np.arange(102)).all()
                        assert abs(accuracy_score(y,pred)-rep[checkpoint][role]['accuracy'])<1e-12
                        assert abs(f1_score(y,pred,labels=np.arange(102),average='macro',zero_division=0)-rep[checkpoint][role]['macro_f1'])<1e-12;checks+=1
            reports.append(rep)
    means={kind:{m:float(np.mean([v['best']['valid'][m] for v in reports if v['kind']==kind])) for m in ['accuracy','macro_f1']} for kind in kinds}
    def score(kind,seed,metric):return next(v for v in reports if v['kind']==kind and v['seed']==seed)['best']['valid'][metric]
    comparisons={}
    for a,b in c['comparisons']:
        ds=[score(a,s,'accuracy')-score(b,s,'accuracy') for s in c['seeds']];fs=[score(a,s,'macro_f1')-score(b,s,'macro_f1') for s in c['seeds']]
        comparisons[a+' - '+b]={'accuracy_mean_delta_pp':float(np.mean(ds)*100),'accuracy_per_seed_pp':[float(v*100) for v in ds],'macro_f1_mean_delta_pp':float(np.mean(fs)*100),'pass':bool(np.mean(ds)>=.01 and min(ds)>0 and np.mean(fs)>=0)}
    interaction={}
    for m in ['accuracy','macro_f1']:
        ds=[(score('progressive_transformer',s,m)-score('shallow_transformer',s,m))-(score('progressive_mlp',s,m)-score('shallow_mlp',s,m)) for s in c['seeds']]
        interaction[m]={'mean_pp':float(np.mean(ds)*100),'per_seed_pp':[float(v*100) for v in ds]}

    passes=[k for k in c['conditions'] if comparisons[k+' - shallow_transformer']['pass']]
    preferred=max(passes,key=lambda k:means[k]['accuracy']) if passes else 'shallow_transformer'
    summary={'reports':reports,'means':means,'comparisons':comparisons,'interaction':interaction,'passed_candidates':passes,'preferred_candidate':preferred,'generator_cross_head_gate':comparisons['progressive_transformer - shallow_transformer']['pass'] and comparisons['progressive_mlp - shallow_mlp']['pass'],'metric_groups_verified':checks,'target_reached':{k:v['accuracy']>=.9 for k,v in means.items()},'scope':'observed source/valid; jointly trained heads; capacity and compute differ; no teacher/future data','passed_own_model_candidates':[k for k in passes if k!='rf_matched']}
    save(RUN/'artifacts/summary.json',summary)
    msg='12项新训练+3项历史基线完成；'+ '/'.join(f"{k}={means[k]['accuracy']*100:.3f}%" for k in kinds)+'；通过候选'+str(passes)+'；60组预测指标和冻结hash通过。'
    rows=['# 网站指纹生成器跨分类头结果','',msg,'','|条件|accuracy三seed均值|Macro-F1|≥90%|','|---|---:|---:|---|']
    for k in kinds:rows.append(f"|{k}|{means[k]['accuracy']*100:.3f}%|{means[k]['macro_f1']*100:.3f}%|{means[k]['accuracy']>=.9}|")
    rows+=['','|条件|seed|best步数|source accuracy|valid accuracy|valid Macro-F1|秒|','|---|---:|---:|---:|---:|---:|---:|']
    for v in reports:rows.append(f"|{v['kind']}|{v['seed']}|{v['best_step']}|{v['best']['source']['accuracy']*100:.3f}%|{v['best']['valid']['accuracy']*100:.3f}%|{v['best']['valid']['macro_f1']*100:.3f}%|{v['elapsed_seconds']:.1f}|")
    rows+=['','```json',json.dumps({'comparisons':comparisons,'interaction':interaction,'cross_head_gate':summary['generator_cross_head_gate']},indent=2,ensure_ascii=False),'```','','四组合分别联合训练，只检验生成器结构跨分类头适配，非冻结权重迁移/缺时间适配/跨数据集通用或抗漂移。RF为官方结构同配方对照，原生85.948%仅不同配方历史参考。参数与计算不匹配，不能唯一归因某单一结构因素。详见PLAN.md/SOURCES.md。']
    (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');update('completed',msg)

def main():
    c=config();verify_freeze();todo=[(k,s) for s in c['seeds'] for k in c['conditions']];active=[];done=[];start=time.monotonic()
    update('running','浅层/逐级生成器×Transformer/逐tokenMLP，另同配方RF；12新+3历史基线。GPU0最多三路且RF最多1，12800步/20次选模，source/valid，未来关闭。')
    signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(RuntimeError('supervisor terminated')))
    try:
        while todo or active:
            if time.monotonic()-start>c['pipeline_seconds']:raise TimeoutError('whole pipeline budget')
            while todo and len(active)<c['max_gpu_parallel']:
                eligible=next((i for i,(k,s) in enumerate(todo) if k!='rf_matched' or not any(key.startswith('rf_matched_') for key,p,l,t in active)),None)
                if eligible is None:break
                kind,seed=todo.pop(eligible);key=f'{kind}_{seed}';log=(RUN/'logs'/f'{key}.log').open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                p=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),'--kind',kind,'--seed',str(seed),'--device','cuda'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active.append((key,p,log,time.monotonic()))
            for row in list(active):
                key,p,log,started=row
                if p.poll() is not None:
                    log.close();active.remove(row)
                    if p.returncode:raise RuntimeError(key+' failed')
                    assert (RUN/'artifacts'/key/'report.json').exists();done.append(key)
                elif time.monotonic()-started>c['gpu_job_seconds']+120:raise TimeoutError(key+' watchdog')
            save(RUN/'artifacts/progress.json',{'status':'running','active':[{'task':k,'pid':p.pid,'device':'cuda'} for k,p,l,t in active],'completed':done,'pending':[f'{k}_{s}' for k,s in todo],'new_total':12,'historical_reused':3,'elapsed_seconds':time.monotonic()-start});time.sleep(5)
        finalize(c);save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'historical_reused':3,'elapsed_seconds':time.monotonic()-start})
    except BaseException:
        for key,p,log,t in active:stop(p);log.close()
        err=traceback.format_exc();(RUN/'logs/error.txt').write_text(err)
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n执行中止，部分产物保留：\n'+err)
        save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err})
        update('failed','生成器跨分类头实验中止，部分产物保留，见logs/error.txt；无预算/未来权限追加。');raise

if __name__=='__main__':main()
