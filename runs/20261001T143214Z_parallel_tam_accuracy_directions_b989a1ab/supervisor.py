"""One bounded GPU0 queue and independent CPU queue; no adaptive extra tasks."""
import os,sys,time,json,signal,subprocess,traceback
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from common import RUN,ROOT,config,verify_freeze,save,metric

PREFIX='当前TAM并行方向实验：'

def update(state,summary):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
    p=ROOT/'STATUS.md';lines=[v for v in p.read_text().splitlines() if not v.startswith(PREFIX)]
    lines.insert(1,PREFIX+'`'+RUN.name+'`。'+summary);p.write_text('\n'.join(lines)+'\n')

def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def finalize(c):
    verify_freeze();torch.set_num_threads(2)
    raw=torch.load(RUN/'artifacts/prepared.pt',map_location='cpu',weights_only=False)
    kinds=['baseline']+c['conditions'];reports=[];verified=0
    for seed in c['seeds']:
        base=RUN/'artifacts'/f'baseline_{seed}'
        refstate=torch.load(base/'initial_state.pt',weights_only=True,map_location='cpu')
        refstream=torch.load(base/'index_stream.pt',weights_only=True,map_location='cpu')
        for kind in kinds:
            out=RUN/'artifacts'/f'{kind}_{seed}';rep=json.loads((out/'report.json').read_text())
            assert rep['steps_completed']==c['optimizer_steps'] and rep['validation_opportunities']==len(c['eval_steps'])
            assert rep['verified_roles']==c['roles'] and rep['inactive_parameters_unchanged']
            if kind!='baseline':
                assert rep['generator_parameter_delta_l2']>0
                expected=c['parameter_counts'][kind]
                assert rep['parameters_total']==expected['total'] and rep['parameters_trainable']==expected['trainable']
            history=json.loads((out/'history.json').read_text())
            assert [v['step'] for v in history]==c['eval_steps']
            assert rep['best_step']==max(history,key=lambda v:v['valid']['accuracy'])['step']
            stream=torch.load(out/'index_stream.pt',weights_only=True,map_location='cpu');assert torch.equal(stream,refstream)
            state=torch.load(out/'initial_state.pt',weights_only=True,map_location='cpu')
            for k,v in refstate.items():
                if kind=='cpu_temporal_cnn' and k.startswith('head.'):continue
                assert torch.equal(state[k],v),(kind,k)
            for ck in ['best','last']:
                with np.load(out/f'predictions_{ck}.npz') as f:
                    for role in c['roles']:
                        y=raw[role]['labels'].numpy();pred=f[role]
                        assert pred.shape==y.shape and np.isin(pred,np.arange(102)).all()
                        assert abs(accuracy_score(y,pred)-rep[ck][role]['accuracy'])<1e-12
                        assert abs(f1_score(y,pred,labels=np.arange(102),average='macro',zero_division=0)-rep[ck][role]['macro_f1'])<1e-12
                        verified+=1
            reports.append(rep)
    means={kind:{m:float(np.mean([v['best']['valid'][m] for v in reports if v['kind']==kind])) for m in ['accuracy','macro_f1']} for kind in kinds}
    def score(kind,seed,metric):return next(v for v in reports if v['kind']==kind and v['seed']==seed)['best']['valid'][metric]
    comparisons={}
    for kind in c['gpu_conditions']+['cpu_temporal_cnn']:
        baseline='cpu_baseline' if kind=='cpu_temporal_cnn' else 'baseline'
        ds=[score(kind,s,'accuracy')-score(baseline,s,'accuracy') for s in c['seeds']]
        fs=[score(kind,s,'macro_f1')-score(baseline,s,'macro_f1') for s in c['seeds']]
        comparisons[kind+' - '+baseline]={'accuracy_mean_delta_pp':float(np.mean(ds)*100),'accuracy_per_seed_pp':[float(v*100) for v in ds],'macro_f1_mean_delta_pp':float(np.mean(fs)*100),'pass':bool(np.mean(ds)>=c['decision_gate']['accuracy_gain'] and all(v>0 for v in ds) and np.mean(fs)>=0)}
    device_delta={m:{'per_seed_pp':[float((score('cpu_baseline',s,m)-score('baseline',s,m))*100) for s in c['seeds']],'mean_pp':float((means['cpu_baseline'][m]-means['baseline'][m])*100)} for m in ['accuracy','macro_f1']}
    passes=[v.split(' - ')[0] for v,r in comparisons.items() if r['pass']]
    preferred=max(passes,key=lambda k:means[k]['accuracy']) if passes else 'baseline'
    summary={'reports':reports,'means':means,'comparisons':comparisons,'cpu_minus_gpu_baseline':device_delta,'preferred_candidate':preferred,'passed_candidates':passes,'metric_groups_verified':verified,'target_reached':{k:means[k]['accuracy']>=.9 for k in kinds},'scope':'observed source/valid development; seeds are optimization repeats, not independent datasets; no drift evaluation'}
    save(RUN/'artifacts/summary.json',summary)
    text='24项新训练+3项历史基线完成；通过候选='+str(passes)+'；最高accuracy='+f"{max(v['accuracy'] for v in means.values())*100:.3f}%"+'；108组指标复算及冻结hash通过；未来关闭。'
    rows=['# TAM并行方向实验结果','',text,'','|条件|设备|accuracy三seed均值|Macro-F1|≥90%|','|---|---|---:|---:|---|']
    for kind in kinds:rows.append(f"|{kind}|{'CPU' if kind in c['cpu_conditions'] else 'GPU0'}|{means[kind]['accuracy']*100:.3f}%|{means[kind]['macro_f1']*100:.3f}%|{means[kind]['accuracy']>=.9}|")
    rows+=['','|条件|seed|best步数|source accuracy|valid accuracy|valid Macro-F1|秒|','|---|---:|---:|---:|---:|---:|---:|']
    for rep in reports:rows.append(f"|{rep['kind']}|{rep['seed']}|{rep['best_step']}|{rep['best']['source']['accuracy']*100:.3f}%|{rep['best']['valid']['accuracy']*100:.3f}%|{rep['best']['valid']['macro_f1']*100:.3f}%|{rep['elapsed_seconds']:.1f}|")
    rows+=['','```json',json.dumps({'comparisons':comparisons,'cpu_minus_gpu_baseline':device_delta},indent=2,ensure_ascii=False),'```','','GPU候选分别对历史GPU baseline；CPU卷积候选只对本轮CPU baseline，CPU−GPU baseline只描述设备数值差异。baseline三seed历史复用并经新代码重载；其余24任务从头初始化，无历史checkpoint训练或teacher。','每候选单独改变一个设计因素，不混合候选；尺度融合/注意力/CNN改变参数与函数形式，相对时间改变表示权限内的映射，不称纯参数量匹配归因。所有候选匹配source/valid行、标签、索引流、步数与20次选择机会，算力与耗时不同。','固定valid反复参与开发，七候选并行比较增加选择偏差；开发门槛不是统计确认。三seed不是三个独立数据集，不自动追调未过候选或拼接过门槛模块。source泛化不等于抗时间漂移；没有未来日期、WTT/AWF或适应。','方向专属限制、来源、数据规则、预算见PLAN.md/config.json/SOURCES.md；完整任务曲线、预测、梯度、checkpoint与重载报告保留。']
    (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n');update('completed',text)

def main():
    c=config();verify_freeze();torch.set_num_threads(2)
    todo=[(k,s,'cuda') for s in c['seeds'] for k in c['gpu_conditions']]+[(k,s,'cpu') for s in c['seeds'] for k in c['cpu_conditions']]
    active=[];done=[];started=time.monotonic()
    update('running','六GPU单因素方向×3seed（18项）+CPU Transformer/保序卷积×3seed（6项）；GPU0最多3、CPU最多2并发。12800步/20次选模，基线3项历史复用，仅source/valid。')
    signal.signal(signal.SIGTERM,lambda signum,frame:(_ for _ in ()).throw(RuntimeError('supervisor terminated')))
    try:
        while todo or active:
            if time.monotonic()-started>c['pipeline_seconds']:raise TimeoutError('whole pipeline budget')
            for device,limit in [('cuda',c['max_gpu_parallel']),('cpu',c['max_cpu_parallel'])]:
                while sum(row['device']==device for row in active)<limit:
                    item=next((v for v in todo if v[2]==device),None)
                    if item is None:break
                    todo.remove(item);kind,seed,device=item;key=f'{kind}_{seed}'
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if device=='cuda' else '',OMP_NUM_THREADS=str(c['gpu_threads'] if device=='cuda' else c['cpu_threads']),MKL_NUM_THREADS=str(c['gpu_threads'] if device=='cuda' else c['cpu_threads']))
                    log=(RUN/'logs'/f'{key}.log').open('x')
                    proc=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),'--kind',kind,'--seed',str(seed),'--device',device],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active.append({'task':key,'device':device,'process':proc,'log':log,'started':time.monotonic()})
            for row in list(active):
                proc=row['process'];budget=c['gpu_job_seconds'] if row['device']=='cuda' else c['cpu_job_seconds']
                if proc.poll() is not None:
                    row['log'].close();active.remove(row)
                    if proc.returncode!=0:raise RuntimeError(f"{row['task']} exited {proc.returncode}")
                    assert (RUN/'artifacts'/row['task']/'report.json').exists();done.append(row['task'])
                elif time.monotonic()-row['started']>budget+120:raise TimeoutError(row['task']+' watchdog')
            save(RUN/'artifacts/progress.json',{'status':'running','active':[{'task':v['task'],'device':v['device'],'pid':v['process'].pid} for v in active],'completed':done,'pending':[f'{k}_{s}' for k,s,d in todo],'new_total':24,'historical_reused':3,'elapsed_seconds':time.monotonic()-started})
            time.sleep(5)
        finalize(c);save(RUN/'artifacts/progress.json',{'status':'completed','completed':done,'historical_reused':3,'elapsed_seconds':time.monotonic()-started})
    except BaseException:
        for row in active:stop(row['process']);row['log'].close()
        err=traceback.format_exc();(RUN/'logs/error.txt').write_text(err)
        with (RUN/'RESULTS.md').open('a') as f:f.write('\n执行中止，部分产物保留：\n'+err)
        save(RUN/'artifacts/progress.json',{'status':'failed','completed':done,'error':err,'elapsed_seconds':time.monotonic()-started})
        update('failed','本轮执行中止，部分产物保留；见logs/error.txt，无追加预算或未来访问。');raise

if __name__=='__main__':main()
