"""v18 explicit cache reuse, support-only fitting/selection, terminal query."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from methods import (unit, predict, metrics, fit_linear, proto, local_scores, policies,
                     select_policy, apply_choice, frontier, best_under_damage, least_damage_for_recovery)

RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[1]
C=json.loads((RUN/'config.json').read_text())
ART=RUN/'artifacts'; CK=RUN/'checkpoints'
P=ROOT/C['parent_run']; PA=P/'artifacts'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    p=ART/name; t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)


def read(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def npz(path,**values):
    t=path.with_suffix(path.suffix+'.tmp')
    with t.open('wb') as f:np.savez_compressed(f,**values)
    t.replace(path)


def state_key(seed,role,draw,cap):return f'{seed}_{role}_{draw}_cap{cap}'


def registry(state,summary):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
    import fcntl
    with (ROOT/'.status.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=ROOT/'STATUS.md';lines=p.read_text().splitlines();prefix=f'n-shot低NAR v18：`{RUN.name}`。'
        lines=[l for l in lines if not l.startswith(prefix)];lines.insert(1,prefix+summary)
        for i,l in enumerate(lines):
            if l.startswith('当前研究动作：'):lines[i]='当前研究动作：'+prefix+summary+' WTT/AWF关闭，旧实验封存不变。'
        p.write_text('\n'.join(lines)+'\n')


def verify():
    assert C['status']=='frozen' and C['id']==RUN.name
    for path,digest in json.loads((ART/'freeze.json').read_text()).items():assert sha(ROOT/path)==digest,path


def freeze():
    assert not (ART/'freeze.json').exists()
    assert json.loads((PA/'audit.json').read_text())['passed']
    paths=list(RUN.glob('*.py'))+[RUN/n for n in ['config.json','PLAN.md','SOURCES.md']]
    paths += [ROOT/C['dataset_config'],P/'config.json',PA/'audit.json',PA/'scores.json',PA/'delivery_manifest.json']
    manifest=json.loads((PA/'delivery_manifest.json').read_text())
    reused=[]
    for seed in C['model_seeds']:
        reused.append(PA/f'source_features_{seed}.npz')
        for role in C['roles']:
            for kind in ['reference','query']:reused.append(PA/f'features_{seed}_{role}_{kind}.npz')
            reused.append(PA/f'pred_{seed}_{role}_frozen.npz')
            for draw in C['support_seeds']:
                for cap in C['caps']:
                    key=state_key(seed,role,draw,cap)
                    for arm in ['head_l2','shrink_proto']:
                        reused += [P/'checkpoints'/f'{key}_{arm}.npz',PA/f'pred_{key}_{arm}.npz']
    for role in C['roles']:
        reused.append(PA/f'evaluation_labels_{role}.npz')
        for draw in C['support_seeds']:
            for cap in C['caps']:reused.append(PA/f'support_{role}_{draw}_cap{cap}.npz')
    for p in set(reused):assert sha(p)==manifest[str(p.relative_to(P))],p
    paths+=list(set(reused))
    save('freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    preflight=[]
    for seed in C['model_seeds']:
        source=read(PA/f'source_features_{seed}.npz')
        x=(source['features']-source['mu'])/source['sd']
        w0=source['weight']*source['sd'][None,:]; b0=source['bias']+source['weight']@source['mu']
        np.testing.assert_allclose(x@w0.T+b0,source['logits'],atol=.001,rtol=1e-5)
        preflight.append({'seed':seed,'source_head_equivalence':True})
    save('preflight.json',{'passed':True,'source_checks':preflight,'parent_hashes_verified':True,'new_query_label_metrics':False})
    registry('running','代码/权限/预算冻结；复用v17特征，CPU支持集三折拟合和风险选择启动，query未参与选择。')


def load_support(seed,role,draw,cap):
    source=read(PA/f'source_features_{seed}.npz')
    reference=read(PA/f'features_{seed}_{role}_reference.npz')
    support=read(PA/f'support_{role}_{draw}_cap{cap}.npz')
    i=support['reference_positions']
    assert np.array_equal(reference['rows'][i],support['rows'])
    x=(reference['features'][i]-source['mu'])/source['sd']
    original=reference['logits'][i]
    w0=source['weight']*source['sd'][None,:];b0=source['bias']+source['weight']@source['mu']
    return source,support,x,original,w0,b0


def train():
    CK.mkdir(exist_ok=True)
    states=[];count=0
    for seed in C['model_seeds']:
        for role in C['roles']:
            for draw in C['support_seeds']:
                for cap in C['caps']:
                    key=state_key(seed,role,draw,cap)
                    source,support,x,original,w0,b0=load_support(seed,role,draw,cap)
                    y=support['labels']; folds=np.empty(len(y),dtype=np.int64)
                    for cls in np.unique(y):
                        i=np.flatnonzero(y==cls);folds[i]=(cls+np.arange(len(i)))%C['folds']
                    models={}; histories={}
                    for name,old in [('head1','head_l2'),('shrink','shrink_proto')]:
                        path=CK/f'{key}_{name}.npz';shutil.copy2(P/'checkpoints'/f'{key}_{old}.npz',path)
                        models[name]=read(path)
                    for name in ['head8','protect']:
                        model,history=fit_linear(x,y,w0,b0,name,C['linear']);count+=1
                        npz(CK/f'{key}_{name}.npz',**model);histories[name]=history
                    oof={name:np.zeros((len(y),102),dtype=np.float32) for name in C['base_models']}
                    local=np.zeros_like(original)
                    for fold in range(C['folds']):
                        train=np.flatnonzero(folds!=fold);hold=np.flatnonzero(folds==fold)
                        assert not set(train)&set(hold) and len(train) and len(hold)
                        local[hold]=local_scores(x[hold],x[train],y[train])
                        for name in C['base_models']:
                            if name=='shrink':model=proto(x[train],y[train],source['means'],C['prototype_pseudocount'])
                            else:
                                model,history=fit_linear(x[train],y[train],w0,b0,name,C['linear']);count+=1
                                histories[f'fold{fold}_{name}']=history
                            npz(CK/f'oof_{key}_fold{fold}_{name}.npz',**model)
                            z=predict(model,x[hold])
                            oof[name][hold]=z
                    outputs=list(policies(original,oof,local,C['policies'],C['prototype_logit_scale']))
                    selection=select_policy(y,original,outputs,C['selectors'])
                    npz(ART/f'oof_{key}.npz',labels=y,rows=support['rows'],folds=folds,original=original,local=local,**oof)
                    save(f'selection_{key}.json',selection)
                    save(f'history_{key}.json',histories)
                    states.append({'key':key,'seed':seed,'role':role,'draw':draw,'cap':cap,'labels':len(y),
                                   'strict_nshot':bool((support['class_counts']==cap).all()),'source_support_sha256':sha(PA/f'support_{role}_{draw}_cap{cap}.npz')})
                    save('states.json',states)
                    save('progress.json',{'phase':'support_only_training','support_states':len(states),'total':54,'new_heads':count,'last':key})
                    print(json.dumps({'fit_state':len(states),'new_heads':count,'key':key}),flush=True)
    assert len(states)==54 and count==C['budget']['new_supervised_heads']
    hashes={str(p.relative_to(RUN)):sha(p) for p in list(CK.glob('*.npz'))+list(ART.glob('oof_*.npz'))+list(ART.glob('selection_*.json'))}
    save('fit_freeze.json',hashes)
    save('fits_complete.json',{'passed':True,'new_heads':count,'support_states':len(states),'query_used':False,'manifest_sha256':sha(ART/'fit_freeze.json')})
    registry('running','594个新监督头、54个支持集OOF与风险路由已冻结；进入1410状态query终点评价。')


def verify_fits():
    complete=json.loads((ART/'fits_complete.json').read_text());assert complete['passed']
    assert sha(ART/'fit_freeze.json')==complete['manifest_sha256']
    for path,digest in json.loads((ART/'fit_freeze.json').read_text()).items():assert sha(RUN/path)==digest,path


def query_outputs(state):
    seed,role,draw,cap=(state[k] for k in ['seed','role','draw','cap']);key=state['key']
    source,support,sx,_,_,_=load_support(seed,role,draw,cap)
    query=read(PA/f'features_{seed}_{role}_query.npz')
    x=(query['features']-source['mu'])/source['sd']
    logits={name:predict(read(CK/f'{key}_{name}.npz'),x) for name in C['base_models']}
    local=local_scores(x,sx,support['labels'])
    outputs=list(policies(query['logits'],logits,local,C['policies'],C['prototype_logit_scale']))
    selections=json.loads((ART/f'selection_{key}.json').read_text())
    base=query['logits'].argmax(1)
    for name,choice in selections.items():
        p,g=apply_choice(choice,outputs,base);outputs.append((name,p,g))
    return query['rows'],base,outputs


def score():
    verify_fits()
    states=json.loads((ART/'states.json').read_text());scores=[];matches=[]
    for state in states:
        key=state['key'];role=state['role'];seed=state['seed']
        rows,base,outputs=query_outputs(state)
        labels=read(PA/f'evaluation_labels_{role}.npz');y=labels['labels'];assert np.array_equal(rows,labels['rows'])
        correct=int((base==y).sum());wrong=len(y)-correct
        basef1=metrics(y,base,base)['macro_f1'];curves={};at1={};records=[]
        if state['draw']==C['support_seeds'][0] and state['cap']==1:
            records.append({'key':f'{seed}_{role}_frozen','seed':seed,'role':role,'draw':None,'cap':0,'policy':'frozen','labels':0,**metrics(y,base,base),'delta_f1_pp':0.,'Recovery_retention':None})
        for name,p,g in outputs:
            curve=frontier(y,p,base,g);curves[name]=curve;at1[name]=best_under_damage(curve,int(correct*.01))
            row={k:state[k] for k in ['seed','role','draw','cap','labels']}
            row.update({'key':key+'_'+name,'policy':name,**metrics(y,p,base)})
            row['delta_f1_pp']=100*(row['macro_f1']-basef1);row['diagnostic_Recovery_at_NAR1']=at1[name]['R']/wrong
            records.append(row)
            npz(ART/f'pred_{key}_{name}.npz',predictions=p.astype(np.int16),gain=g)
            if name in ['raw_head1','raw_shrink']:
                old='head_l2' if name=='raw_head1' else 'shrink_proto'
                assert np.array_equal(p,read(PA/f'pred_{key}_{old}.npz')['predictions'])
        fullhead=next(r for r in records if r['policy']=='raw_head1')
        anchor=at1['raw_head1']
        for row in records:
            if row['policy']=='frozen':continue
            name=row['policy'];row['Recovery_retention']=row['Recovery']/fullhead['Recovery'] if fullhead['Recovery'] else None
            same=best_under_damage(curves[name],anchor['D']);matched=least_damage_for_recovery(curves[name],anchor['R'])
            matches.append({'key':row['key'],'seed':seed,'role':role,'draw':state['draw'],'cap':state['cap'],'policy':name,
                            'anchor_head_at1':anchor,'same_damage':same,'same_damage_Recovery_gain_pp':100*(same['R']-anchor['R'])/wrong,
                            'same_recovery':matched,'same_recovery_feasible_at_NAR1':bool(matched is not None and matched['D']<=correct*.01)})
        scores.extend(records);save('scores.json',scores);save('matches.json',matches)
        save('progress.json',{'phase':'terminal_scoring','states':len(scores),'total':1410,'last':key})
        if len(matches)%234==0:print(json.dumps({'score_states':len(scores),'last':key}),flush=True)
    assert len(scores)==1410 and len(matches)==1404
    save('scores_complete.json',{'passed':True,'states':len(scores),'query_selected_thresholds_deployed':False})


def summarize():
    scores=json.loads((ART/'scores.json').read_text());matches=json.loads((ART/'matches.json').read_text())
    summary=[]
    for role in C['roles']:
        for cap in C['caps']:
            for name in [p['name'] for p in C['policies']]+list(C['selectors']):
                rs=[r for r in scores if r['role']==role and r['cap']==cap and r['policy']==name]
                ms=[m for m in matches if m['role']==role and m['cap']==cap and m['policy']==name]
                summary.append({'role':role,'cap':cap,'policy':name,'states':len(rs),
                    **{k:float(np.mean([r[k] for r in rs])) for k in ['macro_f1','delta_f1_pp','NAR','Recovery','Recovery_retention','diagnostic_Recovery_at_NAR1']},
                    'max_NAR':max(r['NAR'] for r in rs),'all_NAR_le_1pct':all(r['NAR']<=.01 for r in rs),
                    'same_damage_Recovery_gain_pp':float(np.mean([m['same_damage_Recovery_gain_pp'] for m in ms])),
                    'same_R_feasible_states':sum(m['same_recovery_feasible_at_NAR1'] for m in ms)})
    save('summary.json',summary)
    gates=[]
    for role in C['roles']:
        s=next(r for r in summary if r['role']==role and r['cap']==5 and r['policy']==C['primary']['policy'])
        ms=[m for m in matches if m['role']==role and m['cap']==5 and m['policy']==C['primary']['policy']]
        nonneg=sum(np.mean([m['same_damage_Recovery_gain_pp'] for m in ms if m['seed']==seed])>=0 for seed in C['model_seeds'])
        gates.append({'role':role,'all9_NAR_le_1pct':s['all_NAR_le_1pct'],'mean_delta_f1_pp':s['delta_f1_pp'],
                      'mean_Recovery_retention':s['Recovery_retention'],'same_damage_Recovery_gain_pp':s['same_damage_Recovery_gain_pp'],
                      'nonnegative_model_seeds':int(nonneg),'passed':bool(s['all_NAR_le_1pct'] and s['delta_f1_pp']>=1 and s['Recovery_retention']>=.5 and s['same_damage_Recovery_gain_pp']>=0 and nonneg>=2)})
    decision={'primary':C['primary'],'passed':all(g['passed'] for g in gates),'roles':gates,'deployed':False,'risk_certified':False}
    save('decision.json',decision)
    lines=['# n-shot低NAR v18','', '全部预定支持集拟合与终点评价完成，待终端审计。方法/输出策略在query评分前冻结；以下都是已观察TemporalDrift开发证据，不是独立确认。', '',
           '源骨干不变，复用v17冻结特征和相同支持集，无新GPU任务。严格1shot每日期102标签；cap3=303/305、cap5=496/504，不是完整3/5shot。支持集用于拟合和选择均计入预算，无额外校准标签。', '',
           '24固定策略+2个支持集OOF选择器，共54支持状态、594个新监督头、1410评价状态。未用query选模型/混合系数/阈值。OOF选择后full refit存在迁移，经验低损伤不是风险保证。', '',
           '## 主判决', '', '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','',
           '## 全部固定策略与选择器', '',
           '表中跨三modelseed×三support抽样取均值；最大NAR和全9状态是否达标同时保留。纠错保留率相对同支持状态raw_head1。相同D增量为query标签定位的诊断前沿，不能部署。', '',
           '|日期|预算|策略|F1 %|ΔF1 pp|NAR均值 %|NAR最大 %|Recovery %|纠错保留 %|同D纠错差 pp|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['role']}|{'strict1' if r['cap']==1 else 'cap'+str(r['cap'])}|{r['policy']}|{100*r['macro_f1']:.3f}|{r['delta_f1_pp']:+.3f}|{100*r['NAR']:.3f}|{100*r['max_NAR']:.3f}|{100*r['Recovery']:.3f}|{100*r['Recovery_retention']:.1f}|{r['same_damage_Recovery_gain_pp']:+.3f}|")
    lines+=['','## 限制与解释边界','','主判决不随某个次级策略更好而改写；若发现有潜力的固定策略，只能报告本轮探索证据，不能挑成独立确认。单独降低NAR也可能只是回退frozen造成，须同时看纠错保留、ΔF1和同损伤前沿。','',
            'OOF折的训练集合重叠；1shot时被留出类可能没有目标支持，源先验fallback保留。每类支持样本数较小，且OOF选择了多策略/阈值，不能把0次观测损伤或固定规则的名义置信上界当作选择后1%保证。最终full-support refit与OOF有差别，报告实际query NAR。模型与抽样seed共享同一query，不视为独立数据集。','',
            '原型概率融合使用固定20倍cosine，不等于经过目标独立校准。局部邻居支持来自同一冻结编码器，真标签提供新的类别锚点，但几何证据仍相关。未引入新标签、query适应、WTT/AWF或新骨干。','']
    (RUN/'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','score','summarize']);args=parser.parse_args()
    torch.set_num_threads(C['budget']['threads']);torch.use_deterministic_algorithms(True)
    if args.stage!='freeze':verify()
    globals()[args.stage]()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save('failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        registry('failed',f'执行失败：{type(exc).__name__}；部分产物保留。');raise
