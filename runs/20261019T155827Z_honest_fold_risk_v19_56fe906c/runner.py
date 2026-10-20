"""Fixed-route held-out calibration; deployment keeps the same fold models."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
import torch
from methods import (predict,local_scores,policies,select_policy,apply_choice,metrics,
                     frontier,best_under_damage,least_damage_for_recovery)

RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
C=json.loads((RUN/'config.json').read_text());ART=RUN/'artifacts'
P=ROOT/C['parent_run'];PA=P/'artifacts';F=ROOT/C['feature_run'];FA=F/'artifacts'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def save(name,value):
    p=ART/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)


def npz(path,**values):
    t=path.with_suffix(path.suffix+'.tmp')
    with t.open('wb') as f:np.savez_compressed(f,**values)
    t.replace(path)


def registry(state,summary):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
    import fcntl
    with (ROOT/'.status.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);p=ROOT/'STATUS.md';lines=p.read_text().splitlines()
        prefix=f'固定折风险校准 v19：`{RUN.name}`。';lines=[l for l in lines if not l.startswith(prefix)];lines.insert(1,prefix+summary)
        for i,l in enumerate(lines):
            if l.startswith('当前研究动作：'):lines[i]='当前研究动作：'+prefix+summary+' WTT/AWF关闭，旧封存不变。'
        p.write_text('\n'.join(lines)+'\n')


def verify():
    assert C['status']=='frozen' and C['id']==RUN.name
    for path,h in json.loads((ART/'freeze.json').read_text()).items():assert sha(ROOT/path)==h,path


def freeze():
    assert not (ART/'freeze.json').exists()
    assert json.loads((PA/'audit.json').read_text())['passed'] and json.loads((FA/'audit.json').read_text())['passed']
    paths=list(RUN.glob('*.py'))+[RUN/n for n in ['config.json','PLAN.md','SOURCES.md']]
    paths += [ROOT/C['dataset_config'],P/'config.json',PA/'audit.json',PA/'delivery_manifest.json',FA/'audit.json',FA/'delivery_manifest.json']
    manifests={P:json.loads((PA/'delivery_manifest.json').read_text()),F:json.loads((FA/'delivery_manifest.json').read_text())}
    states=[];reused=[]
    for seed in C['model_seeds']:
        reused.append(F/('artifacts/source_features_'+str(seed)+'.npz'))
        for role in C['roles']:
            reused.extend([FA/f'features_{seed}_{role}_query.npz',FA/f'features_{seed}_{role}_reference.npz',FA/f'evaluation_labels_{role}.npz'])
            for draw in C['support_seeds']:
                for cap in C['caps']:
                    key=f'{seed}_{role}_{draw}_cap{cap}'
                    states.append({'key':key,'seed':seed,'role':role,'draw':draw,'cap':cap})
                    reused += [FA/f'support_{role}_{draw}_cap{cap}.npz',PA/f'oof_{key}.npz']
                    for fold in range(3):
                        for model in C['models']:reused.append(P/'checkpoints'/f'oof_{key}_fold{fold}_{model}.npz')
                    for name in ['raw_head1','local_head1']:reused.append(PA/f'pred_{key}_{name}.npz')
    for p in set(reused):
        origin=P if p.is_relative_to(P) else F
        assert sha(p)==manifests[origin][str(p.relative_to(origin))],p
    paths+=list(set(reused));save('freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    save('states.json',states);save('preflight.json',{'passed':True,'parent_audits_and_hashes_verified':True,'new_query_evaluation':False})
    registry('running','固定折模型与三路由已冻结；同预算648个held-out校准器启动，不full-refit、不新梯度训练。')


def calibration_from_oof(state):
    oof=read(PA/f"oof_{state['key']}.npz")
    support=read(FA/f"support_{state['role']}_{state['draw']}_cap{state['cap']}.npz")
    assert np.array_equal(oof['labels'],support['labels']) and np.array_equal(oof['rows'],support['rows'])
    calibration={}
    for fold in range(3):
        hold=np.flatnonzero(oof['folds']==fold)
        outputs=list(policies(oof['original'][hold],{n:oof[n][hold] for n in C['models']},oof['local'][hold],C['routes'],C['prototype_logit_scale']))
        calibration[str(fold)]={}
        for output in outputs:
            calibration[str(fold)][output[0]]=select_policy(oof['labels'][hold],oof['original'][hold],[output],C['calibration'])
    return calibration


def calibrate():
    states=json.loads((ART/'states.json').read_text())
    for i,state in enumerate(states):
        save(f"calibration_{state['key']}.json",calibration_from_oof(state))
        save('progress.json',{'phase':'support_only_calibration','completed':i+1,'total':36})
    save('calibration_freeze.json',{p.name:sha(p) for p in sorted(ART.glob('calibration_*cap*.json'))})
    assert len(json.loads((ART/'calibration_freeze.json').read_text()))==36
    save('calibration_complete.json',{'passed':True,'calibrators':648,'full_refit':False,'query_used':False,'manifest_sha256':sha(ART/'calibration_freeze.json')})
    registry('running','648个固定路由held-out阈值已冻结；相同折模型直接推断并投票，进入726状态终点评价。')


def verify_calibration():
    complete=json.loads((ART/'calibration_complete.json').read_text());assert complete['passed']
    assert sha(ART/'calibration_freeze.json')==complete['manifest_sha256']
    for name,h in json.loads((ART/'calibration_freeze.json').read_text()).items():assert sha(ART/name)==h


def fold_outputs(state):
    seed,role,draw,cap=(state[k] for k in ['seed','role','draw','cap']);key=state['key']
    source=read(FA/f'source_features_{seed}.npz');ref=read(FA/f'features_{seed}_{role}_reference.npz')
    support=read(FA/f'support_{role}_{draw}_cap{cap}.npz');oof=read(PA/f'oof_{key}.npz')
    query=read(FA/f'features_{seed}_{role}_query.npz');base=query['logits'].argmax(1)
    x=(query['features']-source['mu'])/source['sd'];sx=(ref['features'][support['reference_positions']]-source['mu'])/source['sd']
    result=[]
    for fold in range(3):
        train=np.flatnonzero(oof['folds']!=fold)
        logits={n:predict(read(P/'checkpoints'/f'oof_{key}_fold{fold}_{n}.npz'),x) for n in C['models']}
        local=local_scores(x,sx[train],support['labels'][train])
        result.append(list(policies(query['logits'],logits,local,C['routes'],C['prototype_logit_scale'])))
    return query['rows'],base,result


def voted(base,proposals,gains,k):
    ps=np.stack(proposals);gs=np.stack(gains)
    counts=np.zeros((len(base),102),dtype=np.int8);i=np.arange(len(base))
    for p in ps:counts[i,p]+=1
    candidate=counts.argmax(1)
    accepted=(candidate!=base)&(counts[i,candidate]>=k)
    out=np.where(accepted,candidate,base)
    support=(ps==candidate[None,:])
    gain=np.where(accepted,(gs*support).sum(0)/np.maximum(support.sum(0),1),0).astype(np.float32)
    return out,gain


def outputs_from_folds(state,base,folds):
    calibration=json.loads((ART/f"calibration_{state['key']}.json").read_text())
    all_outputs=[];details=[]
    for route in [r['name'] for r in C['routes']]:
        for mode in C['modes']:
            ps,gs=[],[]
            for fold in range(3):
                output=next(o for o in folds[fold] if o[0]==route)
                if mode=='uncalibrated':p,g=output[1:]
                else:p,g=apply_choice(calibration[str(fold)][route][mode],[output],base)
                ps.append(p);gs.append(g)
                if mode!='uncalibrated':details.append({'route':route,'mode':mode,'fold':fold,'choice':calibration[str(fold)][route][mode],'prediction':p})
            for k in C['votes']:
                p,g=voted(base,ps,gs,k)
                all_outputs.append((f'{route}_{mode}_vote{k}',p,g))
    for name,old in [('full_head1','raw_head1'),('full_local_head1','local_head1')]:
        cached=read(PA/f"pred_{state['key']}_{old}.npz")
        all_outputs.append((name,cached['predictions'].astype(np.int64),cached['gain']))
    return all_outputs,details


def score():
    verify_calibration();states=json.loads((ART/'states.json').read_text());scores=[];matches=[];transfer=[]
    for state in states:
        seed,role,draw,cap=(state[k] for k in ['seed','role','draw','cap']);key=state['key']
        rows,base,folds=fold_outputs(state);outputs,details=outputs_from_folds(state,base,folds)
        labels=read(FA/f'evaluation_labels_{role}.npz');assert np.array_equal(rows,labels['rows']);y=labels['labels']
        correct=int((base==y).sum());wrong=len(y)-correct;basef1=metrics(y,base,base)['macro_f1']
        records=[];curves={};at1={}
        if draw==C['support_seeds'][0] and cap==3:
            records.append({'key':f'{seed}_{role}_frozen','seed':seed,'role':role,'draw':None,'cap':0,'labels':0,'policy':'frozen',**metrics(y,base,base),'delta_f1_pp':0.,'Recovery_retention':None})
        for name,p,g in outputs:
            curve=frontier(y,p,base,g);curves[name]=curve;at1[name]=best_under_damage(curve,int(correct*.01))
            row={**state,'key':key+'_'+name,'policy':name,'labels':len(read(FA/f'support_{role}_{draw}_cap{cap}.npz')['labels']),**metrics(y,p,base)}
            row['delta_f1_pp']=100*(row['macro_f1']-basef1);row['diagnostic_Recovery_at_NAR1']=at1[name]['R']/wrong
            records.append(row);npz(ART/f'pred_{key}_{name}.npz',predictions=p.astype(np.int16),gain=g)
        full=next(r for r in records if r['policy']=='full_head1');anchor=at1['full_head1']
        for row in records:
            if row['policy']=='frozen':continue
            name=row['policy'];row['Recovery_retention']=row['Recovery']/full['Recovery']
            same=best_under_damage(curves[name],anchor['D']);same_r=least_damage_for_recovery(curves[name],anchor['R'])
            matches.append({k:row[k] for k in ['key','seed','role','draw','cap','policy']})
            matches[-1].update({'anchor_head_at1':anchor,'same_damage':same,'same_damage_Recovery_gain_pp':100*(same['R']-anchor['R'])/wrong,'same_recovery':same_r,'same_recovery_feasible_at_NAR1':bool(same_r is not None and same_r['D']<=correct*.01)})
        for d in details:
            p=d.pop('prediction');transfer.append({**state,**d,'query_metrics':metrics(y,p,base)})
        scores.extend(records);save('scores.json',scores);save('matches.json',matches);save('fold_risk_transfer.json',transfer)
        save('progress.json',{'phase':'terminal_scoring','states':len(scores),'total':726,'last':key})
        if len(matches)%180==0:print(json.dumps({'scored':len(scores),'last':key}),flush=True)
    assert len(scores)==726 and len(matches)==720 and len(transfer)==648
    save('scores_complete.json',{'passed':True,'states':len(scores),'query_fitted':False})


def summarize():
    scores=json.loads((ART/'scores.json').read_text());matches=json.loads((ART/'matches.json').read_text())
    names=[f"{r['name']}_{m}_vote{k}" for r in C['routes'] for m in C['modes'] for k in C['votes']]+['full_head1','full_local_head1']
    summary=[]
    for role in C['roles']:
        for cap in C['caps']:
            for name in names:
                rs=[r for r in scores if r['role']==role and r['cap']==cap and r['policy']==name]
                ms=[r for r in matches if r['role']==role and r['cap']==cap and r['policy']==name]
                summary.append({'role':role,'cap':cap,'policy':name,'states':len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in ['macro_f1','delta_f1_pp','NAR','Recovery','Recovery_retention','diagnostic_Recovery_at_NAR1']},
                                'max_NAR':max(r['NAR'] for r in rs),'all_NAR_le_1pct':all(r['NAR']<=.01 for r in rs),
                                'same_damage_Recovery_gain_pp':float(np.mean([r['same_damage_Recovery_gain_pp'] for r in ms]))})
    save('summary.json',summary)
    name=f"{C['primary']['route']}_{C['primary']['mode']}_vote{C['primary']['votes']}";gates=[]
    for role in C['roles']:
        s=next(r for r in summary if r['role']==role and r['cap']==5 and r['policy']==name)
        ms=[r for r in matches if r['role']==role and r['cap']==5 and r['policy']==name]
        nonneg=sum(np.mean([r['same_damage_Recovery_gain_pp'] for r in ms if r['seed']==seed])>=0 for seed in C['model_seeds'])
        gates.append({'role':role,'all9_NAR_le_1pct':s['all_NAR_le_1pct'],'delta_f1_pp':s['delta_f1_pp'],'Recovery_retention':s['Recovery_retention'],
                      'same_damage_Recovery_gain_pp':s['same_damage_Recovery_gain_pp'],'nonnegative_model_seeds':int(nonneg),
                      'passed':bool(s['all_NAR_le_1pct'] and s['delta_f1_pp']>=1 and s['Recovery_retention']>=.5 and s['same_damage_Recovery_gain_pp']>=0 and nonneg>=2)})
    decision={'primary':C['primary'],'passed':all(g['passed'] for g in gates),'roles':gates,'risk_certified':False,'deployed':False}
    save('decision.json',decision)
    lines=['# 固定折模型校准与投票 v19','','全部预定校准和终点评价完成，待审计。仅已观察TemporalDrift开发；受v18反馈影响，无独立确认主张。','','同v17/v18标签预算，cap3=303/305、cap5=496/504；不新增梯度训练或GPU任务。648个固定路由阈值、726评价状态。对应模型未用自己校准折标签训练；选择后不full-refit。折间仍有依赖，经验低NAR不是风险认证。','','## 主判决','','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','','## 完整操作点','','|日期|预算|固定策略|F1 %|ΔF1 pp|NAR均值 %|NAR最大 %|Recovery %|纠错保留 %|同D纠错差 pp|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['role']}|cap{r['cap']}|{r['policy']}|{100*r['macro_f1']:.3f}|{r['delta_f1_pp']:+.3f}|{100*r['NAR']:.3f}|{100*r['max_NAR']:.3f}|{100*r['Recovery']:.3f}|{100*r['Recovery_retention']:.1f}|{r['same_damage_Recovery_gain_pp']:+.3f}|")
    lines+=['','2票与3票同时报告，uncalibrated投票是同折模型/结构对照；相同D诊断前沿只用于解释，不能部署query阈值。所有均值共享同一query，不能按726次独立重复解释。最差状态比平均NAR更能暴露不稳定性。','',
            '主判决不因次级方法更好改写。不据终点评价追加阈值/投票调整。不同于v18的设计同时固定了路由并取消full-refit，若收益改善，仍不能单独将其归因于refit这一因素。WTT/AWF关闭，旧结果不改。','']
    (RUN/'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','calibrate','score','summarize']);a=parser.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if a.stage!='freeze':verify()
    globals()[a.stage]()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save('failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});registry('failed',f'执行失败：{type(exc).__name__}，保留证据。');raise
