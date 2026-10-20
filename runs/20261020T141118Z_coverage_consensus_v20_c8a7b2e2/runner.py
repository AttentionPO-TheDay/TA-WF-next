"""Support-only fits, then independent terminal evaluation, bounded v20."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from maintenance_base import fit_head,predict,metrics,frontier,best_under_damage
from methods import bags,local,outputs

RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
C=json.loads((RUN/'config.json').read_text());A=RUN/'artifacts';CK=RUN/'checkpoints'
F=ROOT/C['feature_run'];FA=F/'artifacts';P=ROOT/C['fold_run'];PA=P/'artifacts'
V=ROOT/C['parent_run'];VA=V/'artifacts'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def save(name,value):
    p=A/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)


def npz(path,**values):
    t=path.with_suffix(path.suffix+'.tmp')
    with t.open('wb') as f:np.savez_compressed(f,**values)
    t.replace(path)


def registry(status,summary):
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',status,'--summary',summary],check=True)
    import fcntl
    with (ROOT/'.status.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'STATUS.md';prefix=f'支持覆盖与共识路由 v20：`{RUN.name}`。'
        lines=[l for l in path.read_text().splitlines() if not l.startswith(prefix)]
        lines.insert(1,prefix+summary)
        for i,l in enumerate(lines):
            if l.startswith('当前研究动作：'):lines[i]='当前研究动作：'+prefix+summary+' WTT/AWF关闭，旧封存不变。'
        path.write_text('\n'.join(lines)+'\n')


def states():
    return [{'key':f'{s}_{r}_{d}_cap5','seed':s,'role':r,'draw':d,'cap':5}
            for s in C['model_seeds'] for r in C['roles'] for d in C['support_seeds']]


def support(state):
    s,r,d=(state[k] for k in ['seed','role','draw'])
    source=read(FA/f'source_features_{s}.npz');ref=read(FA/f'features_{s}_{r}_reference.npz')
    sup=read(FA/f'support_{r}_{d}_cap5.npz');ids=sup['reference_positions']
    assert np.array_equal(ref['rows'][ids],sup['rows'])
    x=(ref['features'][ids]-source['mu'])/source['sd']
    w=source['weight']*source['sd'][None,:];b=source['bias']+source['weight']@source['mu']
    return source,sup,x,w,b


def freeze():
    assert C['status']=='frozen' and C['id']==RUN.name and not (A/'freeze.json').exists()
    paths=list(RUN.glob('*.py'))+[RUN/n for n in ['config.json','PLAN.md','SOURCES.md']]+[ROOT/C['dataset_config']]
    manifests={}
    for origin in [F,P,V]:
        assert json.loads((origin/'artifacts/audit.json').read_text())['passed']
        manifest=origin/'artifacts/delivery_manifest.json'
        # v19's delivery contains its predictions; older input manifests cover all reused caches.
        assert manifest.exists(),manifest
        manifests[origin]=json.loads(manifest.read_text())
        paths += [origin/'config.json',origin/'artifacts/audit.json',manifest]
    reused=[]
    for st in states():
        s,r,d=(st[k] for k in ['seed','role','draw']);key=st['key']
        reused += [FA/f'source_features_{s}.npz',FA/f'features_{s}_{r}_reference.npz',FA/f'features_{s}_{r}_query.npz',FA/f'support_{r}_{d}_cap5.npz',FA/f'evaluation_labels_{r}.npz',PA/f'oof_{key}.npz']
        reused += [P/'checkpoints'/f'oof_{key}_fold{b}_head1.npz' for b in range(3)]
        reused += [PA/f'pred_{key}_{n}.npz' for n in ['raw_head1','local_head1']]
        reused += [VA/f'pred_{key}_local_head1_uncalibrated_vote2.npz']
    for p in set(reused):
        origin=next(o for o in [F,P,V] if p.is_relative_to(o))
        assert sha(p)==manifests[origin][str(p.relative_to(origin))],p
    paths+=list(set(reused));save('freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    for s in C['model_seeds']:
        z=read(FA/f'source_features_{s}.npz');x=(z['features']-z['mu'])/z['sd']
        np.testing.assert_allclose(x@(z['weight']*z['sd'][None,:]).T+z['bias']+z['weight']@z['mu'],z['logits'],atol=.001,rtol=1e-5)
    save('states.json',states());save('preflight.json',{'passed':True,'all_input_hashes_verified':True,'source_head_equivalence':True,'query_fitted':False})
    registry('running','方法/预算/输入已冻结；18支持状态、90个保类五袋小头启动，CPU2线程；query未参与拟合。')


def verify():
    assert C['status']=='frozen'
    for p,h in json.loads((A/'freeze.json').read_text()).items():assert sha(ROOT/p)==h,p


def train():
    verify();torch.set_num_threads(C['budget']['threads']);count=0
    for st in states():
        source,sup,x,w,b=support(st);y=sup['labels'];history={}
        splits=bags(y,C['bags'])
        npz(A/f"splits_{st['key']}.npz",**{f'bag{i}':ids for i,ids in enumerate(splits)},rows=sup['rows'])
        for i,ids in enumerate(splits):
            assert len(np.unique(y[ids]))==102
            model,h=fit_head(x[ids],y[ids],w,b,C['linear']);history[str(i)]=h
            npz(CK/f"{st['key']}_bag{i}.npz",**model);count+=1
        save(f"history_{st['key']}.json",history)
        save('progress.json',{'phase':'support_only_training','heads':count,'total':90,'last':st['key']})
        print(json.dumps({'fit':st['key'],'heads':count}),flush=True)
    assert count==C['budget']['new_heads']
    save('fit_freeze.json',{str(p.relative_to(RUN)):sha(p) for p in list(CK.glob('*.npz'))+list(A.glob('splits_*.npz'))})
    save('fit_complete.json',{'passed':True,'heads':count,'query_used':False,'manifest_sha256':sha(A/'fit_freeze.json')})
    registry('running','90个新头及抽样清单已冻结，进入186个终点评价状态；不按query调整票数或阈值。')


def verify_fits():
    done=json.loads((A/'fit_complete.json').read_text());assert done['passed']
    assert sha(A/'fit_freeze.json')==done['manifest_sha256']
    for p,h in json.loads((A/'fit_freeze.json').read_text()).items():assert sha(RUN/p)==h,p


def query_outputs(st):
    source,sup,sx,_,_=support(st);s,r=st['seed'],st['role'];key=st['key']
    q=read(FA/f'features_{s}_{r}_query.npz');x=(q['features']-source['mu'])/source['sd']
    oof=read(PA/f'oof_{key}.npz');bl=read(A/f'splits_{key}.npz');y=sup['labels']
    oldz=[];oldl=[];bagz=[];bagl=[]
    for i in range(3):
        ids=np.flatnonzero(oof['folds']!=i)
        oldz.append(predict(read(P/'checkpoints'/f'oof_{key}_fold{i}_head1.npz'),x))
        oldl.append(local(x,sx[ids],y[ids]))
    for i in range(C['bags']):
        ids=bl[f'bag{i}'];bagz.append(predict(read(CK/f'{key}_bag{i}.npz'),x));bagl.append(local(x,sx[ids],y[ids]))
    full=local(x,sx,y);out=outputs(q['logits'],oldz,oldl,bagz,bagl,full)
    for name,old in [('full_head','raw_head1'),('full_local','local_head1')]:
        z=read(PA/f'pred_{key}_{old}.npz');out[name]=(z['predictions'].astype(np.int64),z['gain'])
    assert set(out)==set(C['policies'])
    old=read(VA/f'pred_{key}_local_head1_uncalibrated_vote2.npz')
    assert np.array_equal(out['old_local_vote2'][0],old['predictions'])
    np.testing.assert_allclose(out['old_local_vote2'][1],old['gain'],atol=1e-6)
    return q,out,{'old_logits':np.stack(oldz),'old_local':np.stack(oldl),'bag_logits':np.stack(bagz),'bag_local':np.stack(bagl),'full_local':full}


def score():
    verify();verify_fits();records=[];transfers=[]
    for st in states():
        q,out,detail=query_outputs(st);base=q['logits'].argmax(1)
        label=read(FA/f"evaluation_labels_{st['role']}.npz");assert np.array_equal(q['rows'],label['rows'])
        y=label['labels'];assert not set(q['rows'])&set(support(st)[1]['rows'])
        bm=metrics(y,base,base);wrong=int((base!=y).sum());baseline=out['old_local_vote2'][0]
        curves={name:frontier(y,p,base,g) for name,(p,g) in out.items()}
        if st['draw']==C['support_seeds'][0]:records.append({**st,'policy':'frozen',**bm,'delta_f1_pp':0.})
        for name in C['policies']:
            p,g=out[name];m=metrics(y,p,base);row={**st,'policy':name,**m,'delta_f1_pp':100*(m['macro_f1']-bm['macro_f1'])}
            row['Recovery_retention']=m['Recovery']/metrics(y,out['full_head'][0],base)['Recovery']
            for control in ['full_head','full_local','old_local_vote2']:
                matched=best_under_damage(curves[control],m['D'])
                row[f'actualD_gain_vs_{control}_pp']=100*(m['R']-matched['R'])/wrong
            records.append(row)
            npz(A/f"pred_{st['key']}_{name}.npz",predictions=p.astype(np.int16),gain=g)
            added=(p!=base)&(p!=baseline);lost=(baseline!=base)&(p!=baseline)
            transfers.append({**st,'policy':name,'new_correct_vs_old':int((added&(p==y)).sum()),'lost_correct_vs_old':int((lost&(baseline==y)).sum()),'new_damage_vs_old':int(((base==y)&(p!=y)&(baseline==y)).sum()),'removed_damage_vs_old':int(((base==y)&(p==y)&(baseline!=y)).sum())})
        save('scores.json',records);save('transfers.json',transfers)
        save('progress.json',{'phase':'terminal_scoring','states':len(records),'total':186,'last':st['key']})
        print(json.dumps({'scored':len(records),'last':st['key']}),flush=True)
    assert len(records)==186
    save('scores_complete.json',{'passed':True,'states':len(records),'query_fitted':False})


def summarize():
    rows=json.loads((A/'scores.json').read_text());summary=[];decisions=[]
    for role in C['roles']:
        for name in C['policies']:
            rs=[r for r in rows if r['role']==role and r['policy']==name]
            summary.append({'role':role,'policy':name,'states':len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in ['macro_f1','accuracy','delta_f1_pp','NAR','Recovery','Recovery_retention','actualD_gain_vs_full_local_pp','actualD_gain_vs_old_local_vote2_pp']},'max_NAR':max(r['NAR'] for r in rs)})
        ss=next(r for r in summary if r['role']==role and r['policy']==C['primary']);old=next(r for r in summary if r['role']==role and r['policy']=='old_local_vote2')
        gain=100*(ss['Recovery']-old['Recovery'])
        seed_pass=sum(np.mean([r['actualD_gain_vs_full_local_pp'] for r in rows if r['role']==role and r['policy']==C['primary'] and r['seed']==s])>=0 for s in C['model_seeds'])
        ok=ss['max_NAR']<=.01 and gain>=.5 and ss['macro_f1']>=old['macro_f1'] and ss['actualD_gain_vs_full_local_pp']>=0 and seed_pass>=2
        decisions.append({'role':role,'all9_NAR_le_1pct':ss['max_NAR']<=.01,'Recovery_gain_vs_v19_pp':gain,'deltaF1_vs_v19_pp':ss['delta_f1_pp']-old['delta_f1_pp'],'actualD_Recovery_gain_vs_full_local_pp':ss['actualD_gain_vs_full_local_pp'],'nonnegative_model_seeds_vs_full_local':int(seed_pass),'passed':bool(ok)})
    decision={'primary':C['primary'],'passed':all(d['passed'] for d in decisions),'roles':decisions,'deployed':False,'risk_certified':False}
    save('summary.json',summary);save('decision.json',decision)
    lines=['# 支持覆盖与共识路由 v20','','全部预定拟合及终点评价完成，待审计。仅TemporalDrift已观察开发；标签和query角色沿用v17。','','90个新线性头，18支持状态，186评价状态。Day90/270每次496/504标签，所有支持均来自原reference，cap5不是严格5-shot。五袋保留每类覆盖，单样本始终训练；无校准或测试阈值选取。','','## 预设主判决','','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','','## 全部预设操作点','','|日期|策略|F1 %|ΔF1 pp|NAR均值 %|NAR最差 %|Recovery %|原头纠错保留 %|同实际D较full-local纠错差 pp|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:lines.append(f"|{s['role']}|{s['policy']}|{100*s['macro_f1']:.3f}|{s['delta_f1_pp']:+.3f}|{100*s['NAR']:.3f}|{100*s['max_NAR']:.3f}|{100*s['Recovery']:.3f}|{100*s['Recovery_retention']:.1f}|{s['actualD_gain_vs_full_local_pp']:+.3f}|")
    lines+=['','同实际D控制阈值由query标签定位，仅用于诊断，不能部署。主方法未因次级结果改写。旧三折模型/对照复用不是新重复。9状态共享日期query，折与支持抽样相关，不能当186独立试验。','', 'v20同时改变模型训练子集和类覆盖，保类五袋相对旧三折的效果不能单独归因于某一项。单次近邻路由消融保留同一批头，可隔离重复路由的影响。WTT/AWF保持关闭，未部署。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','score','summarize']);arg=parser.parse_args()
    globals()[arg.stage]()
