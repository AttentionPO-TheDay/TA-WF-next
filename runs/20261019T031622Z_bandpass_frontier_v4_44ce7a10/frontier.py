"""Terminal-label diagnostics. No thresholds from this file enter inference/training."""
import csv,json
from collections import defaultdict
import numpy as np
from runtime import RUN,ROOT,C,save,metrics,sha
from worker import CONTROLS

SIGNALS=['combined']+C['acceptance_v4']['other_diagnostic_scores']
def factor(s):return 'Version' if s.startswith('version') else 'Network' if s.startswith('network') else 'Behavior'
def aggregate(rows,key):
    factors=defaultdict(list)
    for r in rows:factors[factor(r['slice'])].append(r[key])
    return float(np.mean([np.mean(v) for v in factors.values()]))

def writecsv(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (RUN/'artifacts'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def curve(cells,signal):
    # A single threshold per method/seed, shared across all included slices.
    nfactor=len({factor(v['slice']) for v in cells});counts={f:sum(factor(v['slice'])==f for v in cells) for f in {factor(v['slice']) for v in cells}}
    arrays=[];oracle=0.;raw_nar=0.;raw_rec=0.;eligible_rec=0.;eligible_nar=0.
    for v in cells:
        y,bp,cp,sc=v['y'],v['bp'],v['cp'],v['scores'][signal];correct=bp==y;wrong=~correct;change=cp!=bp
        wr=1/(nfactor*counts[factor(v['slice'])]);cn=max(correct.sum(),1);wn=max(wrong.sum(),1)
        damage=change&correct;recovered=change&wrong&(cp==y)
        raw_nar+=wr*damage.sum()/cn;raw_rec+=wr*recovered.sum()/wn;oracle+=wr*recovered.sum()/wn
        eligible=change & (sc>=0) if signal=='combined' else change
        eligible_rec+=wr*(recovered&eligible).sum()/wn;eligible_nar+=wr*(damage&eligible).sum()/cn
        arrays.append((sc[eligible],wr*damage[eligible]/cn,wr*recovered[eligible]/wn))
    s=np.concatenate([v[0] for v in arrays]);d=np.concatenate([v[1] for v in arrays]);r=np.concatenate([v[2] for v in arrays])
    if len(s):
        order=np.argsort(-s,kind='stable');s=s[order];d=d[order];r=r[order];ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
        nar=np.r_[0.,np.cumsum(d)[ends]];rec=np.r_[0.,np.cumsum(r)[ends]];threshold=np.r_[np.inf,s[ends]]
    else:nar=np.array([0.]);rec=np.array([0.]);threshold=np.array([np.inf])
    assert np.all(np.diff(nar)>=-1e-12) and np.all(np.diff(rec)>=-1e-12)
    result={'raw_nar':raw_nar,'raw_recovery':raw_rec,'candidate_oracle_recovery':oracle,'eligible_recovery':eligible_rec,'eligible_nar':eligible_nar,'eligible_changes':len(s)}
    for cap in [.01,.02]:
        ix=np.flatnonzero(nar<=cap+1e-12)[-1];k=str(int(cap*100));result['rec_at_nar'+k]=float(rec[ix]);result['attained_nar'+k]=float(nar[ix]);result['diagnostic_threshold'+k]=float(threshold[ix]) if np.isfinite(threshold[ix]) else None
        result['random_rec_at_nar'+k]=eligible_rec*min(1,cap/eligible_nar) if eligible_nar>0 else eligible_rec
    for target in [.02,.04]:
        ix=np.flatnonzero(rec>=target-1e-12);result['nar_at_rec'+str(int(target*100))]=float(nar[ix[0]]) if len(ix) else None
    return result,(nar,rec,threshold)

def load_cells(method,seed):
    out=[];old=ROOT/C['baseline_run']
    for name in C['slices']:
        with np.load(old/'artifacts'/f'base_{seed}_{name}.npz') as b:bp=b['predictions'];y=b['labels'];ids=b['rows']
        with np.load(RUN/'artifacts'/f'raw_{seed}_{name}_{method}.npz') as b:cp=b['predictions'];assert np.array_equal(ids,b['rows']) and np.array_equal(y,b['labels'])
        with np.load(RUN/'artifacts'/f'signal_{seed}_{name}_{method}.npz') as s:scores={k:s[k] for k in SIGNALS}
        out.append({'slice':name,'bp':bp,'cp':cp,'y':y,'scores':scores})
    return out

def analyze(phase):
    methods=CONTROLS.copy()
    if phase!='controls':methods+=C['conditions']
    trigger_path=RUN/'artifacts/shuffle_trigger.json'
    if phase=='final' and trigger_path.exists():
        t=json.loads(trigger_path.read_text())
        if t['enabled']:methods+=['shuffle_'+t['arm']]
    out=[];factorrows=[];curves={}
    for method in methods:
        for seed in C['seeds']:
            cells=load_cells(method,seed)
            for signal in SIGNALS:
                row,cv=curve(cells,signal);out.append({'condition':method,'seed':seed,'signal':signal,**row});curves[(method,seed,signal)]=cv
                np.savez_compressed(RUN/'artifacts'/f'curve_{phase}_{seed}_{method}_{signal}.npz',nar=cv[0],recovery=cv[1],threshold=cv[2])
                if signal=='combined':
                    for f in ['Version','Network','Behavior']:
                        fr,_=curve([v for v in cells if factor(v['slice'])==f],signal);factorrows.append({'condition':method,'seed':seed,'factor':f,**fr})
    writecsv(f'frontier_{phase}_cells.csv',out);writecsv(f'frontier_{phase}_factors.csv',factorrows);save(f'frontier_{phase}_cells.json',out)
    summary=[]
    for method in methods:
        for signal in SIGNALS:
            rows=[r for r in out if r['condition']==method and r['signal']==signal];rr={'condition':method,'signal':signal}
            for key in ['raw_nar','raw_recovery','candidate_oracle_recovery','eligible_recovery','rec_at_nar1','rec_at_nar2','random_rec_at_nar1','random_rec_at_nar2','nar_at_rec2','nar_at_rec4']:
                a=[r[key] for r in rows];rr[key+'_mean']=float(np.mean(a)) if all(v is not None for v in a) else None;rr[key+'_sd']=float(np.std(a,ddof=1)) if all(v is not None for v in a) else None
            summary.append(rr)
    save(f'frontier_{phase}_summary.json',summary);writecsv(f'frontier_{phase}_summary.csv',summary)
    if phase=='controls':return summary
    decisions=[];base={r['seed']:r for r in out if r['condition']=='ungated_lora' and r['signal']=='combined'}
    for method in C['conditions']:
        rs=[r for r in out if r['condition']==method and r['signal']=='combined'];diff=[100*(r['rec_at_nar1']-base[r['seed']]['rec_at_nar1']) for r in rs]
        fd={}
        for f in ['Version','Network','Behavior']:
            aa=[r['rec_at_nar1'] for r in factorrows if r['condition']==method and r['factor']==f];bb=[r['rec_at_nar1'] for r in factorrows if r['condition']=='ungated_lora' and r['factor']==f];fd[f]=100*(float(np.mean(aa))-float(np.mean(bb)))
        simple=np.mean(diff)>=.5 and all(v>0 for v in diff);passed=bool(simple and sum(v>0 for v in fd.values())>=2 and min(fd.values())>=-.5)
        decisions.append({'condition':method,'primary':method=='soft_geometry','delta_rec_at_nar1_pp':float(np.mean(diff)),'per_seed_delta_pp':diff,'factor_deltas_pp':fd,'passed':passed,'shuffle_eligible':bool(simple and method in ['soft_band','soft_geometry'])})
    save('registered_decisions.json',decisions)
    if phase=='adapted':
        eligible=[r for r in decisions if r['shuffle_eligible']];chosen=max(eligible,key=lambda r:(r['delta_rec_at_nar1_pp'],r['condition']=='soft_geometry')) if eligible else None
        save('shuffle_trigger.json',{'enabled':chosen is not None,'arm':chosen['condition'] if chosen else None,'diagnostic_feedback_used':True,'rule_frozen_in_config':True,'decisions':decisions})
    return summary

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    allrows=[]
    for mode in ['controls','adapted','shuffle']:
        for seed in C['seeds']:
            p=RUN/'artifacts'/f'{mode}_scores_{seed}.json'
            if p.exists():allrows+=json.loads(p.read_text())
    save('scores.json',allrows);summary=[];methods=list(dict.fromkeys(r['condition'] for r in allrows))
    for method in methods:
        for policy in C['policies']:
            row={'condition':method,'policy':policy}
            for key in ['macro_f1','accuracy','nar','recovery','delta_f1_pp']:
                v=[aggregate([r for r in allrows if r['condition']==method and r['policy']==policy and r['seed']==seed],key) for seed in C['seeds']];scale=1 if key=='delta_f1_pp' else 100
                row[key+'_mean']=float(np.mean(v))*scale;row[key+'_sd']=float(np.std(v,ddof=1))*scale
            summary.append(row)
    save('summary.json',summary);writecsv('summary.csv',summary)
    fs=json.loads((RUN/'artifacts/frontier_final_summary.json').read_text());lines=['# 原始与源校准输出','', '| Method | Policy | ΔF1 pp | NAR % | Recovery % |','|---|---|---:|---:|---:|']
    for r in summary:lines.append('| '+r['condition']+' | '+r['policy']+' | '+' | '.join(f"{r[k+'_mean']:.3f}±{r[k+'_sd']:.3f}" for k in ['delta_f1_pp','nar','recovery'])+' |')
    (RUN/'table_operating.md').write_text('\n'.join(lines)+'\n')
    lines=['# 诊断安全—纠错前沿','', '每个seed内各切片共用一个分数阈值，三因素等权。阈值利用query标签事后定位，不可作为部署安全保证。','', '| Method | Recovery@NAR≤1% % | Recovery@NAR≤2% % | Random@1% % | Candidate oracle % |','|---|---:|---:|---:|---:|']
    for r in fs:
        if r['signal']=='combined':lines.append('| '+r['condition']+' | '+' | '.join(f"{100*r[k+'_mean']:.3f}±{100*r[k+'_sd']:.3f}" for k in ['rec_at_nar1','rec_at_nar2','random_rec_at_nar1','candidate_oracle_recovery'])+' |')
    (RUN/'table_frontier.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));grid=np.linspace(0,.1,301)
    for method in methods:
        ys=[]
        for seed in C['seeds']:
            with np.load(RUN/'artifacts'/f'curve_final_{seed}_{method}_combined.npz') as a:
                x,y=a['nar'],a['recovery'];ux=np.unique(x);uy=np.array([y[x==v].max() for v in ux]);idx=np.searchsorted(ux,grid,side='right')-1;yy=uy[np.maximum(idx,0)];ys.append(yy)
        yy=np.array(ys)*100;axes[0].plot(grid*100,yy.mean(0),label=method);axes[0].fill_between(grid*100,yy.min(0),yy.max(0),alpha=.1)
        r=next(r for r in fs if r['condition']==method and r['signal']=='combined');nar=r['raw_nar_mean'];rec=r['raw_recovery_mean'];axes[1].plot(grid*100,np.minimum(1,grid/max(nar,1e-12))*rec*100,label=method)
    for ax,title in zip(axes,['Observed score-threshold frontier (seed range)','Expected uniform random acceptance']):
        ax.set(xlabel='Negative adaptation rate (%)',ylabel='Recovery rate (%)',title=title,xlim=(0,10));ax.axvline(1,color='gray',ls=':');ax.axvline(2,color='gray',ls=':');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_frontier.{ext}',dpi=180)
    plt.close(fig)
    decisions=json.loads((RUN/'artifacts/registered_decisions.json').read_text());passed=next(r for r in decisions if r['primary'])['passed'];trigger=json.loads((RUN/'artifacts/shuffle_trigger.json').read_text())
    text=['# 安全—纠错前沿 v4 结果','',f'主判据：{"PASS（诊断开发信号）" if passed else "FAIL"}。完整判据及逐seed/因素结果见artifacts/registered_decisions.json。', '',f'已完成{len(allrows)}个评分状态；新增适应{90+(30 if trigger["enabled"] else 0)}次，历史对照120个原始状态复用。打乱对照触发：{trigger["enabled"]}。', '', '[运行点主表](table_operating.md)；[诊断前沿主表](table_frontier.md)；[风险—纠错图](figure_frontier.png)。','', '解释边界：前沿使用目标标签事后选择共同阈值，是诊断可达结果；source_nar1/source_nar2才是按源验证校准、未用目标标签选择的输出点，其目标NAR仍须实测。全部回退不算成功。曲线不将不同slice分别挑最好阈值再平均。', '', '所有数据已观察，本轮为机制开发，不是独立确认。source valid已参与底座选模。三seed重复共享query，不当作独立样本；无显著性宣称。Version/Network/Behavior沿用既有诊断身份与物理解释限制。WTT/AWF未新增评分。', '', 'B/C/D固定目标损失按批大小归一化，关系项使用加权类中心并乘mean(w)，source identity不变；报告权重和梯度供判断步长衰减因素。直接重排的输出分数使用构造的log(q)，并非另一个训练模型。', '', '若触发打乱，基于本轮诊断反馈选定arm，只作权重对应性的归因；不视为独立确认。无路由对照复用v3，源前向、架构、损失及梯度一致性由preflight核对。']
    (RUN/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print((RUN/'table_frontier.md').read_text(),flush=True);print((RUN/'table_operating.md').read_text(),flush=True)

if __name__=='__main__':
    import sys
    phase=sys.argv[1];analyze(phase)
    if phase=='final':report()
