import json,csv
import numpy as np
from runtime import RUN,ROOT,C,save,sha,verify,views,state
from scoring import fit,scores
from metrics import curve,audit_direct,preflight

def run():
    verify();save('metric_preflight.json',preflight());cal=fit();m=json.loads((RUN/'artifacts/freeze.json').read_text());p=RUN/'artifacts/source_calibration.json';m[str(p.relative_to(ROOT))]=sha(p);save('freeze.json',m);verify();calhash=sha(p)
    old=json.loads((ROOT/C['v11_run']/'artifacts/cells.json').read_text());cells=[];curves={};checks=0
    for partition in ['valid','query']:
        # Source-valid is shared across target slices; it is not an independent repeat.
        data=[(None,)+views('valid')] if partition=='valid' else [(n,)+views('query',n) for n in C['slices']]
        for data_name,z,rows,y in data:
            for name in C['slices'] if partition=='valid' else [data_name]:
                for seed in C['seeds']:
                    s=state(seed,name,partition);assert np.array_equal(rows,s['rows']) and np.array_equal(y,s['labels']);ss=scores(z,s,cal,seed,name,partition);key=f'{partition}_{seed}_{name}'
                    np.savez_compressed(RUN/'artifacts'/f'scores_{key}.npz',**ss,labels=y,base=s['base'],candidates=s['candidate'],rows=rows)
                    for score,values in ss.items():
                        met,cr,pred=curve(y,s['base'],s['candidate'],values);audit_direct(y,s['base'],s['candidate'],values,met);row=dict(seed=seed,slice=name,partition=partition,score=score,**met)
                        if score in ['v9_margin','v11_min']:
                            match=next(r for r in old if r['seed']==seed and r['slice']==name and r['partition']==partition and r['arm']=='direction_iat' and r['pool']=='v9_smooth' and r['score']==('baseline' if score=='v9_margin' else 'joint_support'));assert abs(match['rec_at_nar1']-met['rec_at_nar1'])<1e-12;checks+=1
                        cells.append(row);curves[f'{key}_{score}']=cr
                    save('progress.json',dict(phase='score',cells=len(cells),total=96));print(json.dumps(dict(partition=partition,slice=name,seed=seed,cells=len(cells))),flush=True)
    assert len(cells)==96 and checks==24 and sha(RUN/'artifacts/source_calibration.json')==calhash
    save('cells.json',cells);np.savez_compressed(RUN/'artifacts/curves.npz',**curves);save('score_audit.json',dict(passed=True,states=96,source_calibration_frozen_before_query=True,calibration_sha256=calhash,historical_frontiers_exact=checks,new_training=0,new_inference=0))

def audit():
    verify();cal=json.loads((RUN/'artifacts/source_calibration.json').read_text());original=json.loads((RUN/'artifacts/cells.json').read_text());count=0
    for partition in ['valid','query']:
        data=[(None,)+views('valid')] if partition=='valid' else [(n,)+views('query',n) for n in C['slices']]
        for data_name,z,rows,y in data:
            for name in C['slices'] if partition=='valid' else [data_name]:
                for seed in C['seeds']:
                    s=state(seed,name,partition);ss=scores(z,s,cal,seed,name,partition)
                    with np.load(RUN/'artifacts'/f'scores_{partition}_{seed}_{name}.npz') as a:
                        assert np.array_equal(a['rows'],rows) and np.array_equal(a['base'],s['base']) and np.array_equal(a['candidates'],s['candidate']) and np.array_equal(a['labels'],y)
                        for score,x in ss.items():
                            assert np.array_equal(a[score],x);met,_,_=curve(y,s['base'],s['candidate'],x);audit_direct(y,s['base'],s['candidate'],x,met);old=next(r for r in original if r['seed']==seed and r['slice']==name and r['partition']==partition and r['score']==score)
                            for k,v in met.items():
                                if v is None:assert old[k] is None
                                else:assert abs(v-old[k])<1e-12,(k,v,old[k])
                            count+=1
    assert count==96;save('reload_audit.json',dict(passed=True,score_and_metric_states=count,candidate_identity_exact=True,partial_roc_and_operational_thresholds_recomputed=True,calibration_unchanged=True))

def report():
    cells=json.loads((RUN/'artifacts/cells.json').read_text());summary=[]
    for partition in ['valid','query']:
        for name in C['slices']:
            for score in C['scores']:
                rr=[r for r in cells if r['partition']==partition and r['slice']==name and r['score']==score];assert len(rr)==3;out=dict(partition=partition,slice=name,score=score)
                for k in ['pauc_raw','pauc_norm','step_mean_recovery','rec_at_nar1','attained_nar1','diag_f1','other_wrong_accepted','other_wrong_retention','other_per_correction','corrected','damaged','precision']:
                    vals=[r[k] for r in rr];out[k+'_mean']=float(np.mean(vals)) if all(v is not None for v in vals) else None;out[k+'_sd']=float(np.std(vals,ddof=1)) if all(v is not None for v in vals) else None
                summary.append(out)
    decisions=[]
    for name in C['slices']:
        def row(score):return next(r for r in summary if r['partition']=='query' and r['slice']==name and r['score']==score)
        b,p=row('v9_margin'),row(C['primary']);gain=p['pauc_norm_mean']-b['pauc_norm_mean'];rec=100*(p['rec_at_nar1_mean']-b['rec_at_nar1_mean']);step=100*(p['step_mean_recovery_mean']-b['step_mean_recovery_mean']);ratio=p['other_per_correction_mean']<=b['other_per_correction_mean'];decisions.append(dict(slice=name,pauc_norm_gain=gain,rec1_gain_pp=rec,step_mean_recovery_gain_pp=step,invalid_per_correction_not_worse=bool(ratio),passed=bool(gain>=.01 and rec>=.2 and step>=0 and ratio)))
    decision=dict(primary=C['primary'],passed=all(r['passed'] for r in decisions),slices=decisions,candidate_pool='v9_smooth_unchanged',global_auroc_used_for_selection=False,integrated=False)
    save('summary.json',summary);save('registered_decisions.json',decision)
    for n,data in [('cells',cells),('summary',summary)]:
        fields=list(dict.fromkeys(k for r in data for k in r))
        with (RUN/'artifacts'/f'{n}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    lines=['# 固定v9候选的低NAR软重排序 v12','',f'主判据：{decision["passed"]}。零训练、零新前向，只改变固定v9候选的排序。','', 'FPR以全部frozen正确样本为分母，等于候选阈值NAR。partial ROC正例只含v9可真正纠正的候选，其他错误改写另记；pAUC归一化值为raw area/.01的低风险平均TPR，非McClish标准化。确定性面积用最大可行Recovery左阶梯，不拆同分。','', '|切片|评分|pAUC/.01 %|确定性平均Recovery %|Rec@NAR1 %|无效改写数|每纠错无效改写|','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['partition']!='query':continue
        f=lambda k,m=1:f'{r[k]*m:.3f}' if r[k] is not None else 'NA'
        lines.append('|'+ '|'.join([r['slice'],r['score'],f('pauc_norm_mean',100),f('step_mean_recovery_mean',100),f('rec_at_nar1_mean',100),f('other_wrong_accepted_mean'),f('other_per_correction_mean')])+'|')
    (RUN/'table_main.md').write_text('\n'.join(lines[6:])+'\n');lines+=['','主判决（只margin_plus_soft，不用次评分替换）：','', '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','', '全部96个valid/query逐seed结果与标准差见cells/summary CSV。sourcevalid在两目标切片中重复用于诊断，不是独立重复。候选生成完全不变，raw v9全接纳性能不变；NAR1阈值使用query标签作终端诊断，不能部署或当作风险保证。没有将其他错误改写算成修复，完整计数见cells.csv。','', '15个视图相关，源NLL权重只表示源域支持度，不能解释为独立证据连乘或真实误改概率。soft加权均值并非完整贝叶斯模型。源valid曾用于checkpoint/温度与本轮校准，不能称未见确认。main/aux源均值SD在查询评分前冻结，没有目标拟合、阈值迁移或超参搜索。v10/v11辅助模型成本与推理缓存成本仍存在，本轮零新计算不代表完整方法零开销。','', '本轮只在已观察Version045/USA开发切片评价，不声称物理丢包率已知；标签对应沿用内容支持假设而非官方映射。WTT/AWF关闭，旧结果封存不改。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    with np.load(RUN/'artifacts/curves.npz') as a:
        fig,axes=plt.subplots(1,2,figsize=(11,4));grid=np.linspace(0,.01,201)
        for ax,name in zip(axes,C['slices']):
            for score in ['v9_margin','v11_min','weighted_soft','margin_plus_soft','shuffled_margin_soft']:
                rs=[]
                for seed in C['seeds']:
                    cr=a[f'query_{seed}_{name}_{score}'];last=np.flatnonzero(np.r_[cr[1:,0]!=cr[:-1,0],True]);x=cr[last,0];r=cr[last,1];idx=np.searchsorted(x,grid,side='right')-1;rs.append(r[np.maximum(idx,0)])
                ax.step(grid*100,np.mean(rs,0)*100,where='post',label=score)
            ax.set_title(name);ax.set_xlim(0,1);ax.set_xlabel('NAR budget (%)');ax.set_ylabel('Recovery (%, deterministic diagnostic)');ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(RUN/f'figure_low_nar.{ext}',dpi=170)
        plt.close(fig)

if __name__=='__main__':
    import sys
    {'run':run,'audit':audit,'report':report}[sys.argv[1]]()
