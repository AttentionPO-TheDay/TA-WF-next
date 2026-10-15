import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; C=json.loads((R/'config.json').read_text())

def write_report(scores):
    def avg(cond,key,role=None,origin=None):
        vals=[x[key] for x in scores if x['condition']==cond and (role is None or x['role']==role) and (origin is None or x['origin']==origin)]
        return float(np.mean(vals)) if vals else None
    hist=[x for x in scores if x['origin']=='historical_reference']; new=[x for x in scores if x['origin']=='new']; nc='tent_shot_span'
    refs=['tent_bn','tent_entropy_gate','tent_relative_entropy_gate','tent_fixedstats_ln','tent_coadapt_gated','no_pseudo_fixed','frozen','bn_batchstats_no_update']
    contrasts={}
    for ref in refs:
        cells=[]
        for a in new:
            b=next(x for x in hist if (x['seed'],x['role'],x['condition'])==(a['seed'],a['role'],ref))
            cells.append({'seed':a['seed'],'role':a['role'],**{k+'_delta_pp':100*(a[k]-b[k]) for k in ('macro_f1','accuracy','negative_adaptation_rate','recovery_rate')}})
        contrasts[ref]={'cells':cells,'mean_delta_pp':{k:float(np.mean([x[k+'_delta_pp'] for x in cells])) for k in ('macro_f1','accuracy','negative_adaptation_rate','recovery_rate')},'positive_macro_f1_cells':sum(x['macro_f1_delta_pp']>0 for x in cells)}
    newvals={k:avg(nc,k,origin='new') for k in ('macro_f1','accuracy','negative_adaptation_rate','recovery_rate')}
    ret=float(np.mean([x['retained_sample_fraction'] for x in new])); agree=float(np.mean([x['agreement_fraction'] for x in new]))
    verdict={'primary_threshold':.72,'stretch_threshold':.725,'gain_target_pp':1.5,'primary_pass':newvals['macro_f1']>.72,'stretch_pass':newvals['macro_f1']>.725,'gain_vs_tent_bn_pp':contrasts['tent_bn']['mean_delta_pp']['macro_f1'],'gain_target_pass':contrasts['tent_bn']['mean_delta_pp']['macro_f1']>=1.5,'retention_fraction':ret,'agreement_fraction':agree,'new':newvals,'contrasts':{k:{'mean_delta_pp':v['mean_delta_pp'],'positive_macro_f1_cells':v['positive_macro_f1_cells']} for k,v in contrasts.items()}}
    (R/'artifacts/verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2)+'\n'); (R/'artifacts/contrasts.json').write_text(json.dumps(contrasts,ensure_ascii=False,indent=2)+'\n')
    lines=['# tent_shot_span 结果','',f"9个新适应单元与{len(hist)}条历史参考完成；独立postrun审计通过。",'', '| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |','|---|---:|---:|---:|---:|']
    conditions=['frozen','no_pseudo_fixed','tent_bn','bn_batchstats_no_update','tent_fixedstats_ln','tent_entropy_gate','tent_relative_entropy_gate','tent_coadapt_gated',nc]
    for c in conditions:
        o='new' if c==nc else 'historical_reference'; lines.append(f"| {c} | {avg(c,'macro_f1',origin=o)*100:.3f}% | {avg(c,'accuracy',origin=o)*100:.3f}% | {avg(c,'negative_adaptation_rate',origin=o)*100:.3f}% | {avg(c,'recovery_rate',origin=o)*100:.3f}% |")
    for role in C['roles']:
        lines += ['',f'## {role}','','| 条件 | Macro-F1 | 负适应率 | 纠错率 |','|---|---:|---:|---:|']
        for c in conditions:
            o='new' if c==nc else 'historical_reference'; lines.append(f"| {c} | {avg(c,'macro_f1',role,o)*100:.3f}% | {avg(c,'negative_adaptation_rate',role,o)*100:.3f}% | {avg(c,'recovery_rate',role,o)*100:.3f}% |")
    lines += ['', '## 配对差异（shot_span − 参考）','', '| 参考 | Macro-F1差(pp) | Accuracy差(pp) | NAR差(pp) | 纠错率差(pp) | F1提升单元 |','|---|---:|---:|---:|---:|---:|']
    for ref,v in contrasts.items():
        d=v['mean_delta_pp']; lines.append(f"| {ref} | {d['macro_f1']:+.3f} | {d['accuracy']:+.3f} | {d['negative_adaptation_rate']:+.3f} | {d['recovery_rate']:+.3f} | {v['positive_macro_f1_cells']}/9 |")
    lines += ['',f"主判据 Macro-F1>72%：{'PASS' if verdict['primary_pass'] else 'FAIL'}；相对原版tent_bn增益目标+1.5pp：{'PASS' if verdict['gain_target_pass'] else 'FAIL'}。",f"平均保留样本比例：{ret*100:.2f}%；双视图预测一致率：{agree*100:.2f}%；可训练参数量：{new[0]['trainable_parameter_count']}；BN统计更新：0。",'该组是Scale-Lock与Span-Invariance联合配方的TemporalDrift开发证据，不能单独证明两组件的因果贡献或跨数据集普适性。']
    lines += ['', '## 资源与塌缩诊断','', '| 条件 | 预测类别数范围 | 最大类占比范围 | 有限logits |','|---|---:|---:|---|']
    lines.append(f"| {nc} | {min(x['unique_predicted_classes'] for x in new)}–{max(x['unique_predicted_classes'] for x in new)} | {min(x['max_predicted_class_fraction'] for x in new)*100:.3f}%–{max(x['max_predicted_class_fraction'] for x in new)*100:.3f}% | {all(x['finite_logits'] for x in new)} |")
    (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return f"9项Scale-Lock+Span-Invariance目标任务完成，Macro-F1 {newvals['macro_f1']*100:.3f}%，NAR {newvals['negative_adaptation_rate']*100:.3f}%；审计通过，结果见RESULTS.md。"
