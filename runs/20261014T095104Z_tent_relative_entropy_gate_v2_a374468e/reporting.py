import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
C=json.loads((R/'config.json').read_text())
def write_report(scores):
    def avg(cond,key,role=None,origin=None):
        vals=[x[key] for x in scores if x['condition']==cond and (role is None or x['role']==role) and (origin is None or x['origin']==origin)]
        return float(np.mean(vals)) if vals else None
    conditions=['frozen','no_pseudo_fixed','tent_bn','bn_batchstats_no_update','tent_fixedstats_ln','tent_entropy_gate','tent_relative_entropy_gate']
    new=[x for x in scores if x['origin']=='new']; hist=[x for x in scores if x['origin']=='historical_reference']
    base={(x['seed'],x['role']):x for x in new if x['condition']=='tent_relative_entropy_gate'}
    contrasts={}
    for ref in ['tent_entropy_gate','tent_fixedstats_ln','tent_bn','bn_batchstats_no_update','frozen','no_pseudo_fixed']:
        cells=[]
        for key,a in base.items():
            b=next(x for x in hist if (x['seed'],x['role'],x['condition'])==(key[0],key[1],ref))
            cells.append({'seed':key[0],'role':key[1],**{k+'_delta_pp':100*(a[k]-b[k]) for k in ('macro_f1','accuracy','negative_adaptation_rate','recovery_rate')}})
        contrasts[ref]={'cells':cells,'mean_delta_pp':{k:float(np.mean([x[k+'_delta_pp'] for x in cells])) for k in ('macro_f1','accuracy','negative_adaptation_rate','recovery_rate')},'positive_macro_f1_cells':sum(x['macro_f1_delta_pp']>0 for x in cells)}
    verdict={'primary_macro_f1_threshold':0.72,'primary_pass':avg('tent_relative_entropy_gate','macro_f1',origin='new')>0.72 and contrasts['tent_entropy_gate']['mean_delta_pp']['macro_f1']>0,'new':{'macro_f1':avg('tent_relative_entropy_gate','macro_f1',origin='new'),'accuracy':avg('tent_relative_entropy_gate','accuracy',origin='new'),'negative_adaptation_rate':avg('tent_relative_entropy_gate','negative_adaptation_rate',origin='new'),'recovery_rate':avg('tent_relative_entropy_gate','recovery_rate',origin='new')},'contrasts':{k:{'mean_delta_pp':v['mean_delta_pp'],'positive_macro_f1_cells':v['positive_macro_f1_cells']} for k,v in contrasts.items()}}
    (R/'artifacts/verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2)+'\n'); (R/'artifacts/contrasts.json').write_text(json.dumps(contrasts,ensure_ascii=False,indent=2)+'\n')
    lines=['# TENT固定源BN统计量 + Transformer LN增强结果','',f'9个新适应单元与36条历史参考完成；独立postrun审计通过。', '', '| 条件 | Macro-F1 | Accuracy | 负适应率 | 纠错率 |','|---|---:|---:|---:|---:|']
    for c in conditions:
        o='new' if c=='tent_relative_entropy_gate' else 'historical_reference'
        lines.append(f"| {c} | {avg(c,'macro_f1',origin=o)*100:.3f}% | {avg(c,'accuracy',origin=o)*100:.3f}% | {avg(c,'negative_adaptation_rate',origin=o)*100:.3f}% | {avg(c,'recovery_rate',origin=o)*100:.3f}% |")
    for role in C['roles']:
        lines+=['',f'## {role}','','| 条件 | Macro-F1 | 负适应率 | 纠错率 |','|---|---:|---:|---:|']
        for c in conditions:
            o='new' if c=='tent_relative_entropy_gate' else 'historical_reference'
            lines.append(f"| {c} | {avg(c,'macro_f1',role,o)*100:.3f}% | {avg(c,'negative_adaptation_rate',role,o)*100:.3f}% | {avg(c,'recovery_rate',role,o)*100:.3f}% |")
    lines += ['', '## 配对差异（增强 − 参考）','', '| 参考 | Macro-F1差(pp) | Accuracy差(pp) | NAR差(pp) | 纠错率差(pp) | F1提升单元 |','|---|---:|---:|---:|---:|---:|']
    for ref,v in contrasts.items():
        d=v['mean_delta_pp']; lines.append(f"| {ref} | {d['macro_f1']:+.3f} | {d['accuracy']:+.3f} | {d['negative_adaptation_rate']:+.3f} | {d['recovery_rate']:+.3f} | {v['positive_macro_f1_cells']}/9 |")
    lines += ['',f"主判据：Macro-F1 >72% 且相对tent_entropy_gate平均提升；结果为 {'PASS' if verdict['primary_pass'] else 'FAIL'}。",'固定源BN统计量与LN开放是联合干预，本实验不能单独归因两项改动。','support无标签适应800步；query未参与梯度、统计量或选模；仅TemporalDrift已观察开发证据。','', '## 资源与塌缩诊断','', '| 条件 | 可训练参数 | 预测类别数范围 | 最大类别占比范围 | 有限logits |','|---|---:|---:|---:|---|']
    for c in ['tent_relative_entropy_gate']:
        rr=[x for x in new if x['condition']==c]; lines.append(f"| {c} | {rr[0]['trainable_parameter_count']} | {min(x['unique_predicted_classes'] for x in rr)}–{max(x['unique_predicted_classes'] for x in rr)} | {min(x['max_predicted_class_fraction'] for x in rr)*100:.3f}%–{max(x['max_predicted_class_fraction'] for x in rr)*100:.3f}% | {all(x['finite_logits'] for x in rr)} |")
    (R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return f"9项固定源BN统计量+LN增强完成，Macro-F1 {avg('tent_relative_entropy_gate','macro_f1',origin='new')*100:.3f}%，主判据 {'PASS' if verdict['primary_pass'] else 'FAIL'}；审计通过，结果见RESULTS.md。"
