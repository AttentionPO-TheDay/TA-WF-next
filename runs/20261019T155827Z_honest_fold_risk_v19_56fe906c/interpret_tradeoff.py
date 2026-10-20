"""Post-completion interpretation, not a method or primary-gate revision.

The highlighted candidate was a preregistered secondary arm, selected for
discussion after terminal results. Reuses only frozen cached predictions.
"""
import json
from pathlib import Path
import numpy as np
from runner import RUN,C,ART,P,PA,FA,read,save
from methods import frontier,best_under_damage


def main():
    assert json.loads((ART/'audit.json').read_text())['passed']
    candidate='local_head1_uncalibrated_vote2'
    scores=json.loads((ART/'scores.json').read_text());matches=json.loads((ART/'matches.json').read_text())
    states=[s for s in json.loads((ART/'states.json').read_text()) if s['cap']==5]
    cells=[]
    for s in states:
        key=s['key'];seed,role=s['seed'],s['role']
        base=read(FA/f'features_{seed}_{role}_query.npz')['logits'].argmax(1)
        y=read(FA/f'evaluation_labels_{role}.npz')['labels'];wrong=int((base!=y).sum())
        new=next(r for r in scores if r['key']==key+'_'+candidate)
        controls={}
        for name in ['full_head1','full_local_head1']:
            old=read(ART/f'pred_{key}_{name}.npz')
            curve=frontier(y,old['predictions'],base,old['gain'])
            point=best_under_damage(curve,new['D'])
            controls[name]={'at_candidate_actual_D':point,'candidate_Recovery_minus_control_pp':100*(new['R']-point['R'])/wrong}
        cells.append({**s,'candidate_actual_D':new['D'],'candidate_actual_R':new['R'],'NAR':new['NAR'],
                      'Recovery':new['Recovery'],'controls':controls})
    summary=[]
    for role in C['roles']:
        rs=[r for r in scores if r['role']==role and r['cap']==5 and r['policy']==candidate]
        ms=[m for m in matches if m['role']==role and m['cap']==5 and m['policy']==candidate]
        cs=[r for r in cells if r['role']==role]
        summary.append({'role':role,'labels_per_trial':rs[0]['labels'],'states':len(rs),'all_actual_NAR_le_1pct':all(r['NAR']<=.01 for r in rs),
                        'mean_NAR':float(np.mean([r['NAR'] for r in rs])),'max_NAR':max(r['NAR'] for r in rs),
                        'macro_f1':float(np.mean([r['macro_f1'] for r in rs])),'delta_f1_pp':float(np.mean([r['delta_f1_pp'] for r in rs])),
                        'Recovery':float(np.mean([r['Recovery'] for r in rs])),'Recovery_retention':float(np.mean([r['Recovery_retention'] for r in rs])),
                        'same_head_at1_D_Recovery_gain_pp':float(np.mean([m['same_damage_Recovery_gain_pp'] for m in ms])),
                        'at_candidate_actual_D_Recovery_gain_vs_head_pp':float(np.mean([r['controls']['full_head1']['candidate_Recovery_minus_control_pp'] for r in cs])),
                        'at_candidate_actual_D_Recovery_gain_vs_local_pp':float(np.mean([r['controls']['full_local_head1']['candidate_Recovery_minus_control_pp'] for r in cs])),
                        'nonnegative_model_seeds_vs_head':int(sum(np.mean([m['same_damage_Recovery_gain_pp'] for m in ms if m['seed']==seed])>=0 for seed in C['model_seeds']))})
    report={'qualification':'post-completion interpretation of preregistered secondary arm; no model/threshold/gate changes; development only',
            'candidate':candidate,'summary':summary,'cells':cells}
    save('exploratory_candidate.json',report)
    path=RUN/'RESULTS.md'
    marker='\n## 完成后解释：固定局部路由与两票通过\n'
    text=path.read_text();text=text.split(marker)[0]
    lines=[marker,'这是预先列出的次级方法，在终点评价后选作讨论，未替换主臂；v18/v19主判决均保持False。以下不增加训练、标签或阈值选择。','',
           '|日期|每次标签数|F1 %|ΔF1 pp|NAR均值 %|NAR最大 %|Recovery %|原head纠错保留 %|同实际D较head纠错差 pp|同实际D较full-local纠错差 pp|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['role']}|{r['labels_per_trial']}|{100*r['macro_f1']:.3f}|{r['delta_f1_pp']:+.3f}|{100*r['mean_NAR']:.3f}|{100*r['max_NAR']:.3f}|{100*r['Recovery']:.3f}|{100*r['Recovery_retention']:.1f}|{r['at_candidate_actual_D_Recovery_gain_vs_head_pp']:+.3f}|{r['at_candidate_actual_D_Recovery_gain_vs_local_pp']:+.3f}|")
    lines+=['','该候选在18个模型/抽样状态的直接输出上均观察到NAR<1%，不使用query定位阈值；这支持一个值得冻结再验证的操作点。不是未来域风险保证，也不是新的独立确认。','',
            '同实际D比较的控制阈值由query标签定位，只是诊断对照。若相对full-local前沿仍下降，则本轮证明的是更保守且可直接执行的折模型投票，不能据此宣称超过最强几何路由的整体Pareto前沿。','',
            'v18保护项对结果影响很小，优先保留简单head；v19额外零损伤阈值和三票全同意会大量丢失纠错，主方案因此失败。当前证据不支持继续加严拒绝。三个模型只是共享冻结特征后的三个小头，不需要三个骨干前向。','',
            '少数类只有一个支持样本时，一个折的训练近邻库会缺该类；三票局部同意将无法纠正到这个类，两票仍有可能。该结构事实不能单独解释全部纠错损失。','']
    path.write_text(text+'\n'.join(lines))
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
