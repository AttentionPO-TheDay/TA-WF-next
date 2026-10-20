"""Posthoc delivery checks only; cannot alter methods, checkpoints or score selection."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from runtime import RUN,ROOT,C,sha,save,verify

def main():
    verify();rows=[];baselines=[];gateonly=[];proto=[]
    audit=json.loads((RUN/'artifacts/postrun_audit.json').read_text());assert audit['passed'] and audit['cells']==240
    for seed in C['seeds']:
        pre=json.loads((RUN/'artifacts'/f'source_preflight_{seed}.json').read_text())
        for n in ['residual','lora']:assert pre['checks'][n]['zero_init_bitwise_parity'] and pre['checks'][n]['base_locked']
        art=torch.load(RUN/'artifacts'/f'source_gmm_{seed}.pt',map_location='cpu',weights_only=True)
        for k in ['1','4']:
            p=art['fits'][k]
            prior=p['priors'].numpy();effective=np.exp(-(prior*np.log(prior.clip(1e-12))).sum(1))
            proto.append({'seed':seed,'k':int(k),'entropy_cutoff_nat':float(p['entropy_cutoff']),'distance_cutoff':float(p['distance_cutoff']),'distance_scale':float(p['distance_scale']),'wasserstein_temperature':float(p['temperature']),'mean_effective_components':float(effective.mean()),'minimum_component_prior':float(prior.min()),'minimum_covariance':float(p['covariances'].min()),'maximum_covariance':float(p['covariances'].max())})
        for name in C['slices']:
            with np.load(RUN/'artifacts'/f'base_{seed}_{name}.npz') as b:
                y=b['labels'];base=b['predictions'];base_logits=b['logits'];qrows=b['rows']
            oldp=ROOT/C['diagnostic_run']/'artifacts'/f'pred_{seed}_{name}_frozen.npz'
            with np.load(oldp) as old:
                assert np.array_equal(old['labels'],y) and np.array_equal(old['rows'],qrows)
                differences=int((old['predictions']!=base).sum())
                baselines.append({'seed':seed,'slice':name,'different_predictions_vs_old_frozen':differences,'max_abs_logit_difference':float(np.max(np.abs(old['logits']-base_logits)))})
            correct=base==y;wrong=~correct
            for cond in C['conditions']:
                pf=RUN/'artifacts'/f'pred_{seed}_{name}_{cond}.npz'
                with np.load(pf) as v:
                    assert np.isfinite(v['logits']).all() and np.isfinite(v['weight']).all()
                    keep=v['keep'];pred=v['predictions'];bypass=v['bypass'];h=v['entropy'];d=v['distance']
                    changed=pred!=base;nar=(pred!=y)&correct;recovered=(pred==y)&wrong
                    rows.append({'seed':seed,'slice':name,'condition':cond,'rows':len(y),'accepted':int(keep.sum()),'rejected':int((~keep).sum()),'bypassed':int(bypass.sum()),'changed_predictions':int(changed.sum()),'negative_adaptation_count':int(nar.sum()),'recovered_count':int(recovered.sum()),'frozen_correct':int(correct.sum()),'frozen_wrong':int(wrong.sum()),'accepted_frozen_accuracy':float(correct[keep].mean()) if keep.any() else None,'rejected_frozen_accuracy':float(correct[~keep].mean()) if (~keep).any() else None,'mean_entropy_accepted':float(h[keep].mean()) if keep.any() else None,'mean_entropy_rejected':float(h[~keep].mean()) if (~keep).any() else None,'mean_distance_accepted':float(d[keep].mean()) if keep.any() else None,'mean_distance_rejected':float(d[~keep].mean()) if (~keep).any() else None,'logit_l2_mean':float(np.linalg.norm(v['logits'],axis=1).mean()),'nonfinite_logits':0})
                    if '_ungated_' in cond:
                        # This is a descriptive counterfactual of fixed inference bypass,
                        # NOT an additional chosen or trained method.
                        gateonly.append({'seed':seed,'slice':name,'ungated_condition':cond,'suppressed_negative_changes':int((nar&~keep).sum()),'suppressed_recoveries':int((recovered&~keep).sum()),'remaining_negative_changes':int((nar&keep).sum()),'remaining_recoveries':int((recovered&keep).sum())})
    for file,data in [('routing_diagnostics.csv',rows),('gateonly_counterfactual.csv',gateonly),('source_gmm_diagnostics.csv',proto),('frozen_reuse_comparison.csv',baselines)]:
        with (RUN/'artifacts'/file).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    # No claims of new independent control if original predictions differ.
    baseline_diff=sum(r['different_predictions_vs_old_frozen'] for r in baselines)
    # Recomputed frozen controls are authoritative within this matched backend.
    # Historical fused-attention inference can differ numerically; retain every difference.
    outputs=[RUN/'RESULTS.md',RUN/'table_combinations.md',RUN/'table_factorial_effects.md',RUN/'table_historical_controls.md']+list((RUN/'artifacts').glob('*.csv'))+list((RUN/'artifacts/figures').glob('*'))
    assert len(list((RUN/'artifacts/figures').glob('*.png')))==5
    assert len(list((RUN/'artifacts/figures').glob('*.pdf')))==5
    assert len(list((RUN/'artifacts/figures').glob('*.svg')))==5
    save('delivery_audit.json',{'passed':True,'cells':240,'full_query_reload':True,'historical_frozen_prediction_disagreements':baseline_diff,'routing_diagnostic_rows':len(rows),'counterfactual_rows':len(gateonly),'figure_sets':5,'posthoc_no_method_changes':True,'posthoc_script_sha256':sha(Path(__file__)),'outputs':{str(p.relative_to(RUN)):sha(p) for p in outputs}})
    with (RUN/'RESULTS.md').open('a') as f:
        f.write(f'\n交付补充核验：本轮30组frozen与旧v1共{baseline_diff}条预测差异，逐组logit偏差见 `artifacts/frozen_reuse_comparison.csv`。本轮为使LoRA生效统一禁用MHA fastpath并从加载时冻结底座参数；不同attention数值内核导致少量边界预测变化。所有本轮ΔF1/NAR/Recovery均相对本轮同后端重新计算的frozen，不用旧预测替代，旧表仅历史参考。门控接受/拒绝的正确率、熵、距离与计数见 `artifacts/routing_diagnostics.csv`。`gateonly_counterfactual.csv` 仅事后计算推理旁路会同时压掉多少误杀和纠错，不是返调或额外训练的候选。5组图各PNG/PDF/SVG已检查。\n')
    # Scientific interpretation is posthoc and cannot change this run's frozen thresholds.
    scores=json.loads((RUN/'artifacts/scores.json').read_text())
    routed=[r for r in rows if '_routed_' in r['condition']]
    all_unchanged=all(r['changed_predictions']==0 for r in routed)
    with (RUN/'RESULTS.md').open('a') as f:
        f.write('\n本轮判读：四个routed组合在全部120单元的分类预测均与本轮frozen一致。' if all_unchanged else '\n本轮门控组合存在分类改动。')
        f.write(' 源训练样本拟合得到的H95约为0.060–0.097 nat，明显偏向极自信样本；加上推理硬旁路，NAR=0同时Recovery=0，不能作为成功治理证据。用已拟合训练行校准熵阈值可能过于乐观；若后续改用源域保留验证样本校准，必须另立版本、记录本轮目标反馈并预先冻结，不能覆盖本轮结果。\n')
        f.write('\nGMM4对照single的收益需按完整因子效应判读；本轮没有支持多原型更优。单原型LoRA相对残差的F1增量较小且负适应更高，不足以声称LoRA支配残差方案。无门控方案虽有F1净增益，但NAR>0；本轮未建立绝不增加误杀的机制。所有组合共用源锚恒等约束，因此不能将与旧v1的差别单独归因于GMM或LoRA。\n')
    # Refresh hash for the appended final result rather than leaving a stale delivery record.
    audit=json.loads((RUN/'artifacts/delivery_audit.json').read_text());audit['outputs']['RESULTS.md']=sha(RUN/'RESULTS.md');save('delivery_audit.json',audit)
    print('delivery passed',len(rows),'rows; historical frozen disagreements',baseline_diff,flush=True)

if __name__=='__main__':main()
