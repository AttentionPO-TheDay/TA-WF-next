import json
import numpy as np
from runtime import RUN, C, save, csvout


def main():
    cells = json.loads((RUN/'artifacts/cells.json').read_text())
    matched = json.loads((RUN/'artifacts/matched_frontiers.json').read_text())
    policies = json.loads((RUN/'artifacts/source_policy_target.json').read_text())
    summary = []
    for sl in C['slices']:
        for score in C['scores']:
            rr = [r for r in cells if r['slice'] == sl and r['score'] == score]
            out = dict(slice=sl, score=score)
            for k in ['pauc_norm', 'step_mean_recovery', 'rec_at_nar1', 'attained_nar1', 'diag_f1', 'corrected', 'damaged', 'other_wrong_accepted', 'other_per_correction']:
                vv = [r[k] for r in rr]
                out[k+'_mean'] = float(np.mean(vv))
                out[k+'_sd'] = float(np.std(vv, ddof=1))
            summary.append(out)
    decisions = []
    for sl in C['slices']:
        b = next(r for r in summary if r['slice'] == sl and r['score'] == 'v12_primary')
        p = next(r for r in summary if r['slice'] == sl and r['score'] == C['primary'])
        mm = [r for r in matched if r['slice'] == sl and r['score'] == C['primary'] and r['mode'] == 'v12_corrections']
        feasible = all(r['feasible'] for r in mm)
        ratio = float(np.mean([r['W'] for r in mm]))/b['other_wrong_accepted_mean'] if feasible else None
        dchange = float(np.mean([r['D'] for r in mm]))-b['damaged_mean'] if feasible else None
        risk = [r for r in matched if r['slice'] == sl and r['score'] == C['primary'] and r['mode'] == 'v12_actual_damage']
        gain = 100*(float(np.mean([r['Recovery'] for r in risk]))-b['rec_at_nar1_mean'])
        pauc = p['pauc_norm_mean']-b['pauc_norm_mean']
        rec = 100*(p['rec_at_nar1_mean']-b['rec_at_nar1_mean'])
        clean = feasible and ratio <= C['gates']['same_correction_W_ratio_max'] and dchange <= 0
        recovery = gain >= C['gates']['same_damage_rec_gain_pp']
        passed = pauc >= C['gates']['pauc_gain_min'] and rec >= -C['gates']['rec1_loss_tolerance_pp'] and (clean or recovery)
        decisions.append(dict(slice=sl, passed=bool(passed), pauc_norm_gain=pauc, rec1_gain_pp=rec, same_damage_rec_gain_pp=gain, matched_correction_feasible=feasible, matched_correction_W_ratio=ratio, matched_correction_D_change=dchange, cleanup_gate=bool(clean), recovery_gate=bool(recovery)))
    pp = [r for r in policies if r['score'] == C['primary']]
    decision = dict(primary=C['primary'], primary_passed=all(r['passed'] for r in decisions), slices=decisions, source_threshold_all_target_NAR_le_1pct_and_positive_recovery=all(r['NAR'] <= .01 and r['Recovery'] > 0 for r in pp), query_thresholds_diagnostic_only=True, integrated=False)
    save('summary.json', summary)
    csvout('summary.csv', summary)
    save('registered_decisions.json', decision)
    lines = ['# 源域候选确认重排 v14', '', f'预注册主判据：{decision["primary_passed"]}。仅训练12个微型线性重排器，主网络、LoRA和辅助网络冻结；v9候选不变。', '', '## 低NAR诊断前沿', '', '下表为三个源seed均值。pAUC/.01 是FPR≤.01区间平均TPR，FPR分母为所有frozen正确样本，等于候选阈值NAR；不是McClish标准化。', '', '|切片|评分|pAUC/.01 %|Recovery@NAR1 %|平均Recovery %|W|W/R|', '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        vals = [100*r['pauc_norm_mean'], 100*r['rec_at_nar1_mean'], 100*r['step_mean_recovery_mean'], r['other_wrong_accepted_mean'], r['other_per_correction_mean']]
        lines.append('|'+'|'.join([r['slice'], r['score']]+[f'{v:.3f}' for v in vals])+'|')
    lines += ['', '## 同v12纠错数 / 同实际损伤预算', '', '|切片|匹配|评分|可行seed|R|D|W|', '|---|---|---|---:|---:|---:|---:|']
    for sl in C['slices']:
        for mode in ['v12_corrections', 'v12_actual_damage']:
            for score in ['v12_primary', 'margin_control', C['primary']]:
                rr = [r for r in matched if r['slice'] == sl and r['mode'] == mode and r['score'] == score and r['feasible']]
                vals = [float(np.mean([r[k] for r in rr])) for k in ['R', 'D', 'W']] if rr else [None]*3
                lines.append('|'+'|'.join([sl, mode, score, str(len(rr))]+[f'{v:.3f}' if v is not None else 'NA' for v in vals])+'|')
    lines += ['', '## 源验证阈值直接迁移', '', '源阈值只使用816条已观察valid的五种合成输入校准，目标不重新选阈值。各合成输入不是独立trace。以下报告目标实际NAR和Recovery，不声称风险保证。', '', '|切片|评分|目标NAR均值 %|目标NAR最大 %|Recovery均值 %|W均值|', '|---|---|---:|---:|---:|---:|']
    for sl in C['slices']:
        for score in C['scores']:
            rr = [r for r in policies if r['slice'] == sl and r['score'] == score]
            vals = [100*np.mean([r['NAR'] for r in rr]), 100*max(r['NAR'] for r in rr), 100*np.mean([r['Recovery'] for r in rr]), np.mean([r['W'] for r in rr])]
            lines.append('|'+'|'.join([sl, score]+[f'{v:.3f}' for v in vals])+'|')
    lines += ['', '## 预注册判决', '', '```json', json.dumps(decision, ensure_ascii=False, indent=2), '```', '', '## 成本与信息边界', '', '源train固定24行/类，共2448行；valid固定8行/类，共816行。主模型已用整个18553行训练，元训练并非模型外预测。合成扰动保持网站标签是建模假设，强截尾不保证可识别；源阈值不保证真实域风险。辅助三seed×五视图仍有额外推理成本，本轮只是复用其目标缓存，不能称方法零开销。', '', '目标仍为已观察Version045/USA诊断切片，标签身份沿用内容支持假设，不是官方映射；Version/Network数据事实及外部数据角色不变。query标签未用于拟合任何参数；本轮的方法设计受v13目标诊断启发，不能称独立确认。', '', '主评分未因对照更好而替换；所有24目标评分状态、12源特征状态、各seed及所有源阈值结果均保留。全局AUROC不用于主验收。目标诊断前沿的阈值由query标签选择，不能部署；源阈值策略与之明确分开。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(RUN/'artifacts/frontiers.npz') as zz:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        grid = np.linspace(0, .01, 201)
        for ax, sl in zip(axes, C['slices']):
            for score in C['scores']:
                ys = []
                for seed in C['seeds']:
                    rr = next(r for r in cells if r['seed'] == seed and r['slice'] == sl and r['score'] == score)
                    cr = zz[f'{seed}_{sl}_{score}']
                    x, y = cr[:, 2]/rr['frozen_correct'], cr[:, 1]/rr['frozen_wrong']
                    last = np.flatnonzero(np.r_[x[1:] != x[:-1], True])
                    xx, yy = x[last], y[last]
                    ys.append(yy[np.maximum(np.searchsorted(xx, grid, side='right')-1, 0)])
                ax.step(100*grid, 100*np.mean(ys, 0), where='post', label=score)
            ax.set(title=sl, xlabel='NAR budget (%)', ylabel='Recovery (%)', xlim=(0, 1))
            ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(RUN/f'figure_frontier.{ext}', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
