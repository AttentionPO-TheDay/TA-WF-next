"""Report both fitting and generalization, with old/new selection separated."""
import subprocess,statistics
from common import *

def main():
    c=config_read();names=[v['id'] for v in c['conditions']];reports={};errors=[];checked=0
    for seed in c['seeds']:
        for name in names:
            key=f'{name}_{seed}';out=RUN/'artifacts'/key
            try:
                r=json.loads((out/'report.json').read_text());v=json.loads((out/'verification.json').read_text())
                assert r['status']=='completed' and v['complete'] and r['config_sha256']==sha(RUN/'config.json')
                for p,h in v['artifact_sha256'].items():assert sha(ROOT/p)==h,p
                reports[key]=r;checked+=v['prediction_sets_checked']
            except Exception as e:errors.append(f'{key}: {type(e).__name__}: {e}')
    for seed in c['seeds']:
        rr=[reports.get(f'{n}_{seed}') for n in names]
        if any(r is None for r in rr) or len({r['initial_state_sha256'] for r in rr})!=1:errors.append(f'{seed}: pairing missing/mismatch')
    complete=len(reports)==6 and checked==36 and not errors
    atomic_json(RUN/'artifacts/integrity.json',{'complete':complete,'prediction_sets_checked':checked,'expected_prediction_sets':36,'errors':errors,'future_access':False})
    lines=['# 每类80条充分训练结果','',f'状态：{"completed" if complete else "incomplete"}。明确复用前3200步，再续训9600步；新阶段20次选模。','', '| 条件 | seed | new best step | source accuracy | valid accuracy | valid F1 | last source accuracy | last valid F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        for seed in c['seeds']:
            r=reports.get(f'{name}_{seed}')
            if r:lines.append(f"| {name} | {seed} | {r['best_block']*32} | {r['source']['accuracy']*100:.3f}% | {r['valid']['accuracy']*100:.3f}% | {r['valid']['macro_f1']*100:.3f}% | {r['last']['source']['accuracy']*100:.3f}% | {r['last']['valid']['macro_f1']*100:.3f}% |")
            else:lines.append(f'| {name} | {seed} | incomplete | — | — | — | — | — |')
    summary={'complete':complete,'errors':errors}
    if complete:
        means={};changes={}
        for name in names:
            rr=[reports[f'{name}_{s}'] for s in c['seeds']]
            means[name]={}
            for stage in ['new_best','historical_best','last','resume_metrics']:
                vals=rr if stage=='new_best' else [r[stage] for r in rr]
                means[name][stage]={role:{m:statistics.mean(v[role][m] for v in vals) for m in ['accuracy','macro_f1']} for role in ['source','common_source','valid']}
            vf=[100*(r['valid']['macro_f1']-r['historical_best']['valid']['macro_f1']) for r in rr]
            va=[100*(r['valid']['accuracy']-r['historical_best']['valid']['accuracy']) for r in rr]
            sa=[100*(r['source']['accuracy']-r['historical_best']['source']['accuracy']) for r in rr]
            ef=[100*(r['last']['valid']['macro_f1']-r['resume_metrics']['valid']['macro_f1']) for r in rr]
            es=[100*(r['last']['source']['accuracy']-r['resume_metrics']['source']['accuracy']) for r in rr]
            vg=statistics.mean(vf)>=1 and all(x>0 for x in vf) and statistics.mean(va)>=0
            both=vg and statistics.mean(sa)>0
            changes[name]={'new_best_minus_old_best_valid_f1_pp':vf,'new_best_minus_old_best_valid_accuracy_pp':va,'new_best_minus_old_best_source_accuracy_pp':sa,'last_minus_resume_valid_f1_pp':ef,'last_minus_resume_source_accuracy_pp':es,'valid_improvement_gate':vg,'joint_improvement_gate':both}
            for title,arr in [('best valid F1',vf),('best source accuracy',sa),('endpoint valid F1',ef),('endpoint source accuracy',es)]:lines.extend(['',f'{name} {title}增量：逐seed{arr}，平均{statistics.mean(arr):+.3f}pp。'])
            lines.extend(['',f'{name} 验证改善门槛={vg}；训练与验证共同提高门槛={both}。'])
        left,right='D_80_mean','C_80_cls'
        df=[100*(reports[f'{left}_{s}']['valid']['macro_f1']-reports[f'{right}_{s}']['valid']['macro_f1']) for s in c['seeds']]
        da=100*(means[left]['new_best']['valid']['accuracy']-means[right]['new_best']['valid']['accuracy'])
        comparison={'left':left,'right':right,'f1_by_seed_pp':df,'mean_f1_pp':statistics.mean(df),'mean_accuracy_pp':da,'passed':statistics.mean(df)>=1 and all(x>0 for x in df) and da>=0}
        summary.update(means=means,changes=changes,readout_comparison=comparison)
        lines.extend(['','| 条件 | old best source accuracy | new best source accuracy | old best valid F1 | new best valid F1 |','|---|---:|---:|---:|---:|'])
        for name,v in means.items():lines.append(f"| {name} | {v['historical_best']['source']['accuracy']*100:.3f}% | {v['new_best']['source']['accuracy']*100:.3f}% | {v['historical_best']['valid']['macro_f1']*100:.3f}% | {v['new_best']['valid']['macro_f1']*100:.3f}% |")
        lines.extend(['',f'D−C valid F1平均{statistics.mean(df):+.3f}pp，逐seed{df}，门槛{comparison["passed"]}。'])
        values='/'.join(f"{means[n]['new_best']['valid']['macro_f1']*100:.3f}" for n in names)
        gates={n:changes[n]['joint_improvement_gate'] for n in names}
        message=f'80条充分训练完成：CLS/mean valid F1 {values}%；共同提高门槛{gates}；36/36预测核验。'
    else:message=f'80条充分训练未完整通过：{checked}/36预测核验，保留产物，不追加预算。'
    lines.extend(['',message,'',f'错误：{errors}。','', '限制：复用本项目已训练3200步及同一valid反馈；新增9600步和学习率下降共同改变训练流程，不能将收益单独归因于训练时长。旧best和新best分别来自20次选模，旧结果是历史对照，不是独立重复。固定末步对照与完整曲线用于检查过拟合/收敛，不能凭训练高分宣布泛化成功。无新数据划分、未来/外部数据、预训练或TTA。','', '模型/优化器/RNG恢复、采样前缀、步数/LR/选模和best/latest预测独立核验见artifacts；所有代码/数据/锚点散列已冻结。'])
    atomic_json(RUN/'artifacts/summary.json',summary);(RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state','completed' if complete else 'stopped','--summary',message],check=True)
    p=ROOT/'STATUS.md';marker='当前充分训练实验：`'+RUN.name+'`。'
    p.write_text('\n'.join(marker+message if line.startswith(marker) else line for line in p.read_text().splitlines())+'\n')
    print(message,flush=True)
    if not complete:raise SystemExit(2)
if __name__=='__main__':main()
