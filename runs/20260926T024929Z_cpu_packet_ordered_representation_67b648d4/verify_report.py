"""CPU checkpoint replay, independent metrics, and automatic experiment closure."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
import torch
import train
from prepare import sha
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
MARKER='当前CPU首轮实验（2026-09-26）：`'+RUN.name+'`。'


def independent_metric(y,p):
    f=[]
    for label in range(102):
        tp=np.count_nonzero((y==label)&(p==label))
        den=np.count_nonzero(y==label)+np.count_nonzero(p==label)
        f.append(2*tp/den if den else 0.)
    return {'accuracy':float(np.mean(y==p)),'macro_f1':float(np.mean(f))}


def verify():
    config,config_sha=train.load_config()
    torch.set_num_threads(2)
    data=torch.load(RUN/'artifacts/prepared.pt',map_location='cpu',weights_only=False)
    assert set(data)=={'source','valid'}
    assert sha(ROOT/config['sampling_manifest'])==config['sampling_manifest_sha256']
    audit=json.loads((RUN/'artifacts/input_audit.json').read_text())
    assert audit['prepared_sha256']==config['prepared_sha256']
    for role,entry in data.items():
        assert sha(RUN/'artifacts/prepared.pt')==config['prepared_sha256']
        assert __import__('hashlib').sha256(entry['directions'].numpy().tobytes()).hexdigest()==audit['roles'][role]['selected_direction_sha256']
    errors=[];reports=[];checked=0;initials={};parameters=[]
    for condition in config['conditions']:
        batches={role:train.make_batch(entry,condition) for role,entry in data.items()}
        for seed in config['seeds']:
            key=f'{condition}_{seed}'
            try:
                report=json.loads((RUN/'artifacts'/f'metrics_{key}.json').read_text());reports.append(report)
                assert report['config_sha256']==config_sha
                assert report['status']=='completed' and report['epochs_completed']==config['epochs'], 'seed incomplete'
                assert report['device']=='cpu' and report['threads']==config['threads_per_worker']
                history=json.loads((RUN/'artifacts'/f'history_{key}.json').read_text())['history']
                assert [row['epoch'] for row in history]==list(range(1,config['epochs']+1))
                evaluated=[row for row in history if 'valid' in row]
                assert [row['epoch'] for row in evaluated]==list(range(5,101,5))
                best=max(evaluated,key=lambda row:row['valid']['macro_f1'])
                assert best['epoch']==report['best_epoch']
                state=torch.load(RUN/'checkpoints'/f'{key}_best.pt',map_location='cpu',weights_only=True)
                assert state['config_sha256']==config_sha and state['epoch']==best['epoch']
                model=train.make_model(config);model.load_state_dict(state['state_dict'])
                parameters.append(sum(p.numel() for p in model.parameters()))
                initials[key]=report['initial_state_sha256']
                torch.manual_seed(seed);original=train.make_model(config)
                assert train.state_hash(original)==initials[key]
                with np.load(RUN/'artifacts'/f'predictions_{key}.npz',allow_pickle=False) as saved:
                    assert set(saved.files)=={'source','valid'}
                    for role in ('source','valid'):
                        predictions=train.predict(model,batches[role],config['batch_size'])
                        assert np.array_equal(predictions,saved[role]),role+' prediction replay'
                        metrics=independent_metric(data[role]['labels'].numpy(),predictions)
                        for name,value in metrics.items():
                            assert abs(value-report[role][name])<1e-12,role+' '+name
                            assert abs(value-best[role][name])<1e-12,role+' history '+name
                        checked+=1
                last=torch.load(RUN/'checkpoints'/f'{key}_latest.pt',map_location='cpu',weights_only=False)
                assert last['epoch']==100 and last['config_sha256']==config_sha
                assert 'optimizer' in last and 'torch_rng' in last and 'numpy_rng' in last and 'python_rng' in last
            except Exception as error:
                errors.append(key+': '+repr(error))
                print(traceback.format_exc(),flush=True)
    for seed in config['seeds']:
        if initials.get(f'summary_{seed}')!=initials.get(f'ordered_{seed}'):
            errors.append(f'initial states differ for {seed}')
    if len(set(parameters))!=1:errors.append('parameter counts differ or missing')
    result={'prediction_sets_checked':checked,'expected_prediction_sets':12,'errors':errors,
            'complete':checked==12 and not errors,'parameters':parameters,
            'source_valid_overlap':audit['source_valid_direction_overlap'],'future_access':False,'device':'cpu'}
    train.atomic_json(RUN/'artifacts/integrity.json',result)
    return config,reports,result


def close(config,reports,integrity):
    lines=['# 实验结果','',f"状态：{'completed' if integrity['complete'] else 'stopped / incomplete'}；CPU两条件×三seed，source2040/valid510，方向5000包，100epochs。",'',
           '| 条件 | seed | 完成状态 | best epoch | train accuracy | valid accuracy | valid Macro-F1 | 秒 |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for condition in config['conditions']:
        for seed in config['seeds']:
            row=next((r for r in reports if r['condition']==condition and r['seed']==seed),None)
            if row is None or 'valid' not in row:
                lines.append(f'| {condition} | {seed} | missing/no score | — | — | — | — | — |');continue
            lines.append(f"| {condition} | {seed} | {row['status']} | {row['best_epoch']} | {100*row['source']['accuracy']:.3f}% | {100*row['valid']['accuracy']:.3f}% | {100*row['valid']['macro_f1']:.3f}% | {row['elapsed_seconds']:.1f} |")
    if integrity['complete']:
        means={c:{m:float(np.mean([r['valid'][m] for r in reports if r['condition']==c])) for m in ('accuracy','macro_f1')} for c in config['conditions']}
        deltas=[]
        for seed in config['seeds']:
            pair={r['condition']:r for r in reports if r['seed']==seed}
            deltas.append(100*(pair['ordered']['valid']['macro_f1']-pair['summary']['valid']['macro_f1']))
        fdelta=100*(means['ordered']['macro_f1']-means['summary']['macro_f1'])
        adelta=100*(means['ordered']['accuracy']-means['summary']['accuracy'])
        passed=fdelta>=1 and all(d>0 for d in deltas) and adelta>=0
        decision='PASS_CANDIDATE' if passed else 'FAIL_CANDIDATE'
        train.atomic_json(RUN/'artifacts/summary.json',{'means':means,'paired_f1_delta_pp':deltas,'mean_f1_delta_pp':fdelta,'mean_accuracy_delta_pp':adelta,'decision':decision})
        lines+=['',f"summary/ordered 三seed均值：accuracy {100*means['summary']['accuracy']:.3f}/{100*means['ordered']['accuracy']:.3f}%；Macro-F1 {100*means['summary']['macro_f1']:.3f}/{100*means['ordered']['macro_f1']:.3f}%。",
                f'ordered-summary F1差 {fdelta:+.3f}pp；逐seed差 '+', '.join(f'{d:+.3f}' for d in deltas)+'pp；accuracy差 '+f'{adelta:+.3f}pp。',
                f'预定门槛（平均F1至少+1pp、3/3正、accuracy不下降）：{decision}。']
        summary=f"CPU保序表示完成：summary/ordered valid F1 {100*means['summary']['macro_f1']:.3f}/{100*means['ordered']['macro_f1']:.3f}%，差{fdelta:+.3f}pp，{sum(d>0 for d in deltas)}/3 seed正，{decision}；12/12重载核验通过。"
        if passed:
            lines+=['','结论：支持保留包内方向细节作为候选输入改进；不能把收益唯一归因于顺序，也不代表达到DF水准或抗漂移。下一步先结合训练—验证差距确定是否需要优化/泛化诊断，再单独验证run/window增量，不自动启动下一轮。']
        else:
            lines+=['','结论：该受控表示改动未通过预定候选门槛，不支持直接推进后续预训练或TTA。报告保留全部正负seed；不能由一轮否定所有保序编码或Transformer。下一步优先根据训练拟合情况区分优化/汇聚瓶颈与泛化，另立有界方案。']
        state='completed'
    else:
        lines+=['','完整对照未通过核验，不作稳定性或候选通过结论。详见 artifacts/integrity.json 和 logs/。']
        summary=f"CPU保序表示未完整收口：{integrity['prediction_sets_checked']}/12预测核验，{len(integrity['errors'])}项缺失/错误；须先审计，不追加预算。"
        state='stopped'
    lines+=['',f"核验：{integrity['prediction_sets_checked']}/12 source/valid预测与checkpoint、独立指标、最早最佳epoch复算；errors={integrity['errors']}。",
            '', '输入/代码版本见 config.json、artifacts/freeze.json、artifacts/input_audit.json。两条件均118870参数、同seed初始化一致、run/window不变；增加的是packet输入细节，摘要对照具有更低有效输入秩。每轮保存latest及按valid改进保存best。',
            '', '历史DF accuracy49.150%/F1 48.144%只是不同架构及训练/选模预算的性能参照，本轮未重训DF/RF。仅TemporalDrift已观察开发数据；未来日期、WTT/AWF、PCAP、预训练、蒸馏、adapter/TTA均未使用。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/experiment.py'),'status','--id',RUN.name,'--state',state,'--summary',summary],check=True)
    path=ROOT/'STATUS.md';text=path.read_text()
    if any(line.startswith(MARKER) for line in text.splitlines()):
        text='\n'.join(MARKER+summary if line.startswith(MARKER) else line for line in text.splitlines())+'\n'
        path.write_text(text)
    print(summary,flush=True)

if __name__=='__main__':
    config,reports,integrity=verify();close(config,reports,integrity)
    if not integrity['complete']:raise SystemExit(1)
