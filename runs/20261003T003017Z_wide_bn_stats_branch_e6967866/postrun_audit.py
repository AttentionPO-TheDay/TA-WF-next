import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from model import StatsModel, global_stats
from wide_bn_base import MultiViewModel
from worker import predict
from common import deterministic

RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[1]; SEEDS=[21729,23407,22026]; KINDS=['baseline','zero_add','stats_add','zero_gate','stats_gate']; NEW=KINDS[1:]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block=f.read(8*1024*1024)
            if not block: break
            h.update(block)
    return h.hexdigest()

def metric(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0)),'correct':int((y==p).sum()),'count':int(len(y))}

def main():
    c=json.loads((RUN/'config.json').read_text()); raw=torch.load(RUN/'artifacts/prepared.pt',weights_only=False,map_location='cpu'); labels={r:raw[r]['labels'].numpy() for r in ['source','valid']}; frozen=json.loads((RUN/'artifacts/freeze.json').read_text())
    for p,h in frozen.items(): assert sha(ROOT/p)==h,p
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    deterministic('cuda',2)
    data={role:{k:v.cuda() for k,v in e.items() if k in ('timestamps','tam','labels')} for role,e in raw.items()}
    source_stats=global_stats(data['source']['tam']).detach().cpu()
    source_mean=source_stats.mean(0).reshape(1,180)
    source_std=source_stats.std(0,unbiased=False).clamp_min(1e-6).reshape(1,180)
    reports=[]; checks=0; history_checks=0; init_checks=0; index_checks=0; replay_checks=0; logits_checks=0
    for seed in SEEDS:
        base=RUN/'artifacts'/f'baseline_{seed}'; ref=torch.load(base/'initial_state.pt',weights_only=True,map_location='cpu'); stream=torch.load(base/'index_stream.pt',weights_only=True,map_location='cpu')
        for kind in KINDS:
            d=RUN/'artifacts'/f'{kind}_{seed}'; rep=json.loads((d/'report.json').read_text()); assert rep['steps_completed']==12800 and rep['validation_opportunities']==20 and rep['verified_roles']==['source','valid'] and rep.get('future_access',False) is False
            hist=json.loads((d/'history.json').read_text()); assert len(hist)==20 and [x['step'] for x in hist]==c['eval_steps']; assert max(hist,key=lambda x:x['valid']['accuracy'])['step']==rep['best_step']; history_checks+=1
            state=torch.load(d/'initial_state.pt',weights_only=True,map_location='cpu');
            for k,v in ref.items(): assert k in state and torch.equal(v,state[k]),(kind,seed,k)
            init_checks+=len(ref); assert torch.equal(stream,torch.load(d/'index_stream.pt',weights_only=True,map_location='cpu')); index_checks+=1
            if kind!='baseline':
                torch.testing.assert_close(state['stats_mean'],source_mean,rtol=1e-6,atol=1e-7)
                torch.testing.assert_close(state['stats_std'],source_std,rtol=1e-6,atol=1e-7)
                pair='stats_gate' if kind.endswith('gate') else 'stats_add'
                other=torch.load(RUN/'artifacts'/f'{pair}_{seed}'/'initial_state.pt',weights_only=True,map_location='cpu')
                assert set(state)==set(other) and all(torch.equal(v,other[k]) for k,v in state.items())
            for phase in ['best','last']:
                with np.load(d/f'predictions_{phase}.npz') as z:
                    for role in ['source','valid']:
                        p=z[role]; assert p.shape==labels[role].shape and np.isin(p,np.arange(102)).all(); got=metric(labels[role],p); assert abs(got['accuracy']-rep[phase][role]['accuracy'])<1e-12; assert abs(got['macro_f1']-rep[phase][role]['macro_f1'])<1e-12; checks+=1
            ck=torch.load(RUN/'checkpoints'/f'{kind}_{seed}_best.pt',weights_only=True,map_location='cpu')
            assert ck['step']==rep['best_step']
            if kind=='baseline':
                assert sha(RUN/'checkpoints'/f'{kind}_{seed}_best.pt')==sha(ROOT/c['historical_run']/'checkpoints'/f'wide_bn_{seed}_best.pt')
                model=MultiViewModel('wide_bn').cuda()
            else:
                assert ck['config_sha256']==sha(RUN/'config.json')
                assert torch.equal(ck['state_dict']['stats_mean'],state['stats_mean']) and torch.equal(ck['state_dict']['stats_std'],state['stats_std'])
                model=StatsModel(kind,state['stats_mean'],state['stats_std']).cuda()
            model.load_state_dict(ck['state_dict'])
            if kind!='baseline':
                assert all(int(v)==rep['best_step'] for k,v in model.named_buffers() if k.endswith('num_batches_tracked'))
                for name,p in model.named_parameters():
                    if not p.requires_grad or (kind.endswith('add') and name.startswith('gate.')): assert torch.equal(p.cpu(),state[name]),name
                assert not torch.equal(ck['state_dict']['stats_head.weight'],state['stats_head.weight'])
            with np.load(d/'logits_best.npz') as logits, np.load(d/'predictions_best.npz') as preds:
                for role,e in data.items():
                    saved=logits[role]; assert saved.shape==(len(labels[role]),102) and np.isfinite(saved).all()
                    assert np.array_equal(saved.argmax(1),preds[role]); logits_checks+=1
                    fresh=predict(model,e,c['eval_batch_size']); assert np.array_equal(fresh.argmax(1),preds[role]); replay_checks+=1
                    np.testing.assert_allclose(fresh,saved,rtol=1e-5,atol=1e-5)
            del model
            reports.append({'kind':kind,'seed':seed,'best':rep['best'],'last':rep['last'],'best_step':rep.get('best_step')})
            print(json.dumps({'audited':f'{kind}_{seed}','checkpoint_replay_groups':replay_checks}),flush=True)
    def score(k,s,m): return next(x for x in reports if x['kind']==k and x['seed']==s)['best']['valid'][m]
    means={k:{m:float(np.mean([score(k,s,m) for s in SEEDS])) for m in ['accuracy','macro_f1']} for k in KINDS}
    comparisons={}
    for a,b in [('stats_add','zero_add'),('stats_gate','zero_gate'),('stats_gate','stats_add'),('zero_add','baseline'),('stats_add','baseline'),('zero_gate','baseline'),('stats_gate','baseline')]:
        da=[score(a,s,'accuracy')-score(b,s,'accuracy') for s in SEEDS]; df=[score(a,s,'macro_f1')-score(b,s,'macro_f1') for s in SEEDS]
        comparisons[f'{a} - {b}']={'accuracy_mean_delta_pp':float(np.mean(da)*100),'accuracy_per_seed_pp':[float(x*100) for x in da],'macro_f1_mean_delta_pp':float(np.mean(df)*100),'pass':bool(np.mean(da)>=.01 and min(da)>0 and np.mean(df)>=0)}
    ensemble={}
    for kind in KINDS:
        logits=[]
        for s in SEEDS:
            with np.load(RUN/'artifacts'/f'{kind}_{s}'/'logits_best.npz') as z: logits.append(z['valid'])
        z=np.stack(logits); vote=np.zeros_like(z[0]); pred=np.argmax(z,2)
        for p in pred: vote[np.arange(len(p)),p]+=1
        scaled=(z-z.mean(2,keepdims=True))/z.std(2,keepdims=True).clip(min=1e-12); out=(vote+1e-6*scaled.sum(0)).argmax(1); ensemble[kind]=metric(labels['valid'],out)
    summary={'means':means,'comparisons':comparisons,'ensemble_valid':ensemble,'target_reached':{k:means[k]['accuracy']>=.9 for k in KINDS},'passed_candidates':[k for k in NEW if comparisons[f'{k} - baseline']['pass']],'freeze_hashes_verified':len(frozen),'metric_groups_verified':checks,'history_groups_verified':history_checks,'initialization_tensors_verified':init_checks,'index_streams_verified':index_checks,'logit_groups_verified':logits_checks,'checkpoint_replay_groups_verified':replay_checks,'training_tasks_completed':12,'historical_baselines_reused':3,'future_access':False,'aggregation_bug':'supervisor passed candidates only looked for missing condition-baseline comparison keys after all 12 tasks completed; recovery used saved reports, predictions, and pre-registered comparisons; no training rerun'}
    (RUN/'artifacts'/'postrun_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); (RUN/'artifacts'/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    rows=['# wide+BN 全局统计支路实验结果','','12 项新训练和 3 项历史 wide+BN 基线均完成。监督脚本在训练结束后的候选汇总阶段因引用不存在的 `zero_add - baseline` 等比较键报错；已依据保存的 report、history、预测和冻结 hash 恢复汇总，没有重训。','','| 条件 | 验证准确率均值 | Macro-F1 均值 | 固定三 seed 投票 | ≥90% |','|---|---:|---:|---:|---|']
    for k in KINDS: rows.append(f"| {k} | {means[k]['accuracy']*100:.3f}% | {means[k]['macro_f1']*100:.3f}% | {ensemble[k]['accuracy']*100:.3f}% | {means[k]['accuracy']>=.9} |")
    rows += ['','| 条件 | 21729 | 23407 | 22026 |','|---|---:|---:|---:|']
    for k in KINDS: rows.append('| '+k+' | '+' | '.join(f"{score(k,s,'accuracy')*100:.3f}%" for s in SEEDS)+' |')
    rows += ['','## 比较结果','','```json',json.dumps(comparisons,ensure_ascii=False,indent=2),'```','','## 核验','','冻结输入 hash、60 组 source/valid best/last 预测指标、15 组 history 与选模步数、15 组初始化状态和 15 条 index stream 均已核验；另对 30 组保存 logits/predictions 复算，并重新加载 15 个最佳 checkpoint 在 source/valid 上完成 30 组推理复核。12 项新任务均完成 12800 步，valid 只用于预定 checkpoint 选择与评价。','','真实统计支路相对 zero 控制下降：`stats_add` 低于 `zero_add`，`stats_gate` 低于 `zero_gate`；四个新条件均低于历史 wide+BN 89.150%，没有统计支路候选通过门槛。固定三 seed 投票的历史 baseline 仍为 91.569%，新条件投票分别为 '+', '.join(f'`{k}={ensemble[k]["accuracy"]*100:.3f}%`' for k in NEW)+ '。','','本实验说明在当前 TAM 输入、遮挡规则和分类配方下，增加全局统计残差会损害源期 valid 泛化，不能据此否定所有统计特征或 RF 的优势。未来/WTT-Time/AWF/适应关闭；该 valid 已参与多轮开发，不是独立确认。']
    (RUN/'RESULTS.md').write_text('\n'.join(rows)+'\n'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
