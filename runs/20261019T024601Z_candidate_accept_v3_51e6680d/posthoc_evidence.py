"""Exploratory fixed direct-reranking controls, derived from frozen predictions/evidence.
Added after method freeze for interpretation only. No method/threshold selection or training.
"""
import csv,json
from collections import defaultdict
import numpy as np
import torch
from runtime import RUN,C,sha,save,metrics,verify
from candidate import target_distribution

def factor(name):return 'Version' if name.startswith('version') else 'Network' if name.startswith('network') else 'Behavior'
def aggregate(rs,metric):
    d=defaultdict(list)
    for r in rs:d[factor(r['slice'])].append(r[metric])
    return float(np.mean([np.mean(v) for v in d.values()]))

def main():
    torch.set_num_threads(2);verify();assert json.loads((RUN/'artifacts/postrun_audit.json').read_text())['passed']
    rows=[]
    for seed in C['seeds']:
        for name in C['slices']:
            with np.load(RUN/'artifacts'/f'base_{seed}_{name}.npz') as v:z=v['logits'];base=v['predictions'];y=v['labels'];ids=v['rows']
            with np.load(RUN/'artifacts'/f'evidence_{seed}_{name}.npz') as v:votes=v['votes']
            q=target_distribution(torch.from_numpy(z),torch.from_numpy(votes),'soft_neighbor').numpy()
            predictions={'direct_candidate_rerank':q.argmax(-1),'direct_source_knn':votes.mean(0).argmax(-1)}
            bf=metrics(y,base,base)['macro_f1']
            for method,p in predictions.items():
                row={'seed':seed,'slice':name,'condition':method,**metrics(y,p,base)};row['delta_f1_pp']=100*(row['macro_f1']-bf);row['changes']=int((p!=base).sum());row['posthoc_exploratory']=True;rows.append(row)
                np.savez_compressed(RUN/'artifacts'/f'posthoc_{seed}_{name}_{method}.npz',predictions=p,labels=y,rows=ids)
    summary=[]
    for method in ['direct_candidate_rerank','direct_source_knn']:
        row={'condition':method}
        for metric in ['macro_f1','accuracy','delta_f1_pp','nar','recovery']:
            v=np.array([aggregate([r for r in rows if r['seed']==seed and r['condition']==method],metric) for seed in C['seeds']])*(1 if metric=='delta_f1_pp' else 100)
            row[metric+'_mean']=float(v.mean());row[metric+'_sd']=float(v.std(ddof=1))
        summary.append(row)
    for filename,rs in [('posthoc_direct_cells.csv',rows),('posthoc_direct_summary.csv',summary)]:
        with (RUN/'artifacts'/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    save('posthoc_direct_summary.json',summary)
    lines=['# 事后直接重排对照','', '此诊断在训练配方冻结后添加，使用相同已冻结的Top5、T2和源邻域投票公式；没有新训练、阈值搜索或目标返调。它不是预注册比较。所有3seed×10slice完整报告。','', '| 直接推理方法 | F1 % | ΔF1 pp | NAR % | Recovery % |','|---|---:|---:|---:|---:|']
    for row in summary:lines.append('| '+row['condition']+' | '+' | '.join(f"{row[m+'_mean']:.3f}±{row[m+'_sd']:.3f}" for m in ['macro_f1','delta_f1_pp','nar','recovery'])+' |')
    lines+=['','direct_candidate_rerank直接取本轮soft_neighbor固定目标分布argmax；direct_source_knn直接取两bank源邻居平均投票argmax（不限制Top5）。用于检查适应是否超过简单的源证据重排，不据此重写本轮训练或冒充新确认。']
    (RUN/'table_posthoc_direct.md').write_text('\n'.join(lines)+'\n')
    with (RUN/'RESULTS.md').open('a') as f:
        f.write('\n补充 [事后直接重排对照](table_posthoc_direct.md)：使用同一固定源邻域证据、无需任何适应，检查复杂对齐是否优于直接推理。这是配方冻结后补充的探索性诊断，不参与本轮训练或阈值选择。\n')
        f.write('\n验收可行性：三个seed在source valid均无满足预定风险门槛的邻域验收阈值，所以evidence_accept在目标上采用预注册的全部回退；其零NAR与零Recovery是分支不可用的结果，不能算成功治理。valid_entropy是另一个可用分支，阈值按源验证分位数冻结。\n')
        f.write('\n计数说明：cells中的tent_bn_reference的steps=0与parameters=0表示该保存状态未另做优化且推理时参数已冻结；其训练明确复用对应tent_bn的400步及同一BN affine参数，不能解读为无训练获得效果。630状态中180个evidence_accept是上述不可用分支的冻结回退，不是180次新适应。\n')
    delivery=json.loads((RUN/'artifacts/delivery_audit.json').read_text())
    files=[RUN/'RESULTS.md',RUN/'table_posthoc_direct.md',RUN/'artifacts/posthoc_direct_cells.csv',RUN/'artifacts/posthoc_direct_summary.csv']+list((RUN/'artifacts').glob('posthoc_*.npz'))
    for p in files:delivery['outputs'][str(p.relative_to(RUN))]=sha(p)
    delivery['posthoc_direct_control']={'passed':True,'states':60,'script_sha256':sha(RUN/'posthoc_evidence.py'),'method_or_threshold_changed':False,'pre_registered':False}
    save('delivery_audit.json',delivery)
    print((RUN/'table_posthoc_direct.md').read_text(),flush=True)

if __name__=='__main__':main()
