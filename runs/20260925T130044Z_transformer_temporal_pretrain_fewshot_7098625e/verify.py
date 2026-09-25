from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN))
import train


def main():
    prepared=train.load_prepared()
    labels=prepared['source']['labels'].numpy()
    fewshot=prepared['fewshot'].numpy()
    counts=np.bincount(labels[fewshot],minlength=102)
    assert len(fewshot)==510 and np.all(counts==5)
    assert len(prepared['source_pretrain'])==2040
    conditions=('packet_scratch','multiview_scratch','multiview_pretrained')
    metric_rows=[]
    for condition in conditions:
        with (RUN/'artifacts'/f'metrics_{condition}.csv').open() as f:
            metric_rows.extend(csv.DictReader(f))
    with (RUN/'artifacts/metrics_future.csv').open() as f:
        metric_rows.extend(csv.DictReader(f))
    assert len(metric_rows)==63
    histories={condition:json.loads((RUN/'artifacts'/f'history_{condition}.json').read_text()) for condition in conditions}
    prediction_files={condition:np.load(RUN/'artifacts'/f'predictions_{condition}.npz') for condition in conditions}
    future_predictions=np.load(RUN/'artifacts/predictions_future.npz')
    device=torch.device('cuda:0')
    errors=[]; checked=0; parameter_counts=[]
    for role in train.ROLES:
        if role in ('source','valid'):
            batches={condition:train.batch_for(prepared,role,condition)[0] for condition in conditions}
            true=prepared[role]['labels'].numpy()
        else:
            sampling=json.loads(train.PRIOR.read_text())['sampling'][role]
            path=train.DATA_ROOT/sampling['path']
            with np.load(path,allow_pickle=False) as archive:
                x=archive['X'][sampling['rows']].astype(np.float32)
            full=train.views_to_batch(x)
            batches={condition:train.packet_only(full) if condition=='packet_scratch' else full for condition in conditions}
            with np.load(path,allow_pickle=False) as archive:
                true=archive['y'][sampling['rows']].astype(np.int64)
        for condition in conditions:
            for seed in train.SEEDS:
                key=f'{role}_{condition}_{seed}'
                checkpoint=torch.load(RUN/'checkpoints'/f'{condition}_{seed}.pt',map_location='cpu',weights_only=True)
                model=train.make_model().to(device).eval(); model.load_state_dict(checkpoint['state_dict'])
                parameter_counts.append(sum(p.numel() for p in model.parameters()))
                prediction=train.predict(model,batches[condition],device)
                saved=(prediction_files[condition] if role in ('source','valid') else future_predictions)[key]
                if not np.array_equal(prediction,saved): errors.append(key+':prediction')
                row=next(r for r in metric_rows if r['role']==role and r['condition']==condition and int(r['seed'])==seed)
                for metric_name,value in train.metric(true,prediction).items():
                    if abs(value-float(row[metric_name]))>1e-12: errors.append(key+':'+metric_name)
                if int(row['best_epoch'])!=checkpoint['best_epoch']: errors.append(key+':epoch')
                checked+=1
    for condition in conditions:
        for seed in train.SEEDS:
            key=f'{condition}_{seed}'; history=histories[condition][key]['history']
            maximum=max(x['macro_f1'] for x in history)
            earliest=next(x['epoch'] for x in history if x['macro_f1']==maximum)
            if earliest!=histories[condition][key]['best_epoch']: errors.append(key+':selection')
    for file in prediction_files.values(): file.close()
    future_predictions.close()
    assert len(set(parameter_counts))==1
    report={'prediction_sets_checked':checked,'errors':errors,'fewshot_total':len(fewshot),
            'fewshot_per_class':int(counts.min()),'source_unlabeled_pretrain':len(prepared['source_pretrain']),
            'parameters':parameter_counts[0],'source_valid_roles_only_prepared':set(prepared)=={'source','valid','fewshot','source_pretrain'}}
    (RUN/'artifacts/integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert not errors

if __name__=='__main__': main()
