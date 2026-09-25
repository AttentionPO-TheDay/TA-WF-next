from __future__ import annotations
import csv, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import torch

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('experiment_run',RUN/'run.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def digest(row): return hashlib.sha256(np.sign(row).astype(np.int8).tobytes()).hexdigest()

def main():
    data=mod.load_td(); source=data['source']['p']; valid=data['valid']['p']; ext=np.load(RUN/'artifacts/pcap_sequences.npz')['packets']
    overlap=len(set(map(digest,source)) & set(map(digest,valid)))
    external_overlap=len(set(map(digest,ext)) & (set(map(digest,source))|set(map(digest,valid))))
    assert overlap==0 and external_overlap==0
    idx=data['train_idx']; counts=np.bincount(data['source']['y'][idx],minlength=102)
    assert len(idx)==510 and np.all(counts==5)
    errors=[]; checked=0; parameter_counts=[]
    for cond,device_name in mod.DEVICE_BY_CONDITION.items():
        device=torch.device(device_name)
        csv_path=RUN/'artifacts'/f'metrics_{cond}_v2.csv'
        with csv_path.open() as f: rows=list(csv.DictReader(f))
        with np.load(RUN/'artifacts'/f'predictions_{cond}_v2.npz') as archived:
            for seed in mod.SEEDS:
                cp=torch.load(RUN/'checkpoints'/f'{cond}_{seed}_v2.pt',map_location='cpu',weights_only=True)
                model=mod.Fusion().to(device).eval(); model.load_state_dict(cp['state_dict']); parameter_counts.append(sum(p.numel() for p in model.parameters()))
                for role in ('source','valid','day14','day30','day90','day150','day270'):
                    key=f'{role}_{cond}_{seed}'; row=next(r for r in rows if r['role']==role and int(r['seed'])==seed)
                    pred=mod.predict(model,data[role]['p'],data[role]['w'],device)
                    if not np.array_equal(pred,archived[key]): errors.append(key+':prediction')
                    metrics=mod.metric(data[role]['y'],pred)
                    for m in ('accuracy','macro_f1'):
                        if abs(metrics[m]-float(row[m]))>1e-12: errors.append(key+':'+m)
                    checked+=1
    assert len(set(parameter_counts))==1
    result={'prediction_sets_checked':checked,'errors':errors,'source_valid_exact_direction_overlap':overlap,'external_source_valid_exact_direction_overlap':external_overlap,'few_shot_count':len(idx),'few_shot_per_class':5,'parameters':parameter_counts[0],'pcap_sequences':len(ext)}
    (RUN/'artifacts/integrity_v2.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert not errors
if __name__=='__main__': main()
