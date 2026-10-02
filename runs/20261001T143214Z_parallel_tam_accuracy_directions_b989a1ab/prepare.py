"""Reuses source/valid only; relative-time counts require no label fitting."""
from pathlib import Path
import json
import numpy as np
import torch
from common import RUN,ROOT,save,sha,savet

def relative_tam(x):
    values=x[x!=0]
    if len(values)==0:return np.zeros((2,1800),np.float32)
    times=np.abs(values).astype(np.float64)
    bins=np.floor(times/times.max()*1799).astype(np.int64).clip(0,1799)
    code=bins+(values<0).astype(np.int64)*1800
    return np.bincount(code,minlength=3600).reshape(2,1800).astype(np.float32)

def main():
    c=json.loads((RUN/'config.json').read_text());old=ROOT/c['historical_run']
    assert sha(RUN/'artifacts/prepared.pt')==sha(old/'artifacts/prepared.pt')
    for p,h in json.loads((old/'artifacts/freeze.json').read_text()).items():assert sha(ROOT/p)==h,p
    raw=torch.load(RUN/'artifacts/prepared.pt',map_location='cpu',weights_only=False)
    assert set(raw)=={'source','valid'}
    relative={};info={}
    for role,n in [('source',15300),('valid',510)]:
        e=raw[role];assert len(e['labels'])==n
        assert np.array_equal(np.bincount(e['labels'].numpy(),minlength=102),np.full(102,n//102))
        arr=np.stack([relative_tam(x) for x in e['timestamps'].numpy()])
        assert np.array_equal(arr.sum((1,2)),e['timestamps'].ne(0).sum(1).numpy())
        assert np.array_equal(arr.sum((1,2)),e['tam'].sum((1,2)).numpy())
        relative[role]={'tam':torch.from_numpy(arr),'labels':e['labels'],'rows':e['rows']}
        info[role]={'rows':n,'count_parity':True,'label_and_row_identity':True}
    savet(RUN/'artifacts/relative.pt',relative)
    save(RUN/'artifacts/preparation.json',{'prepared_input_hash':sha(RUN/'artifacts/prepared.pt'),'relative_hash':sha(RUN/'artifacts/relative.pt'),'roles':info,'future_access':False,'raw_data_new_access':False})
    print(json.dumps(info))

if __name__=='__main__':main()
